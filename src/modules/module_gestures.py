"""
module_gestures.py

Conversational micro-gestures for TARS-AI.

Quick, expressive body movements (~0.5-1.5s) that run DURING speech
to make TARS feel alive. These are distinct from full movements
(walk, dance, wave) — they're subtle conversational body language.

Key design decisions:
  - Do NOT set servoctl.MOVING or call _notify_movement_start/end.
    These micro-gestures are lightweight and must not pause UI or STT.
  - Use move_legs() directly with small percentage changes from neutral (50).
  - Always return to neutral at the end.
  - Skip silently if a full movement is already in progress.
  - Run in a background thread — never block the TTS pipeline.

The LLM decides when to gesture via an optional "gesture" field in its
JSON response. It may omit the field or set it to null when no gesture
fits the moment.
"""

import time
import threading

from modules.module_messageQue import queue_message

# Lazy import to avoid circular dependency at module load time
_servoctl = None

def _get_servoctl():
    global _servoctl
    if _servoctl is None:
        import modules.module_servoctl as sc
        _servoctl = sc
    return _servoctl


def _move(left_h=None, right_h=None, left_l=None, right_l=None, speed=0.9):
    """Shorthand for move_legs with safety check."""
    sc = _get_servoctl()
    if sc.pca is None:
        return
    sc.move_legs(left_h, right_h, left_l, right_l, speed)


def _neutral(speed=0.7):
    """Return to neutral standing position."""
    _move(50, 50, 50, 50, speed)


# ---------------------------------------------------------------------------
# Gesture definitions
# ---------------------------------------------------------------------------

def _gesture_nod():
    """Small forward tilt and return — agreement, acknowledgment."""
    _move(50, 50, 50, 50, 0.8)
    _move(35, 35, 50, 50, 0.9)
    time.sleep(0.15)
    _move(50, 50, 50, 50, 0.8)


def _gesture_lean_in():
    """Slight forward lean, hold briefly — curiosity, interest."""
    _move(50, 50, 50, 50, 0.7)
    _move(30, 30, 55, 55, 0.7)
    time.sleep(0.6)
    _move(50, 50, 50, 50, 0.6)


def _gesture_recoil():
    """Quick backward lean and return — surprise, shock."""
    _move(70, 70, 45, 45, 1.0)
    time.sleep(0.2)
    _move(50, 50, 50, 50, 0.7)


def _gesture_tilt_curious():
    """Slight side tilt, hold — thinking, pondering."""
    _move(50, 50, 50, 50, 0.7)
    _move(35, 65, 50, 50, 0.8)
    time.sleep(0.7)
    _move(50, 50, 50, 50, 0.6)


def _gesture_shake_no():
    """Quick left-right-left rock — disagreement, disbelief."""
    _move(50, 50, 50, 50, 0.8)
    _move(35, 65, 50, 50, 1.0)
    _move(65, 35, 50, 50, 1.0)
    _move(35, 65, 50, 50, 1.0)
    _move(50, 50, 50, 50, 0.8)


def _gesture_excited_bounce():
    """Small rapid up-down bounces — joy, excitement."""
    _move(50, 50, 50, 50, 0.9)
    for _ in range(3):
        _move(40, 40, 50, 50, 1.0)
        _move(60, 60, 50, 50, 1.0)
    _move(50, 50, 50, 50, 0.8)


def _gesture_droop():
    """Slow slight forward droop — sadness, disappointment."""
    _move(50, 50, 50, 50, 0.5)
    _move(60, 60, 55, 55, 0.4)
    time.sleep(0.8)
    _move(50, 50, 50, 50, 0.4)


def _gesture_puff_up():
    """Rise to full height, hold — confidence, pride."""
    _move(50, 50, 50, 50, 0.7)
    _move(20, 20, 50, 50, 0.7)
    time.sleep(0.6)
    _move(50, 50, 50, 50, 0.5)


def _gesture_shrug():
    """Brief alternating tilt — uncertainty, 'who knows'."""
    _move(50, 50, 50, 50, 0.8)
    _move(30, 70, 50, 50, 0.9)
    time.sleep(0.15)
    _move(70, 30, 50, 50, 0.9)
    time.sleep(0.15)
    _move(50, 50, 50, 50, 0.7)


def _gesture_scan():
    """Slow left-to-right sweep — looking around, surveying."""
    _move(50, 50, 50, 50, 0.6)
    _move(50, 50, 30, 30, 0.5)
    time.sleep(0.3)
    _move(50, 50, 70, 70, 0.5)
    time.sleep(0.3)
    _move(50, 50, 50, 50, 0.5)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GESTURES = {
    "nod":             {"fn": _gesture_nod,             "desc": "Agreement, acknowledgment"},
    "lean_in":         {"fn": _gesture_lean_in,         "desc": "Curiosity, interest"},
    "recoil":          {"fn": _gesture_recoil,          "desc": "Surprise, shock"},
    "tilt_curious":    {"fn": _gesture_tilt_curious,    "desc": "Thinking, pondering"},
    "shake_no":        {"fn": _gesture_shake_no,        "desc": "Disagreement, disbelief"},
    "excited_bounce":  {"fn": _gesture_excited_bounce,  "desc": "Joy, excitement"},
    "droop":           {"fn": _gesture_droop,           "desc": "Sadness, disappointment"},
    "puff_up":         {"fn": _gesture_puff_up,         "desc": "Confidence, pride"},
    "shrug":           {"fn": _gesture_shrug,           "desc": "Uncertainty, 'who knows'"},
    "scan":            {"fn": _gesture_scan,             "desc": "Looking around, surveying"},
}

GESTURE_NAMES = sorted(GESTURES.keys())

# Thread lock — only one gesture at a time
_gesture_lock = threading.Lock()


def execute_gesture(name):
    """Execute a micro-gesture by name. Non-blocking if called from main thread.

    Skips silently if:
      - A full movement is already in progress (servoctl.MOVING)
      - Another gesture is already running
      - The gesture name is unknown
      - PCA9685 is not initialized (no servo board)

    Args:
        name: Gesture name from GESTURES registry.
    """
    if not name:
        return

    sc = _get_servoctl()

    # Don't interrupt a full movement
    if sc.MOVING:
        return

    # Don't stack gestures
    if not _gesture_lock.acquire(blocking=False):
        return

    gesture = GESTURES.get(name)
    if gesture is None:
        queue_message(f"WARNING: Unknown gesture '{name}'")
        _gesture_lock.release()
        return

    try:
        gesture["fn"]()
        sc.disable_all_servos()
    except Exception as e:
        queue_message(f"WARNING: Gesture '{name}' failed: {e}")
        # Try to return to neutral on failure
        try:
            _neutral()
            sc.disable_all_servos()
        except Exception:
            pass
    finally:
        _gesture_lock.release()


def execute_gesture_async(name):
    """Fire a gesture in a background thread. Returns immediately.

    This is the primary entry point for the TTS pipeline — call this
    when speech starts to gesture in parallel with audio playback.
    """
    if not name or name not in GESTURES:
        return
    thread = threading.Thread(
        target=execute_gesture,
        args=(name,),
        name=f"gesture-{name}",
        daemon=True,
    )
    thread.start()
    return thread
