"""
module_gestures.py

Mood-modulated gesture layer for TARS-AI.

Maps 8 core conversational gestures to simple parameterized movement
functions.  Mood (from BodyStateManager) controls speed and amplitude —
a sad TARS moves slowly with minimal range, an excited TARS is fast
and exaggerated.

Also provides an idle fidget system that runs on body-autopilot:
tiny weight shifts and sways during STANDBY, frequency set by mood.
No LLM involvement — part of the subconscious body layer.
"""

import random
import threading
import time

from modules.module_messageQue import queue_message
from modules.module_movements import (
    simple_nod,
    simple_lean,
    simple_recoil,
    simple_rock,
    simple_bounce,
    simple_shrug,
    simple_wave,
    simple_settle,
    fidget_weight_shift,
    fidget_settle,
    fidget_rock,
)


# ── Gesture registry ─────────────────────────────────────────────────────────

GESTURES = {
    "nod":     {"fn": simple_nod,     "desc": "Agreement, acknowledgment"},
    "lean":    {"fn": simple_lean,    "desc": "Curiosity, leaning in with interest"},
    "recoil":  {"fn": simple_recoil,  "desc": "Surprise, shock, disbelief"},
    "rock":    {"fn": simple_rock,    "desc": "Disagreement, thinking, 'no way'"},
    "bounce":  {"fn": simple_bounce,  "desc": "Joy, excitement, laughter"},
    "shrug":   {"fn": simple_shrug,   "desc": "Uncertainty, 'who knows'"},
    "wave":    {"fn": simple_wave,    "desc": "Greeting, hello/goodbye"},
    "settle":  {"fn": simple_settle,  "desc": "Calm acceptance, respect, gratitude"},
}

GESTURE_NAMES = sorted(GESTURES.keys())

_FIDGETS = [fidget_weight_shift, fidget_settle, fidget_rock]

# Thread lock — only one gesture at a time
_gesture_lock = threading.Lock()

# Flag: True while a gesture is physically running (checked by STT)
gesture_active = False


def _get_mood_params():
    """Get (speed, amplitude, fidget_interval) from body state, with defaults."""
    try:
        from modules.module_body_state import get_body_state_manager
        bsm = get_body_state_manager()
        if bsm is not None:
            return bsm.get_motion_params()
    except Exception:
        pass
    return (0.9, 0.8, 50)  # neutral defaults


# ── Gesture execution ─────────────────────────────────────────────────────────

def execute_gesture(name):
    """Execute a gesture by name with mood-modulated speed/amplitude.

    Blocks until complete. Temporarily disables movement callbacks so the
    gesture does NOT pause STT or UI.

    Skips silently if another gesture is running or name is unknown.
    """
    global gesture_active
    if not name:
        return

    if not _gesture_lock.acquire(blocking=False):
        return

    gesture = GESTURES.get(name)
    if gesture is None:
        queue_message(f"WARNING: Unknown gesture '{name}'")
        _gesture_lock.release()
        return

    import modules.module_servoctl as servoctl

    # Save and disable movement callbacks so gesture doesn't pause STT
    old_start = servoctl._on_movement_start
    old_end = servoctl._on_movement_end
    servoctl._on_movement_start = None
    servoctl._on_movement_end = None

    speed, amplitude, _ = _get_mood_params()

    try:
        gesture_active = True
        queue_message(f"GESTURE: {name} (speed={speed:.1f}, amp={amplitude:.1f})")
        gesture["fn"](speed=speed, amplitude=amplitude)
    except Exception as e:
        queue_message(f"WARNING: Gesture '{name}' failed: {e}")
    finally:
        gesture_active = False
        servoctl._on_movement_start = old_start
        servoctl._on_movement_end = old_end
        _gesture_lock.release()


def execute_gesture_async(name):
    """Fire a gesture in a background thread. Returns immediately."""
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


# ── Idle fidget system (body-autopilot) ───────────────────────────────────────

_fidget_stop = threading.Event()
_fidget_thread = None


def _fidget_loop():
    """Background loop that plays subtle idle fidgets at mood-driven intervals."""
    queue_message("GESTURES: Idle fidgets started")
    while not _fidget_stop.is_set():
        _, amplitude, interval = _get_mood_params()

        # Wait for the mood-driven interval (interruptible)
        if _fidget_stop.wait(timeout=interval):
            break

        # Only fidget in STANDBY (not during conversation or TTS)
        try:
            from modules.module_state import get_tars_state, TarsState
            if get_tars_state() != TarsState.STANDBY:
                continue
        except Exception:
            continue

        # Skip if a gesture is already running
        if not _gesture_lock.acquire(blocking=False):
            continue

        try:
            fn = random.choice(_FIDGETS)
            fn(amplitude=amplitude)
        except Exception:
            pass
        finally:
            _gesture_lock.release()

    queue_message("GESTURES: Idle fidgets stopped")


def start_idle_fidgets():
    """Start the idle fidget background loop."""
    global _fidget_thread
    if _fidget_thread is not None and _fidget_thread.is_alive():
        return
    _fidget_stop.clear()
    _fidget_thread = threading.Thread(target=_fidget_loop, daemon=True, name="idle-fidgets")
    _fidget_thread.start()


def stop_idle_fidgets():
    """Stop the idle fidget background loop."""
    _fidget_stop.set()
