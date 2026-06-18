"""
module_gestures.py

Conversational gesture layer for TARS-AI.

Maps expressive gesture names to existing movement functions in
module_movements.py. The LLM picks a gesture via the optional
"gesture" field in its JSON response — this module executes it
in a background thread so it runs parallel to TTS playback.

All movements are the proven, hardware-tested functions from
module_movements.py. This module is just the naming/dispatch layer.
"""

import threading

from modules.module_messageQue import queue_message
from modules.module_movements import (
    bow,
    tilt_right,
    tilt_left,
    side_side,
    laugh,
    excited,
    swing_legs,
    wave_right,
    wave_left,
    pose,
    neutral_legs,
    right_hi,
    left_hi,
)


# ---------------------------------------------------------------------------
# Gesture registry — maps conversational names to movement functions
# ---------------------------------------------------------------------------

GESTURES = {
    "nod":             {"fn": bow,           "desc": "Agreement, acknowledgment"},
    "lean_in":         {"fn": tilt_right,    "desc": "Curiosity, leaning in with interest"},
    "recoil":          {"fn": tilt_left,     "desc": "Surprise, leaning back in shock"},
    "tilt_curious":    {"fn": tilt_left,     "desc": "Thinking, pondering"},
    "shake_no":        {"fn": side_side,     "desc": "Disagreement, disbelief"},
    "excited_bounce":  {"fn": excited,       "desc": "Joy, excitement"},
    "laugh":           {"fn": laugh,         "desc": "Amusement, laughter"},
    "droop":           {"fn": swing_legs,    "desc": "Sadness, restless disappointment"},
    "puff_up":         {"fn": pose,          "desc": "Confidence, pride, standing tall"},
    "shrug":           {"fn": side_side,     "desc": "Uncertainty, 'who knows'"},
    "scan":            {"fn": side_side,     "desc": "Looking around, surveying"},
    "wave":            {"fn": wave_right,    "desc": "Greeting, waving hello"},
    "celebrate":       {"fn": excited,       "desc": "Celebration, big excitement"},
    "hi_right":        {"fn": right_hi,      "desc": "Raising right side in greeting"},
    "hi_left":         {"fn": left_hi,       "desc": "Raising left side in greeting"},
    "bow":             {"fn": bow,           "desc": "Respectful bow, gratitude, thank you"},
}

GESTURE_NAMES = sorted(GESTURES.keys())

# Thread lock — only one gesture at a time
_gesture_lock = threading.Lock()


def execute_gesture(name):
    """Execute a gesture by name. Blocks until complete.

    Temporarily disables the movement start/end callbacks so the gesture
    does NOT pause STT or UI. This is critical — gestures run during or
    right after speech, and pausing STT would cause the conversation to
    drop into sleep mode.

    Skips silently if:
      - Another gesture is already running
      - The gesture name is unknown
      - PCA9685 is not initialized

    Args:
        name: Gesture name from GESTURES registry.
    """
    if not name:
        return

    # Don't stack gestures
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

    try:
        queue_message(f"GESTURE: {name}")
        gesture["fn"]()
    except Exception as e:
        queue_message(f"WARNING: Gesture '{name}' failed: {e}")
    finally:
        # Restore original callbacks
        servoctl._on_movement_start = old_start
        servoctl._on_movement_end = old_end
        _gesture_lock.release()


def execute_gesture_async(name):
    """Fire a gesture in a background thread. Returns immediately.

    This is the primary entry point — called from module_main.py
    when TTS starts playing to gesture in parallel with speech.
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
