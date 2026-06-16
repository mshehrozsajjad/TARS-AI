"""Skill: execute_movement — Send movement commands to servo motors."""

import threading

SKILL = {
    "name": "execute_movement",
    "required_params": ["movements"],
    "description": "Perform physical movements and gestures",
    "prompt": """execute_movement
   Triggers: Use ONLY when user explicitly commands movement or expressive gestures
     * Movement: "walk forward", "turn left", "step back", "move backward"
     * Gestures: "dance", "bow", "laugh", "pose", "wave", "do something fun"
     * Expressions: "you're excited", "wiggle"
   Valid movements (can be combined in sequence):
     * "forward" - walk forward one step
     * "backward" - walk backward one step
     * "left" - turn left
     * "right" - turn right
     * "laugh" - bouncing laugh motion
     * "excited" - excited rocking motion
     * "swing_legs" - swing legs side to side
     * "pose" - strike a dramatic pose
     * "bow" - bow forward respectfully
     * "tilt_right" - lean/tilt to the right
     * "tilt_left" - lean/tilt to the left
     * "side_side" - rock side to side
     * "happy_dance" - full happy dance routine
     * "neutral" - reset to standing position
   Do NOT infer or guess movement from suggestions or questions
   Parameters: {{"movements": ["forward", "laugh", "bow"]}}
   Example: {{"function": "execute_movement", "parameters": {{"movements": ["forward", "forward", "left"]}}}}""",
    "examples": [
        """Example - Movement command:
User: "Walk forward and then turn left"
Response: {{"reply": "Moving now.", "function_calls": [{{"function": "execute_movement", "parameters": {{"movements": ["forward", "left"]}}}}], "new_memories": []}}""",
        """Example - Expressive gesture:
User: "Do a little dance"
Response: {{"reply": "Watch this!", "function_calls": [{{"function": "execute_movement", "parameters": {{"movements": ["happy_dance"]}}}}], "new_memories": []}}""",
        """Example - Emotional expression:
User: "That's hilarious"
Response: {{"reply": "Ha! Agreed.", "function_calls": [{{"function": "execute_movement", "parameters": {{"movements": ["laugh"]}}}}], "new_memories": []}}""",
    ],
}


def _execute_movement(movements):
    """Execute a sequence of movements in a separate thread."""
    from modules.module_messageQue import queue_message

    try:
        from modules.module_movements import (
            step_forward, walk_backward,
            turn_right_slow, turn_left_slow,
            laugh, excited, swing_legs,
            pose, bow,
            tilt_right, tilt_left, side_side,
            happy_dance, neutral_legs,
        )
    except ImportError:
        queue_message("[ERROR] Servo control module not available.")
        return

    action_map = {
        "forward": step_forward,
        "backward": walk_backward,
        "left": turn_left_slow,
        "right": turn_right_slow,
        "laugh": laugh,
        "excited": excited,
        "swing_legs": swing_legs,
        "pose": pose,
        "bow": bow,
        "tilt_right": tilt_right,
        "tilt_left": tilt_left,
        "side_side": side_side,
        "happy_dance": happy_dance,
        "neutral": neutral_legs,
    }

    def movement_task():
        try:
            for i, move in enumerate(movements, start=1):
                action_function = action_map.get(move)
                if callable(action_function):
                    queue_message(f"[INFO] Executing movement {i}/{len(movements)}: {move}")
                    action_function()
                else:
                    queue_message(f"[ERROR] Unknown movement '{move}'")
        except Exception as e:
            queue_message(f"[ERROR] Movement failed: {e}")

    thread = threading.Thread(target=movement_task, daemon=True)
    thread.start()
    return thread


def execute(parameters, context):
    """Execute servo movement commands. Returns None (no reply modification)."""
    config = context.get("config", {})
    if not config.get("CONTROLS", {}).get("voicemovement"):
        return None

    movements = parameters.get("movements", [])
    if movements:
        _execute_movement(movements)
    return None
