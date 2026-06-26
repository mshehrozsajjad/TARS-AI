"""
module_movement_tuner.py

Movement tuning infrastructure for TARS-AI.

Provides tools to:
  - Define movement sequences as data (lists of step tuples)
  - Execute sequences while recording per-step IMU stability data
  - Score movements by stability (max tilt, wobble, gyro)
  - Generate parameter variations for problematic steps
  - Load / save tuned parameters from JSON

The tuning process runs once (via app-movement-tuner.py) and saves
optimized parameters.  At runtime, movement functions load the saved
params — no IMU overhead, full speed.
"""

import json
import math
import os
import time
import threading

from modules.module_messageQue import queue_message

# ── Paths ────────────────────────────────────────────────────────────────────

def _params_path():
    """Path to the tuned movement parameters JSON file."""
    from modules.module_config import character_name
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "character", character_name, "movement_params.json")


# ── Default (hardcoded) movement sequences ───────────────────────────────────
# Extracted from module_movements.py so the tuner has a baseline to work from.
# Format: list of [left_height, right_height, left_swing, right_swing, speed]

DEFAULTS = {
    "step_forward": [
        [50, 50, 50, 50, 0.9],
        [42, 42, 40, 40, 0.9],
        [70, 70, 23, 23, 0.9],
        [30, 30, 30, 30, 0.8],
        [70, 70, 35, 35, 0.9],
        [60, 60, 50, 50, 0.9],
        [50, 50, 50, 50, 0.9],
    ],
    "walk_forward": [
        [50, 50, 50, 50, 0.8],
        [40, 70, 50, 50, 0.5],
        [40, 70, 35, 50, 0.5],
        [50, 50, 35, 50, 0.5],
        [70, 40, 50, 50, 0.5],
        [70, 40, 50, 35, 0.5],
        [50, 50, 50, 35, 0.5],
    ],
    "step_backward": [
        [50, 50, 50, 50, 0.9],
        [30, 30, 55, 55, 0.8],
        [68, 68, 82, 82, 0.8],
        [30, 30, 70, 70, 0.8],
        [50, 50, 62, 62, 0.9],
        [65, 65, 50, 50, 0.9],
        [50, 50, 50, 50, 0.9],
    ],
    "walk_backward": [
        [50, 50, 50, 50, 0.8],
        [50, 65, 50, 50, 0.5],
        [50, 65, 50, 75, 0.5],
        [50, 50, 50, 75, 0.5],
        [65, 50, 50, 50, 0.5],
        [65, 50, 75, 50, 0.5],
        [50, 50, 75, 50, 0.5],
    ],
    "_turn_right": [
        [50, 50, 50, 50, 0.9],
        [70, 70, 50, 50, 0.9],
        [70, 70, 65, 35, 0.9],
        [45, 45, 65, 35, 0.9],
        [52, 52, 50, 50, 0.8],
        [50, 50, 50, 50, 0.8],
    ],
    "_turn_left": [
        [50, 50, 50, 50, 0.9],
        [70, 70, 50, 50, 0.9],
        [70, 70, 35, 65, 0.9],
        [45, 45, 35, 65, 0.9],
        [52, 52, 50, 50, 0.8],
        [50, 50, 50, 50, 0.8],
    ],
}


# ── Param loading / saving ───────────────────────────────────────────────────

_params_cache = None


def load_tuned_params(movement_name):
    """Load tuned step sequence for a movement, or None if not tuned.

    Returns a list of [lh, rh, ls, rs, speed] lists, or None.
    Cached after first load — restart to pick up changes.
    """
    global _params_cache
    if _params_cache is None:
        path = _params_path()
        try:
            with open(path, "r") as f:
                _params_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _params_cache = {}

    entry = _params_cache.get(movement_name)
    if entry is None:
        return None
    return entry.get("steps")


def save_tuned_params(movement_name, steps, score=None):
    """Save tuned step sequence to the movement params JSON file."""
    global _params_cache
    path = _params_path()

    # Load existing file (or start fresh)
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    entry = {"steps": steps}
    if score is not None:
        entry["score"] = score
    data[movement_name] = entry

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    # Update cache
    _params_cache = data
    queue_message(f"TUNER: Saved tuned params for {movement_name}")


def invalidate_cache():
    """Force re-read of params file on next load_tuned_params call."""
    global _params_cache
    _params_cache = None


def get_default_steps(movement_name):
    """Return the default (hardcoded) steps for a movement, or None."""
    return DEFAULTS.get(movement_name)


# ── Step execution ───────────────────────────────────────────────────────────

def execute_step_sequence(steps):
    """Execute a movement as a list of [lh, rh, ls, rs, speed] steps.

    Handles MOVING flag and movement callbacks, same as the original
    movement functions.
    """
    import modules.module_servoctl as servoctl

    if servoctl.MOVING:
        return

    servoctl.MOVING = True
    servoctl._notify_movement_start()
    try:
        for step in steps:
            lh, rh, ls, rs, speed = step
            servoctl.move_legs(lh, rh, ls, rs, speed)
        time.sleep(0.1)
        servoctl.disable_all_servos()
    finally:
        servoctl.MOVING = False
        servoctl._notify_movement_end()


# ── IMU recording during movement ───────────────────────────────────────────

class IMURecorder:
    """Records IMU readings in a background thread during movement.

    Reads from the IMUManager singleton (already polling at 50Hz on its
    own I2C thread).  No additional bus access — zero contention risk.
    Falls back to direct I2C reads if IMUManager is not running (for the
    standalone tuner tool).
    """

    def __init__(self, poll_interval=0.02):
        self._poll_interval = poll_interval
        self._recordings = []      # list of {timestamp, tilt, gyro_total, ax, ay, az}
        self._step_markers = []    # list of (step_index, timestamp)
        self._running = False
        self._thread = None
        self._imu = None
        self._direct_reader = None

    def start(self):
        """Start recording IMU data in a background thread."""
        from modules.module_imu import get_imu_manager
        self._imu = get_imu_manager()

        # If IMUManager isn't running, try direct reads (standalone tool mode)
        if self._imu is None or not self._imu.sensor_initialized:
            self._imu = None
            try:
                import smbus2
                self._direct_reader = _DirectIMUReader()
            except Exception:
                return False

        self._recordings = []
        self._step_markers = []
        self._running = True
        self._thread = threading.Thread(
            target=self._record_loop, daemon=True, name="imu-recorder")
        self._thread.start()
        return True

    def stop(self):
        """Stop recording and return all data."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self._recordings, self._step_markers

    def mark_step(self, step_index):
        """Mark the start of a movement step (called between move_legs calls)."""
        self._step_markers.append((step_index, time.monotonic()))

    def _record_loop(self):
        while self._running:
            reading = self._get_reading()
            if reading:
                tilt = reading["tilt"]
                # Discard stale/initial readings — a standing robot always
                # has at least ~2° of structural tilt.  0.0° means the IMU
                # thread hasn't updated yet (I2C contention from servos).
                if tilt < 0.5 and reading["gyro_total"] < 0.5:
                    time.sleep(self._poll_interval)
                    continue
                self._recordings.append({
                    "t": time.monotonic(),
                    "tilt": tilt,
                    "gyro": reading["gyro_total"],
                    "gx": reading.get("gx", 0.0),
                    "ax": reading["ax"],
                    "ay": reading["ay"],
                    "az": reading["az"],
                })
            time.sleep(self._poll_interval)

    def _get_reading(self):
        if self._imu:
            return self._imu.get_reading()
        if self._direct_reader:
            return self._direct_reader.read()
        return None


class _DirectIMUReader:
    """Fallback IMU reader for standalone tool mode (no IMUManager)."""

    REG_PWR_MGMT_1 = 0x6B
    REG_ACCEL_XOUT_H = 0x3B
    ACCEL_SCALE = 16384.0
    GYRO_SCALE = 131.0

    def __init__(self, address=0x68):
        import smbus2
        self.address = address
        self.bus = smbus2.SMBus(1)
        self.bus.write_byte_data(address, self.REG_PWR_MGMT_1, 0x00)
        time.sleep(0.1)

    def read(self):
        try:
            data = self.bus.read_i2c_block_data(
                self.address, self.REG_ACCEL_XOUT_H, 14)
            raws = [(data[i] << 8 | data[i + 1]) for i in range(0, 14, 2)]
            signed = [(v - 0x10000) if v >= 0x8000 else v for v in raws]

            # Data layout (14 bytes, 7 signed 16-bit values):
            # [0]=accelX [1]=accelY [2]=accelZ [3]=temp [4]=gyroX [5]=gyroY [6]=gyroZ
            ax = signed[0] / self.ACCEL_SCALE
            ay = signed[1] / self.ACCEL_SCALE
            az = signed[2] / self.ACCEL_SCALE
            # signed[3] = temperature, skip
            gx = signed[4] / self.GYRO_SCALE
            gy = signed[5] / self.GYRO_SCALE
            gz = signed[6] / self.GYRO_SCALE

            mag = math.sqrt(ax * ax + ay * ay + az * az)
            gyro_total = math.sqrt(gx * gx + gy * gy + gz * gz)

            if mag > 4.0 or mag < 0.1:
                return None

            tilt = 0.0
            if mag > 0.1:
                ratio = max(-1.0, min(1.0, ax / mag))
                tilt = math.degrees(math.acos(ratio))

            return {
                "ax": ax, "ay": ay, "az": az,
                "gx": gx, "gy": gy, "gz": gz,
                "tilt": tilt,
                "gyro_total": gyro_total,
                "magnitude": mag,
            }
        except Exception:
            return None


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_per_step(recordings, step_markers):
    """Compute stability metrics for each step in a movement.

    Returns a list of dicts, one per step:
        {
            "step": step_index,
            "max_tilt": float,     # peak tilt during this step (degrees)
            "avg_tilt": float,     # average tilt during this step
            "max_gyro": float,     # peak rotation rate (degrees/s)
            "avg_gyro": float,     # average rotation rate
            "wobble": float,       # tilt standard deviation (instability)
            "avg_yaw": float,      # average yaw rate (degrees/s, + = turning one way)
            "readings": int,       # number of IMU samples in this window
        }
    """
    if not recordings or not step_markers:
        return []

    results = []

    for i, (step_idx, start_t) in enumerate(step_markers):
        # End time = next step's start, or last recording
        if i + 1 < len(step_markers):
            end_t = step_markers[i + 1][1]
        else:
            end_t = recordings[-1]["t"] + 0.01

        # Filter readings in this step's time window
        window = [r for r in recordings if start_t <= r["t"] < end_t]
        if not window:
            results.append({
                "step": step_idx, "max_tilt": 0, "avg_tilt": 0,
                "max_gyro": 0, "avg_gyro": 0, "wobble": 0,
                "avg_yaw": 0, "readings": 0,
            })
            continue

        tilts = [r["tilt"] for r in window]
        gyros = [r["gyro"] for r in window]
        yaws = [r.get("gx", 0.0) for r in window]
        avg_tilt = sum(tilts) / len(tilts)
        avg_gyro = sum(gyros) / len(gyros)
        avg_yaw = sum(yaws) / len(yaws)

        # Wobble = standard deviation of tilt (how much it's rocking)
        if len(tilts) > 1:
            variance = sum((t - avg_tilt) ** 2 for t in tilts) / len(tilts)
            wobble = math.sqrt(variance)
        else:
            wobble = 0.0

        results.append({
            "step": step_idx,
            "max_tilt": max(tilts),
            "avg_tilt": round(avg_tilt, 2),
            "max_gyro": max(gyros),
            "avg_gyro": round(avg_gyro, 2),
            "wobble": round(wobble, 2),
            "avg_yaw": round(avg_yaw, 2),
            "readings": len(window),
        })

    return results


def score_movement(step_scores):
    """Compute an overall stability score (0-100) from per-step scores.

    100 = perfectly stable.  Penalizes high tilt, high gyro, wobble,
    yaw drift, and insufficient data (I2C contention).
    """
    if not step_scores:
        return 0

    # Count steps with actual IMU data (readings > 0)
    steps_with_data = [s for s in step_scores if s["readings"] > 2]
    if len(steps_with_data) < len(step_scores) // 2:
        # More than half the steps have no data — unreliable, penalize hard
        return 10

    max_tilt = max(s["max_tilt"] for s in steps_with_data) if steps_with_data else 0
    avg_wobble = sum(s["wobble"] for s in steps_with_data) / len(steps_with_data)
    avg_gyro = sum(s["avg_gyro"] for s in steps_with_data) / len(steps_with_data)

    # Yaw drift: average yaw rate across all steps (consistent turning = bad)
    # abs() because turning left or right is equally bad for "walk straight"
    avg_yaw = abs(sum(s.get("avg_yaw", 0) for s in steps_with_data) / len(steps_with_data))

    # Tilt penalty: 0 at 0°, 100 at 45°+
    tilt_penalty = min(100, (max_tilt / 45.0) * 100)

    # Wobble penalty: 0 at 0°, 100 at 15°+
    wobble_penalty = min(100, (avg_wobble / 15.0) * 100)

    # Gyro penalty: 0 at 0°/s, 100 at 100°/s+
    gyro_penalty = min(100, (avg_gyro / 100.0) * 100)

    # Yaw penalty: 0 at 0°/s, 100 at 20°/s+ sustained turning
    yaw_penalty = min(100, (avg_yaw / 20.0) * 100)

    # Data coverage penalty: penalize if many steps had no IMU data
    coverage = len(steps_with_data) / len(step_scores)
    coverage_penalty = (1.0 - coverage) * 50  # up to 50 points off

    # Weighted combination — yaw gets meaningful weight
    raw = 100 - (tilt_penalty * 0.4 + wobble_penalty * 0.2
                 + gyro_penalty * 0.15 + yaw_penalty * 0.25) - coverage_penalty
    return max(0, min(100, int(round(raw))))


# ── Profiling ────────────────────────────────────────────────────────────────

def profile_movement(steps, num_runs=3, reset_pause=2.0):
    """Run a movement sequence multiple times and return averaged per-step scores.

    Between runs, returns to neutral and pauses for the body to settle.

    Returns:
        (avg_step_scores, overall_score, per_run_scores)
    """
    import modules.module_servoctl as servoctl

    all_run_scores = []

    for run in range(num_runs):
        # Return to neutral and let the body settle
        servoctl.move_legs(50, 50, 50, 50, 0.5)
        time.sleep(reset_pause)

        # Start IMU recording
        recorder = IMURecorder()
        if not recorder.start():
            print(f"  WARNING: IMU recording unavailable for run {run + 1}")
            continue

        # Execute the movement with step markers
        servoctl.MOVING = True
        try:
            for i, step in enumerate(steps):
                recorder.mark_step(i)
                lh, rh, ls, rs, speed = step
                servoctl.move_legs(lh, rh, ls, rs, speed)
            time.sleep(0.3)  # capture settling after last step
        finally:
            servoctl.MOVING = False

        # Stop recording and score
        recordings, markers = recorder.stop()
        step_scores = score_per_step(recordings, markers)
        overall = score_movement(step_scores)
        all_run_scores.append((step_scores, overall))

    if not all_run_scores:
        return [], 0, []

    # Average per-step scores across runs
    num_steps = len(all_run_scores[0][0])
    avg_scores = []
    for step_idx in range(num_steps):
        scores_for_step = [run[0][step_idx] for run in all_run_scores
                           if step_idx < len(run[0])]
        if not scores_for_step:
            continue
        avg_scores.append({
            "step": step_idx,
            "max_tilt": round(sum(s["max_tilt"] for s in scores_for_step) / len(scores_for_step), 1),
            "avg_tilt": round(sum(s["avg_tilt"] for s in scores_for_step) / len(scores_for_step), 1),
            "max_gyro": round(sum(s["max_gyro"] for s in scores_for_step) / len(scores_for_step), 1),
            "avg_gyro": round(sum(s["avg_gyro"] for s in scores_for_step) / len(scores_for_step), 1),
            "wobble": round(sum(s["wobble"] for s in scores_for_step) / len(scores_for_step), 1),
            "avg_yaw": round(sum(s.get("avg_yaw", 0) for s in scores_for_step) / len(scores_for_step), 1),
            "readings": sum(s["readings"] for s in scores_for_step) // len(scores_for_step),
        })

    avg_overall = sum(r[1] for r in all_run_scores) // len(all_run_scores)

    return avg_scores, avg_overall, all_run_scores


# ── Variation generation ─────────────────────────────────────────────────────

def generate_variations(steps, step_index):
    """Generate parameter variations for a single step to try.

    Produces several copies of the full sequence, each with one
    modification to the target step.  Variations:
      - Reduce height range (move heights toward 50 by 5)
      - Reduce swing range (move swings toward 50 by 5)
      - Reduce speed by 0.1
      - Combined: reduce height + speed
      - Combined: reduce swing + speed
      - Combined: reduce height + swing
      - Combined: all three

    Returns a list of (description, modified_steps) tuples.
    """
    if step_index < 0 or step_index >= len(steps):
        return []

    original = steps[step_index]
    lh, rh, ls, rs, speed = original
    variations = []

    def _toward_50(val, amount=5):
        """Move a value toward 50 by `amount`, clamping to [1, 99]."""
        if val > 50:
            return max(50, val - amount)
        elif val < 50:
            return min(50, val + amount)
        return val

    def _make(desc, new_lh, new_rh, new_ls, new_rs, new_speed):
        modified = [list(s) for s in steps]
        modified[step_index] = [new_lh, new_rh, new_ls, new_rs, new_speed]
        variations.append((desc, modified))

    # Individual adjustments
    _make("height ±5",
          _toward_50(lh), _toward_50(rh), ls, rs, speed)

    _make("swing ±5",
          lh, rh, _toward_50(ls), _toward_50(rs), speed)

    spd_reduced = max(0.3, speed - 0.1)
    _make(f"speed {speed:.1f}->{spd_reduced:.1f}",
          lh, rh, ls, rs, spd_reduced)

    # Combined adjustments
    _make("height ±5 + speed",
          _toward_50(lh), _toward_50(rh), ls, rs, spd_reduced)

    _make("swing ±5 + speed",
          lh, rh, _toward_50(ls), _toward_50(rs), spd_reduced)

    _make("height ±5 + swing ±5",
          _toward_50(lh), _toward_50(rh), _toward_50(ls), _toward_50(rs), speed)

    _make("all reduced",
          _toward_50(lh), _toward_50(rh), _toward_50(ls), _toward_50(rs), spd_reduced)

    # Larger height reduction (±10)
    _make("height ±10",
          _toward_50(lh, 10), _toward_50(rh, 10), ls, rs, speed)

    # ── Asymmetric swing variations (for drift/yaw correction) ───────
    # If both legs swing identically but one side pushes harder
    # mechanically, TARS drifts.  These try left/right swing imbalances
    # to compensate.
    BIAS = 3
    if ls == rs:
        # Symmetric swings — try biasing each direction
        _make(f"swing L+{BIAS} R-{BIAS}",
              lh, rh, ls + BIAS, rs - BIAS, speed)
        _make(f"swing L-{BIAS} R+{BIAS}",
              lh, rh, ls - BIAS, rs + BIAS, speed)
    else:
        # Already asymmetric — try widening and narrowing the gap
        _make("swing gap +3",
              lh, rh, ls + BIAS, rs - BIAS, speed)
        _make("swing gap -3",
              lh, rh, ls - BIAS, rs + BIAS, speed)

    # Asymmetric height (one leg slightly higher for mechanical imbalance)
    _make(f"height L+3 R-3",
          min(99, lh + 3), max(1, rh - 3), ls, rs, speed)
    _make(f"height L-3 R+3",
          max(1, lh - 3), min(99, rh + 3), ls, rs, speed)

    return variations


def find_worst_steps(step_scores, threshold=10.0):
    """Return indices of steps with max_tilt or yaw drift above threshold.

    Sorted by a combined badness score (tilt + yaw), worst first.
    """
    bad = []
    for s in step_scores:
        tilt = s["max_tilt"]
        yaw = abs(s.get("avg_yaw", 0))
        # A step is "bad" if it has high tilt OR significant yaw drift
        badness = tilt + yaw * 0.5
        if tilt > threshold or yaw > 10:
            bad.append((s["step"], badness))
    bad.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in bad]
