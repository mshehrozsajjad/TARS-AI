"""
module_imu.py

MPU6050 IMU sensor module for TARS-AI physical awareness.

Reads 6-axis accelerometer + gyroscope data over I2C, detects physical
events (picked up, set down, knocked over, shaking, freefall), and
triggers verbal reactions via the LLM + router — same pattern as the
drives proactive speech system.

Sensor mounting on TARS:
  X-axis = vertical (up when positive, ax ~+1g at rest)
  Y-axis = lateral (left/right)
  Z-axis = forward/back (forward when negative)

The MPU6050 shares I2C bus 1 with the PCA9685 servo driver (0x40) and
optional INA219 battery sensor (0x41). Default address: 0x68.
"""

import asyncio
import json
import math
import os
import random
import time
import threading
from collections import deque
from datetime import datetime

import smbus2

from modules.module_messageQue import queue_message

# ── Singleton ────────────────────────────────────────────────────────────────

_instance = None


def get_imu_manager():
    return _instance


# ── MPU6050 Constants ────────────────────────────────────────────────────────

REG_PWR_MGMT_1   = 0x6B
REG_ACCEL_XOUT_H = 0x3B
REG_GYRO_XOUT_H  = 0x43
REG_WHO_AM_I     = 0x75

# Default ±2g / ±250°/s ranges
ACCEL_SCALE = 16384.0  # LSB/g
GYRO_SCALE  = 131.0    # LSB/(°/s)

# Posture thresholds (tilt from vertical in degrees)
# Calibrated from real sensor data:
#   Standing: tilt ~4°, Forward tilt: ~19°, On back: ~92°
TILT_THRESHOLD     = 20   # degrees — "tilted"
ON_SIDE_THRESHOLD  = 55   # degrees — "on side" / "on back"
INVERTED_THRESHOLD = 135  # degrees — "upside down"

# ── Event Detection Thresholds ───────────────────────────────────────────────
# Calibrated from real sensor data:
#   Stable on table:  mag variance ~0.0001, gyro ~3°/s
#   Held in air:      mag variance ~0.01-0.05, gyro spikes
#   Shaking:          mag swings 0.84-1.29g, rapid direction changes

# Stability — variance of magnitude over sliding window
STABLE_MAG_VARIANCE   = 0.002   # below this = resting on surface
UNSTABLE_MAG_VARIANCE = 0.003   # above this = being handled (resting ~0.0001)

# Pickup — gyro spike as alternative trigger (resting gyro ~3°/s)
PICKUP_GYRO_THRESHOLD = 15.0    # °/s — immediate motion signal

# Set down — require surface-level stillness (hand tremor > this)
# On table gyro is consistently 3.2-3.6°/s. In hand, micro-movements
# push above 4°/s frequently. 3s ensures a hand can't fake a surface.
SURFACE_GYRO_THRESHOLD = 4.0    # °/s — tight margin above surface baseline
SURFACE_TILT_THRESHOLD = 8.0    # degrees — on table ~4-5°, in hand ~8-15°
SETDOWN_SETTLE_TIME    = 3.0    # seconds of sustained stillness required

# Shaking — only violent shaking, not normal handling
# Normal handling peaks ~50°/s with rare spikes to ~90°/s (1-2 readings)
# Violent shaking sustains 60-140°/s (10+ readings above 80°/s)
SHAKE_GYRO_THRESHOLD  = 80.0    # °/s total rotation rate
SHAKE_COUNT_THRESHOLD = 5       # readings above threshold in window

# Freefall — magnitude near zero
FREEFALL_THRESHOLD    = 0.3     # g — below this = freefall
FREEFALL_COUNT        = 3       # consecutive readings needed

# Knocked over — sustained posture change
KNOCKOVER_HOLD_TIME   = 2.0     # seconds in on_side before triggering

# ── Reactions JSON ───────────────────────────────────────────────────────────
# Loaded from character/<name>/imu_reactions.json at init.
# User maintains this file. Audio is cached on first play (same TTS cache
# as wake word responses), so subsequent plays are instant.

_reactions = {}


def _load_reactions(config):
    """Load IMU reaction lines from the character's imu_reactions.json."""
    global _reactions
    char_name = config.get('CHAR', {}).get('character_name', 'TARS')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "character", char_name, "imu_reactions.json")

    try:
        with open(json_path, "r") as f:
            _reactions = json.load(f)
        queue_message(f"LOAD: IMU reactions loaded ({sum(len(v) for v in _reactions.values())} lines)")
    except FileNotFoundError:
        queue_message(f"WARNING: IMU reactions file not found: {json_path}")
        _reactions = {}
    except Exception as e:
        queue_message(f"WARNING: Failed to load IMU reactions: {e}")
        _reactions = {}


def _pick_reaction(event_name):
    """Pick a random reaction line for an event type."""
    lines = _reactions.get(event_name, [])
    if not lines:
        return None
    return random.choice(lines)


# ── Manager ──────────────────────────────────────────────────────────────────

class IMUManager:
    """Reads MPU6050 sensor, detects physical events, triggers verbal reactions."""

    POLL_INTERVAL = 0.02       # 50Hz sensor polling
    WINDOW_SIZE = 25           # ~0.5s of readings at 50Hz
    EVENT_CHECK_INTERVAL = 0.2 # check events every 200ms (not every poll)

    def __init__(self, config, body_state_manager=None, ui_manager=None):
        global _instance
        _instance = self

        self._config = config
        self._ui_manager = ui_manager
        imu_cfg = config.get("IMU", {})
        self._address = int(imu_cfg.get("imu_address", "0x68"), 16)
        self._event_cooldown = int(imu_cfg.get("imu_event_cooldown", 15))

        # Quiet hours (reuse drives config if available)
        drives_cfg = config.get("DRIVES", {})
        self._quiet_start = int(drives_cfg.get("quiet_start", 23))
        self._quiet_end = int(drives_cfg.get("quiet_end", 7))

        # Load reaction lines from character JSON
        _load_reactions(config)

        self._bus = None
        self._sensor_ok = False
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Latest reading
        self._reading = {
            "ax": 0.0, "ay": 0.0, "az": 0.0,
            "gx": 0.0, "gy": 0.0, "gz": 0.0,
            "magnitude": 1.0,
            "tilt": 0.0,
            "gyro_total": 0.0,
        }

        # Derived posture
        self._posture = "upright"

        # Sliding window for event detection
        self._mag_window = deque(maxlen=self.WINDOW_SIZE)
        self._gyro_window = deque(maxlen=self.WINDOW_SIZE)
        self._tilt_window = deque(maxlen=self.WINDOW_SIZE)

        # Event state machine
        self._physical_state = "resting"  # resting, held, knocked_over
        self._state_entered_at = time.time()

        # Posture tracking for knock-over detection
        self._off_upright_since = None  # timestamp when posture left "upright"

        # Event cooldowns: {event_name: last_triggered_timestamp}
        self._last_event = {
            "picked_up": 0, "set_down": 0,
            "knocked_over": 0, "shaking": 0, "freefall": 0,
        }

        # Recent event for body_state prompt (clears after 30s)
        self._recent_event = None
        self._recent_event_time = 0

        # Startup grace period — don't fire events in first 3 seconds
        self._startup_time = time.time()

        # Initialize hardware
        self._init_sensor()

        # Register with body state if available
        if body_state_manager is not None:
            body_state_manager.register_sensor("imu", self.get_sensor_data)
            queue_message("LOAD: IMU registered with body state")

    def _init_sensor(self):
        """Initialize I2C bus and wake the MPU6050."""
        try:
            self._bus = smbus2.SMBus(1)

            # Verify sensor identity
            who = self._bus.read_byte_data(self._address, REG_WHO_AM_I)
            if who not in (0x68, 0x72, 0x73, 0x98):
                queue_message(f"WARNING: IMU WHO_AM_I=0x{who:02X} (expected 0x68)")

            # Wake from sleep
            self._bus.write_byte_data(self._address, REG_PWR_MGMT_1, 0x00)
            time.sleep(0.1)

            # Test read
            self._read_raw_accel()
            self._sensor_ok = True
            queue_message(f"LOAD: MPU6050 initialized at 0x{self._address:02X}")

        except Exception as e:
            queue_message(f"WARNING: MPU6050 not available: {e}")
            self._bus = None
            self._sensor_ok = False

    # ── Raw I2C reads ────────────────────────────────────────────────────

    def _read_signed_16(self, reg):
        """Read signed 16-bit big-endian value from two registers."""
        high = self._bus.read_byte_data(self._address, reg)
        low = self._bus.read_byte_data(self._address, reg + 1)
        value = (high << 8) | low
        if value >= 0x8000:
            value -= 0x10000
        return value

    def _read_raw_accel(self):
        """Read accelerometer X, Y, Z in g."""
        ax = self._read_signed_16(REG_ACCEL_XOUT_H) / ACCEL_SCALE
        ay = self._read_signed_16(REG_ACCEL_XOUT_H + 2) / ACCEL_SCALE
        az = self._read_signed_16(REG_ACCEL_XOUT_H + 4) / ACCEL_SCALE
        return ax, ay, az

    def _read_raw_gyro(self):
        """Read gyroscope X, Y, Z in °/s."""
        gx = self._read_signed_16(REG_GYRO_XOUT_H) / GYRO_SCALE
        gy = self._read_signed_16(REG_GYRO_XOUT_H + 2) / GYRO_SCALE
        gz = self._read_signed_16(REG_GYRO_XOUT_H + 4) / GYRO_SCALE
        return gx, gy, gz

    # ── Derived values ───────────────────────────────────────────────────

    @staticmethod
    def _compute_tilt(ax, ay, az):
        """Compute tilt from vertical in degrees.

        On this TARS build, the X-axis points up. Tilt is the angle
        between the acceleration vector and the X-axis (gravity direction).
        """
        magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        if magnitude < 0.1:
            return 0.0, magnitude
        ratio = max(-1.0, min(1.0, ax / magnitude))
        tilt = math.degrees(math.acos(ratio))
        return tilt, magnitude

    @staticmethod
    def _classify_posture(tilt):
        """Classify posture from tilt angle (degrees from vertical)."""
        if tilt < TILT_THRESHOLD:
            return "upright"
        elif tilt < ON_SIDE_THRESHOLD:
            return "tilted"
        elif tilt < INVERTED_THRESHOLD:
            return "on side"
        else:
            return "upside down"

    # ── Event Detection ──────────────────────────────────────────────────

    def _mag_variance(self):
        """Compute variance of magnitude readings in the sliding window."""
        if len(self._mag_window) < 5:
            return 0.0
        values = list(self._mag_window)
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _shake_count(self):
        """Count readings in gyro window above shake threshold."""
        return sum(1 for g in self._gyro_window if g > SHAKE_GYRO_THRESHOLD)

    def _in_transition(self, now):
        """True if a state change happened recently (suppress noisy events)."""
        return now - self._state_entered_at < 2.0

    def _detect_events(self):
        """State machine for physical event detection.

        States: resting → held → resting (picked_up / set_down)
                resting → knocked_over (sustained non-upright posture)

        Shaking and freefall only fire outside of state transitions to
        avoid false triggers during pickup/setdown (which naturally
        produce gyro spikes and magnitude variance).
        """
        # Don't fire events during startup
        if time.time() - self._startup_time < 3.0:
            return

        now = time.time()
        mag_var = self._mag_variance()
        in_transition = self._in_transition(now)

        with self._lock:
            posture = self._posture
            gyro_total = self._reading["gyro_total"]
            magnitude = self._reading["magnitude"]

        # ── Freefall (highest priority — always checked) ─────────────
        freefall_count = sum(1 for m in list(self._mag_window)[-5:]
                            if m < FREEFALL_THRESHOLD)
        if freefall_count >= FREEFALL_COUNT:
            self._fire_event("freefall", now)
            return

        # ── State machine transitions ────────────────────────────────

        if self._physical_state == "resting":
            # Detect pickup: magnitude variance spikes OR gyro spikes
            # Gyro reacts instantly (>15°/s vs ~3°/s at rest), magnitude
            # variance needs time to build in the sliding window
            picked_up = (mag_var > UNSTABLE_MAG_VARIANCE
                         or gyro_total > PICKUP_GYRO_THRESHOLD)
            if picked_up:
                self._physical_state = "held"
                self._state_entered_at = now
                self._fire_event("picked_up", now)
                return

            # Detect knocked over: posture leaves upright and stays
            elif posture in ("on side", "upside down"):
                if self._off_upright_since is None:
                    self._off_upright_since = now
                elif now - self._off_upright_since > KNOCKOVER_HOLD_TIME:
                    self._physical_state = "knocked_over"
                    self._state_entered_at = now
                    self._fire_event("knocked_over", now)
                    self._off_upright_since = None
                    return
            else:
                self._off_upright_since = None

        elif self._physical_state == "held":
            # Detect set down: stable magnitude + low gyro + near-perfect tilt.
            # On surface: gyro ~3.5°/s, tilt ~4-5°. In hand: gyro drifts,
            # tilt ~8-15°. All three must hold for 3s to confirm surface.
            with self._lock:
                tilt = self._reading["tilt"]
            on_surface = (mag_var < STABLE_MAG_VARIANCE
                          and gyro_total < SURFACE_GYRO_THRESHOLD
                          and tilt < SURFACE_TILT_THRESHOLD)
            if on_surface:
                if not hasattr(self, '_settling_since'):
                    self._settling_since = now
                elif now - self._settling_since > SETDOWN_SETTLE_TIME:
                    self._physical_state = "resting"
                    self._state_entered_at = now
                    self._fire_event("set_down", now)
                    del self._settling_since
                    return
            else:
                if hasattr(self, '_settling_since'):
                    del self._settling_since

        elif self._physical_state == "knocked_over":
            # Recovery: posture returns to upright and stable
            if posture == "upright" and mag_var < STABLE_MAG_VARIANCE:
                self._physical_state = "resting"
                self._state_entered_at = now
                return

        # ── Shaking (80°/s threshold means only violent shaking triggers,
        #    so no state or grace period restrictions needed) ──
        if self._shake_count() >= SHAKE_COUNT_THRESHOLD:
            self._fire_event("shaking", now)

    def _fire_event(self, event_name, now):
        """Check cooldown, log, and trigger verbal reaction."""
        # Cooldown check
        if now - self._last_event.get(event_name, 0) < self._event_cooldown:
            return

        self._last_event[event_name] = now

        # Update recent event for body_state
        with self._lock:
            self._recent_event = event_name
            self._recent_event_time = now

        queue_message(f"IMU: Event detected — {event_name}")

        # Trigger verbal reaction on background thread
        threading.Thread(
            target=self._speak_reaction,
            args=(event_name,),
            name=f"imu-react-{event_name}",
            daemon=True,
        ).start()

    def _is_quiet_hours(self):
        """Check if current time is within quiet hours (no verbal reactions).

        Uses same quiet hours config as drives system.
        """
        hour = datetime.now().hour
        if self._quiet_start > self._quiet_end:
            return hour >= self._quiet_start or hour < self._quiet_end
        else:
            return self._quiet_start <= hour < self._quiet_end

    def _speak_reaction(self, event_name):
        """Pick a random reaction line and play with cached TTS.

        Runs on a background thread. Audio is cached on first play
        (same MD5-hash cache as wake word responses), so subsequent
        plays of the same line are instant — no TTS API call needed.
        """

        # Don't speak if TARS is already talking or thinking
        try:
            from modules.module_state import get_tars_state, TarsState
            state = get_tars_state()
            if state in (TarsState.TALKING, TarsState.THINKING):
                return
        except Exception:
            pass

        line = _pick_reaction(event_name)
        if not line:
            return

        queue_message(f"IMU: Reaction ({event_name}) — \"{line}\"")

        # Push to UI display
        if self._ui_manager:
            char_name = self._config.get('CHAR', {}).get('character_name', 'TARS')
            self._ui_manager.update_data(char_name, line, char_name)

        # Play with caching enabled (is_wakeword=True) — same pattern
        # as wake_word_callback in module_main.py
        try:
            from modules.module_tts import play_audio_chunks
            from modules.module_config import load_config
            config = load_config()
            asyncio.run(play_audio_chunks(line, config['TTS']['ttsoption'], True))
        except Exception as e:
            queue_message(f"WARNING: IMU reaction speech failed: {e}")

    # ── Polling loop ─────────────────────────────────────────────────────

    def _poll_loop(self):
        """Background thread: read sensor, update state, detect events."""
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_event_check = 0
        last_i2c_error = 0  # suppress events after I2C glitches

        while self._running:
            try:
                ax, ay, az = self._read_raw_accel()
                gx, gy, gz = self._read_raw_gyro()

                tilt, magnitude = self._compute_tilt(ax, ay, az)
                posture = self._classify_posture(tilt)
                gyro_total = math.sqrt(gx * gx + gy * gy + gz * gz)

                with self._lock:
                    self._reading = {
                        "ax": ax, "ay": ay, "az": az,
                        "gx": gx, "gy": gy, "gz": gz,
                        "magnitude": magnitude,
                        "tilt": tilt,
                        "gyro_total": gyro_total,
                    }
                    self._posture = posture

                # Update sliding windows
                self._mag_window.append(magnitude)
                self._gyro_window.append(gyro_total)
                self._tilt_window.append(tilt)

                # Check events at lower frequency (every ~200ms)
                # Suppress for 2s after I2C errors — first readings after
                # recovery can be corrupted (false gyro spikes)
                now = time.monotonic()
                if now - last_event_check > self.EVENT_CHECK_INTERVAL:
                    last_event_check = now
                    if now - last_i2c_error > 2.0:
                        self._detect_events()

                consecutive_errors = 0

            except OSError as e:
                consecutive_errors += 1
                last_i2c_error = time.monotonic()
                if consecutive_errors == 1:
                    queue_message(f"WARNING: IMU I2C error: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    queue_message("ERROR: IMU too many I2C errors, stopping poll")
                    self._sensor_ok = False
                    break

            except Exception as e:
                queue_message(f"WARNING: IMU read error: {e}")

            time.sleep(self.POLL_INTERVAL)

    # ── Public API ───────────────────────────────────────────────────────

    def start(self):
        """Start the background polling thread."""
        if not self._sensor_ok:
            queue_message("WARNING: IMU sensor not initialized, cannot start")
            return False

        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="imu-poll",
            daemon=True
        )
        self._thread.start()
        queue_message("LOAD: IMU polling started (50Hz)")
        return True

    def stop(self):
        """Stop the polling thread and close I2C bus."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
        queue_message("INFO: IMU stopped")

    def get_reading(self):
        """Return the latest sensor reading (thread-safe copy)."""
        with self._lock:
            return dict(self._reading)

    def get_posture(self):
        """Return the current posture classification."""
        with self._lock:
            return self._posture

    def get_physical_state(self):
        """Return the current physical state (resting, held, knocked_over)."""
        return self._physical_state

    def get_sensor_data(self):
        """Return dict for body_state sensor registration.

        Called by BodyStateManager._read_sensors() via registered callback.
        """
        with self._lock:
            # Clear recent event after 30s
            recent = self._recent_event
            if recent and time.time() - self._recent_event_time > 30:
                self._recent_event = None
                recent = None

            return {
                "imu_posture": self._posture,
                "imu_magnitude": round(self._reading["magnitude"], 2),
                "imu_tilt": round(self._reading["tilt"], 1),
                "imu_state": self._physical_state,
                "imu_event": recent,
            }

    @property
    def sensor_initialized(self):
        return self._sensor_ok
