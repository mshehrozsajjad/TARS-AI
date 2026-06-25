"""
module_imu.py

MPU6050 IMU sensor module for TARS-AI physical awareness.

Reads 6-axis accelerometer + gyroscope data over I2C and exposes
orientation/posture to the body state system. Designed as a foundation
for Phase 2 event detection (pickup, fall, shake, tilt reactions).

The MPU6050 shares I2C bus 1 with the PCA9685 servo driver (0x40) and
optional INA219 battery sensor (0x41). Default address: 0x68.
"""

import math
import time
import threading
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

# Posture thresholds (degrees from upright)
TILT_THRESHOLD    = 30   # degrees — "tilted"
ON_SIDE_THRESHOLD = 60   # degrees — "on side"
INVERTED_THRESHOLD = 140  # degrees — "upside down"


# ── Manager ──────────────────────────────────────────────────────────────────

class IMUManager:
    """Reads MPU6050 sensor and exposes posture to body state."""

    POLL_INTERVAL = 0.02  # 50Hz

    def __init__(self, config, body_state_manager=None):
        global _instance
        _instance = self

        self._config = config
        imu_cfg = config.get("IMU", {})
        self._address = int(imu_cfg.get("imu_address", "0x68"), 16)

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
            "pitch": 0.0, "roll": 0.0,
        }

        # Derived posture
        self._posture = "upright"

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
                # Some clones report different WHO_AM_I values
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
    def _compute_orientation(ax, ay, az):
        """Compute pitch and roll from accelerometer (degrees)."""
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        roll = math.degrees(math.atan2(ay, az))
        return pitch, roll

    @staticmethod
    def _classify_posture(pitch, roll):
        """Classify posture from pitch/roll angles."""
        tilt_angle = math.sqrt(pitch * pitch + roll * roll)

        if tilt_angle > INVERTED_THRESHOLD:
            return "upside down"
        elif tilt_angle > ON_SIDE_THRESHOLD:
            return "on side"
        elif tilt_angle > TILT_THRESHOLD:
            return "tilted"
        else:
            return "upright"

    # ── Polling loop ─────────────────────────────────────────────────────

    def _poll_loop(self):
        """Background thread: read sensor and update state."""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while self._running:
            try:
                ax, ay, az = self._read_raw_accel()
                gx, gy, gz = self._read_raw_gyro()

                magnitude = math.sqrt(ax * ax + ay * ay + az * az)
                pitch, roll = self._compute_orientation(ax, ay, az)
                posture = self._classify_posture(pitch, roll)

                with self._lock:
                    self._reading = {
                        "ax": ax, "ay": ay, "az": az,
                        "gx": gx, "gy": gy, "gz": gz,
                        "magnitude": magnitude,
                        "pitch": pitch, "roll": roll,
                    }
                    self._posture = posture

                consecutive_errors = 0

            except OSError as e:
                consecutive_errors += 1
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

    def get_sensor_data(self):
        """Return dict for body_state sensor registration.

        Called by BodyStateManager._read_sensors() via registered callback.
        """
        with self._lock:
            return {
                "imu_posture": self._posture,
                "imu_magnitude": round(self._reading["magnitude"], 2),
            }

    @property
    def sensor_initialized(self):
        return self._sensor_ok
