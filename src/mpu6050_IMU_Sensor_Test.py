"""
MPU6050 IMU Sensor Calibration Tool

Standalone script to read and display raw MPU6050 accelerometer and gyroscope
values for baseline calibration. Run directly on Pi without the TARS application.

Sensor mounting on TARS:
  X-axis = vertical (up when positive)
  Y-axis = lateral (left/right)
  Z-axis = forward/back (forward when negative)

Usage:
    python mpu6050_IMU_Sensor_Test.py

Hold TARS in different positions and observe how the values change:
  - Rest on table (baseline)
  - Tilt left / right / forward / back
  - Pick up and hold
  - Shake
  - Set down
"""

import time
import math
import smbus2

# MPU6050 I2C address (0x68 default, 0x69 if AD0 pin is high)
MPU6050_ADDR = 0x68

# MPU6050 register addresses
REG_PWR_MGMT_1   = 0x6B
REG_ACCEL_XOUT_H = 0x3B
REG_GYRO_XOUT_H  = 0x43
REG_WHO_AM_I     = 0x75

# Scale factors (default ±2g / ±250°/s ranges)
ACCEL_SCALE = 16384.0  # LSB/g at ±2g
GYRO_SCALE  = 131.0    # LSB/(°/s) at ±250°/s


def read_signed_16(bus, addr, reg):
    """Read a signed 16-bit big-endian value from two consecutive registers."""
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    if value >= 0x8000:
        value -= 0x10000
    return value


def read_accel(bus):
    """Read accelerometer X, Y, Z in g."""
    ax = read_signed_16(bus, MPU6050_ADDR, REG_ACCEL_XOUT_H) / ACCEL_SCALE
    ay = read_signed_16(bus, MPU6050_ADDR, REG_ACCEL_XOUT_H + 2) / ACCEL_SCALE
    az = read_signed_16(bus, MPU6050_ADDR, REG_ACCEL_XOUT_H + 4) / ACCEL_SCALE
    return ax, ay, az


def read_gyro(bus):
    """Read gyroscope X, Y, Z in °/s."""
    gx = read_signed_16(bus, MPU6050_ADDR, REG_GYRO_XOUT_H) / GYRO_SCALE
    gy = read_signed_16(bus, MPU6050_ADDR, REG_GYRO_XOUT_H + 2) / GYRO_SCALE
    gz = read_signed_16(bus, MPU6050_ADDR, REG_GYRO_XOUT_H + 4) / GYRO_SCALE
    return gx, gy, gz


def classify_posture(tilt):
    """Classify posture from tilt angle."""
    if tilt < 20:
        return "UPRIGHT"
    elif tilt < 55:
        return "TILTED"
    elif tilt < 135:
        return "ON SIDE"
    else:
        return "UPSIDE DOWN"


def main():
    print("MPU6050 IMU Sensor Calibration Tool")
    print("=" * 60)
    print("Sensor axes: X=vertical, Y=lateral, Z=forward/back")
    print()
    print("Test positions:")
    print("  1. Rest on table       (tilt ~0, mag ~1.0g)")
    print("  2. Tilt forward/back   (tilt increases)")
    print("  3. Tilt left/right     (tilt increases)")
    print("  4. Pick up and hold    (mag spikes, gyro spikes)")
    print("  5. Shake               (mag oscillates, gyro high)")
    print("  6. Lay on back/side    (tilt ~90)")
    print()
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        bus = smbus2.SMBus(1)
    except Exception as e:
        print(f"\nERROR: Could not open I2C bus 1: {e}")
        print("Make sure I2C is enabled (sudo raspi-config) and you have permissions.")
        return

    # Check WHO_AM_I register (should return 0x68 for MPU6050)
    try:
        who = bus.read_byte_data(MPU6050_ADDR, REG_WHO_AM_I)
        print(f"\nWHO_AM_I: 0x{who:02X} (expected 0x68 for MPU6050)")
    except Exception as e:
        print(f"\nERROR: MPU6050 not found at address 0x{MPU6050_ADDR:02X}: {e}")
        print("Check wiring and run 'i2cdetect -y 1' to verify.")
        bus.close()
        return

    # Wake up sensor (clear sleep bit in PWR_MGMT_1)
    try:
        bus.write_byte_data(MPU6050_ADDR, REG_PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        print("Sensor initialized\n")
    except Exception as e:
        print(f"\nERROR: Could not initialize MPU6050: {e}")
        bus.close()
        return

    try:
        while True:
            ax, ay, az = read_accel(bus)
            gx, gy, gz = read_gyro(bus)

            magnitude = math.sqrt(ax * ax + ay * ay + az * az)

            # Tilt from vertical — angle between accel vector and X-axis (up)
            if magnitude > 0.1:
                tilt = math.degrees(math.acos(max(-1, min(1, ax / magnitude))))
            else:
                tilt = 0.0

            posture = classify_posture(tilt)

            # Gyro total rotation rate
            gyro_total = math.sqrt(gx * gx + gy * gy + gz * gz)

            print(
                f"mag={magnitude:4.2f}g  tilt={tilt:5.1f}  "
                f"[{posture:^10s}]  "
                f"gyro={gyro_total:5.1f}/s  "
                f"| ax={ax:+5.2f} ay={ay:+5.2f} az={az:+5.2f}",
                end="          \r"
            )

            time.sleep(0.05)  # 20Hz display refresh

    except KeyboardInterrupt:
        print("\n\nDone!")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
