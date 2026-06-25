"""
MPU6050 IMU Sensor Calibration Tool

Standalone script to read and display raw MPU6050 accelerometer and gyroscope
values for baseline calibration. Run directly on Pi without the TARS application.

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
REG_ACCEL_XOUT_H = 0x3B  # 6 bytes: XH, XL, YH, YL, ZH, ZL
REG_GYRO_XOUT_H  = 0x43  # 6 bytes: XH, XL, YH, YL, ZH, ZL
REG_ACCEL_CONFIG  = 0x1C
REG_GYRO_CONFIG   = 0x1B
REG_WHO_AM_I      = 0x75

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


def compute_orientation(ax, ay, az):
    """Compute pitch and roll from accelerometer (degrees)."""
    # pitch = rotation around Y axis, roll = rotation around X axis
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    roll = math.degrees(math.atan2(ay, az))
    return pitch, roll


def main():
    print("MPU6050 IMU Sensor Calibration Tool")
    print("=" * 60)
    print("Hold TARS in different positions to observe values.")
    print("  1. Rest on table        (baseline)")
    print("  2. Tilt left/right      (roll changes)")
    print("  3. Tilt forward/back    (pitch changes)")
    print("  4. Pick up and hold     (magnitude + orientation shift)")
    print("  5. Shake                (high accel spikes)")
    print("  6. Set down             (impact spike → stable)")
    print("  7. Drop / freefall      (magnitude → ~0g)")
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
        print("Sensor initialized (awake, ±2g accel, ±250°/s gyro)")
    except Exception as e:
        print(f"\nERROR: Could not initialize MPU6050: {e}")
        bus.close()
        return

    print()
    print(f"{'ACCEL (g)':^30s}  |  {'MAG':^6s}  |  {'ORIENT (°)':^18s}  |  {'GYRO (°/s)':^30s}")
    print("-" * 100)

    try:
        while True:
            ax, ay, az = read_accel(bus)
            gx, gy, gz = read_gyro(bus)

            magnitude = math.sqrt(ax * ax + ay * ay + az * az)
            pitch, roll = compute_orientation(ax, ay, az)

            print(
                f"ax={ax:+6.2f}  ay={ay:+6.2f}  az={az:+6.2f}  |  "
                f"{magnitude:5.2f}g  |  "
                f"pitch={pitch:+6.1f}  roll={roll:+6.1f}  |  "
                f"gx={gx:+7.1f}  gy={gy:+7.1f}  gz={gz:+7.1f}",
                end="\r"
            )

            time.sleep(0.05)  # 20Hz display refresh

    except KeyboardInterrupt:
        print("\n\nDone!")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
