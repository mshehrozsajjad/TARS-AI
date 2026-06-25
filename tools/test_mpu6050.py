"""
MPU6050 Axis Identification Test
--------------------------------
Prints accelerometer X/Y/Z at ~10 Hz.

Instructions:
1. Place TARS upright on a flat surface. Note the axis reading ~1g (9.8 m/s²).
   That axis is your "vertical" (gravity direction).
2. Tilt TARS forward — whichever horizontal axis changes most is "pitch".
3. Tilt TARS sideways — the remaining axis is "roll".
4. Write down which raw axis (X/Y/Z) maps to pitch, roll, and vertical.

Press Ctrl+C to stop.
"""

import time
import struct
import smbus2

# MPU6050 registers
MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
ACCEL_CONFIG = 0x1C
WHO_AM_I = 0x75

# Accelerometer scale: ±2g (default) → 16384 LSB/g
ACCEL_SCALE = 16384.0


def read_word(bus, addr, reg):
    """Read a signed 16-bit value (big-endian) from two consecutive registers."""
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) | low
    # Convert to signed
    if value >= 0x8000:
        value -= 0x10000
    return value


def main():
    bus = smbus2.SMBus(1)

    # Verify chip identity
    who = bus.read_byte_data(MPU_ADDR, WHO_AM_I)
    print(f"WHO_AM_I register: 0x{who:02X} (expect 0x68 for MPU6050)")
    if who != 0x68:
        print(f"WARNING: Unexpected WHO_AM_I value. Got 0x{who:02X}, expected 0x68.")
        print("         Continuing anyway — may still work if wiring is correct.\n")

    # Wake up the MPU6050 (clear sleep bit)
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.1)

    # Set accelerometer to ±2g (bits 3-4 = 00)
    bus.write_byte_data(MPU_ADDR, ACCEL_CONFIG, 0x00)
    time.sleep(0.05)

    print("\nMPU6050 Accelerometer Test — ~10 Hz")
    print("=" * 60)
    print(f"{'X (g)':>10}  {'Y (g)':>10}  {'Z (g)':>10}  {'Magnitude':>10}")
    print("-" * 60)

    try:
        while True:
            ax = read_word(bus, MPU_ADDR, ACCEL_XOUT_H) / ACCEL_SCALE
            ay = read_word(bus, MPU_ADDR, ACCEL_XOUT_H + 2) / ACCEL_SCALE
            az = read_word(bus, MPU_ADDR, ACCEL_XOUT_H + 4) / ACCEL_SCALE
            mag = (ax**2 + ay**2 + az**2) ** 0.5

            print(f"{ax:>10.3f}  {ay:>10.3f}  {az:>10.3f}  {mag:>10.3f}", end="\r")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nFinal snapshot:")
        ax = read_word(bus, MPU_ADDR, ACCEL_XOUT_H) / ACCEL_SCALE
        ay = read_word(bus, MPU_ADDR, ACCEL_XOUT_H + 2) / ACCEL_SCALE
        az = read_word(bus, MPU_ADDR, ACCEL_XOUT_H + 4) / ACCEL_SCALE
        print(f"  X = {ax:+.3f} g")
        print(f"  Y = {ay:+.3f} g")
        print(f"  Z = {az:+.3f} g")
        print(f"\nThe axis closest to ±1.0 g while upright is your vertical (gravity).")
        print("Done.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
