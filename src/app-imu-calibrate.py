#!/usr/bin/env python3
"""
app-imu-calibrate.py

IMU-based self-calibration tool for TARS-AI.

Uses the MPU6050 accelerometer to automatically level TARS by adjusting
leg servo positions.  Computes optimal perfectXOffset values for config.ini
so every subsequent movement starts from a level stance.

Algorithm:
  1. Move to neutral, read baseline tilt
  2. Small probe move to measure sensitivity (degrees per % of servo offset)
  3. If responsive: calculate target offset directly, apply, verify, fine-tune
  4. If not responsive: skip axis (tilt is structural, not servo-correctable)

Both legs stay on the ground at all times — max offset is clamped to ±8%
(matching the ~20 PWM unit range of perfectXOffset in typical configs).

Run on the Pi with TARS standing on a flat, level surface.

Usage:
    python3 app-imu-calibrate.py            # calibrate and show offsets
    python3 app-imu-calibrate.py --save     # calibrate and write to config.ini
    python3 app-imu-calibrate.py --dry      # read current tilt only, no movement
"""

import argparse
import math
import sys
import time

import smbus2
import board
import busio
from adafruit_pca9685 import PCA9685

from modules.module_config import load_config
import modules.module_servoctl as servoctl


# ── MPU6050 registers ────────────────────────────────────────────────────────

REG_PWR_MGMT_1   = 0x6B
REG_ACCEL_XOUT_H = 0x3B
REG_WHO_AM_I     = 0x75
ACCEL_SCALE      = 16384.0   # LSB/g at ±2 g range


# ── Calibration tuning ───────────────────────────────────────────────────────

TOLERANCE       = 1.5    # degrees — considered "level enough"
MAX_OFFSET      = 8.0    # max % offset from neutral (keeps both legs on ground)
SETTLE_TIME     = 1.0    # seconds to wait after a servo move
NUM_SAMPLES     = 50     # IMU readings to average per measurement
SAMPLE_INTERVAL = 0.02   # seconds between samples (50 Hz)
PROBE_OFFSET    = 3.0    # % offset for sensitivity measurement probe
MIN_SENSITIVITY = 0.1    # deg/% — below this, axis is not servo-correctable
REFINE_ITERS    = 5      # max fine-tuning iterations after initial correction
UNDERSHOOT      = 0.7    # multiply corrections by this to avoid overshoot


# ═══════════════════════════════════════════════════════════════════════════════
#  IMU Reader — minimal MPU6050 access for calibration
# ═══════════════════════════════════════════════════════════════════════════════

class IMUReader:
    """Direct MPU6050 reader.  No event detection — just averaged accel reads."""

    def __init__(self, address=0x68):
        self.address = address
        self.bus = smbus2.SMBus(1)

        who = self.bus.read_byte_data(address, REG_WHO_AM_I)
        if who not in (0x68, 0x72, 0x73, 0x98):
            print(f"       WARNING: WHO_AM_I = 0x{who:02X} (expected 0x68)")

        # Wake sensor from sleep
        self.bus.write_byte_data(address, REG_PWR_MGMT_1, 0x00)
        time.sleep(0.1)

    def read_accel(self):
        """Burst-read accelerometer → (ax, ay, az) in g, or None if corrupt."""
        data = self.bus.read_i2c_block_data(self.address, REG_ACCEL_XOUT_H, 6)
        raws = [(data[i] << 8 | data[i + 1]) for i in range(0, 6, 2)]
        signed = [(v - 0x10000) if v >= 0x8000 else v for v in raws]
        ax, ay, az = [s / ACCEL_SCALE for s in signed]

        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if mag > 4.0 or mag < 0.1:
            return None
        return ax, ay, az

    def read_averaged(self, n=NUM_SAMPLES):
        """Take *n* good readings and return their mean (ax, ay, az), or None."""
        good = []
        for _ in range(n + 20):
            r = self.read_accel()
            if r is not None:
                good.append(r)
            if len(good) >= n:
                break
            time.sleep(SAMPLE_INTERVAL)

        if len(good) < n // 2:
            return None

        ax = sum(g[0] for g in good) / len(good)
        ay = sum(g[1] for g in good) / len(good)
        az = sum(g[2] for g in good) / len(good)
        return ax, ay, az

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Tilt math
# ═══════════════════════════════════════════════════════════════════════════════
#
# TARS IMU mounting:  X = up,  Y = lateral (left / right),  Z = forward / back
#
# Roll  = tilt in the Y direction (side-to-side) → corrected via leg heights
# Pitch = tilt in the Z direction (front / back) → corrected via leg swing

def compute_roll_pitch(ax, ay, az):
    """Decompose gravity vector into roll and pitch (degrees)."""
    roll  = math.degrees(math.atan2(ay, ax))
    pitch = math.degrees(math.atan2(az, ax))
    return roll, pitch


def total_tilt(ax, ay, az):
    """Total angle from vertical (degrees)."""
    mag = math.sqrt(ax * ax + ay * ay + az * az)
    if mag < 0.1:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, ax / mag))))


# ═══════════════════════════════════════════════════════════════════════════════
#  Sensitivity measurement
# ═══════════════════════════════════════════════════════════════════════════════
#
# A small probe move measures how many degrees of tilt change per 1% of
# servo offset.  This tells us:
#   - The correct direction (sign) for correction
#   - Whether this axis is responsive enough to calibrate
#   - The exact offset to apply (direct calculation, not iterative guessing)

def _apply_offset(axis, offset):
    """Move servos to the given offset from neutral.  Both legs stay near 50%."""
    if axis == "roll":
        lh = max(1.0, min(99.0, 50.0 - offset))
        rh = max(1.0, min(99.0, 50.0 + offset))
        servoctl.move_legs(lh, rh, None, None, 0.4)
    else:
        ls = max(1.0, min(99.0, 50.0 - offset))
        rs = max(1.0, min(99.0, 50.0 - offset))
        servoctl.move_legs(None, None, ls, rs, 0.4)


def _read_axis(imu, axis):
    """Read IMU and return the relevant axis value in degrees, or None."""
    r = imu.read_averaged()
    if r is None:
        return None
    roll, pitch = compute_roll_pitch(*r)
    return roll if axis == "roll" else pitch


def _measure_sensitivity(imu, axis):
    """Probe to measure sensitivity and correction direction.

    Returns (sign, sensitivity_deg_per_pct):
        sign: +1 if positive offset reduces tilt, -1 if negative offset does
        sensitivity: absolute degrees of tilt change per 1% of offset
        Returns (0, 0.0) if IMU reads fail.
    """
    # Read baseline at neutral
    baseline = _read_axis(imu, axis)
    if baseline is None:
        return 0, 0.0

    # Probe: apply a small positive offset
    _apply_offset(axis, PROBE_OFFSET)
    time.sleep(SETTLE_TIME)

    probed = _read_axis(imu, axis)

    # Return to neutral
    _apply_offset(axis, 0)
    time.sleep(SETTLE_TIME)

    if probed is None:
        return 0, 0.0

    delta_tilt = probed - baseline            # how tilt changed
    sensitivity = abs(delta_tilt) / PROBE_OFFSET  # deg per %

    # If positive offset reduced the absolute tilt, sign is +1
    sign = 1 if abs(probed) < abs(baseline) else -1

    return sign, sensitivity


# ═══════════════════════════════════════════════════════════════════════════════
#  Correction — measure, calculate, apply, refine
# ═══════════════════════════════════════════════════════════════════════════════

def _correct_axis(imu, axis):
    """Level one axis.  Returns the final offset (%) or 0 if not possible.

    Steps:
      1. Measure sensitivity with a probe move
      2. Calculate the target offset directly from tilt / sensitivity
      3. Apply (clamped to MAX_OFFSET so legs stay on ground)
      4. Fine-tune with a few small iterations
    """
    label = "Roll" if axis == "roll" else "Pitch"

    # ── Step 1: measure sensitivity ──────────────────────────────────────
    print(f"       Measuring sensitivity...")
    sign, sensitivity = _measure_sensitivity(imu, axis)

    if sensitivity < MIN_SENSITIVITY:
        print(f"       Sensitivity = {sensitivity:.3f} deg/% (below {MIN_SENSITIVITY})")
        print(f"       {label} is not servo-correctable — skipping")
        return 0.0

    print(f"       Sensitivity = {sensitivity:.2f} deg/%")
    print(f"       Direction:    {'positive' if sign == 1 else 'negative'} offset reduces tilt")

    # ── Step 2: read current error and calculate target ──────────────────
    error = _read_axis(imu, axis)
    if error is None:
        print(f"       IMU read failed")
        return 0.0

    target_offset = (-error / sensitivity) * sign * UNDERSHOOT
    target_offset = max(-MAX_OFFSET, min(MAX_OFFSET, target_offset))

    print(f"       Current {label.lower()} = {error:+.2f} deg")
    print(f"       Target offset = {target_offset:+.1f}%")
    print()

    # ── Step 3: apply and verify ─────────────────────────────────────────
    _apply_offset(axis, target_offset)
    time.sleep(SETTLE_TIME)

    offset = target_offset

    # ── Step 4: fine-tune ────────────────────────────────────────────────
    for i in range(REFINE_ITERS):
        error = _read_axis(imu, axis)
        if error is None:
            print(f"  #{i + 1}  IMU read failed, skipping")
            time.sleep(0.5)
            continue

        print(f"  #{i + 1}  {label.lower()} = {error:+6.2f} deg   offset = {offset:+5.1f}%")

        if abs(error) < TOLERANCE:
            print(f"       Level (within {TOLERANCE} deg)")
            break

        # Would the correction push us past MAX_OFFSET?
        correction = (-error / sensitivity) * sign * UNDERSHOOT
        new_offset = offset + correction
        if abs(new_offset) > MAX_OFFSET:
            clamped = max(-MAX_OFFSET, min(MAX_OFFSET, new_offset))
            print(f"       Clamped to {clamped:+.1f}% (max ±{MAX_OFFSET}%)")
            if abs(clamped - offset) < 0.1:
                print(f"       At limit — best achievable with servos")
                break
            new_offset = clamped

        offset = new_offset
        _apply_offset(axis, offset)
        time.sleep(SETTLE_TIME)
    else:
        print(f"       Fine-tuning done ({REFINE_ITERS} iterations)")

    return offset


# ═══════════════════════════════════════════════════════════════════════════════
#  Offset computation — convert calibrated % position → config.ini offsets
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_config_offsets(height_offset, swing_offset):
    """Convert percentage offsets to config.ini perfectXOffset values.

    The servo controller applies offsets as:
        leftUpHeight     = base + perfectLeftHeightOffset
        rightUpHeight    = base - perfectRightHeightOffset
        forwardLeftLeg   = base + perfectLeftLegOffset
        forwardRightLeg  = base + perfectRightLegOffset

    We compute the PWM delta between the calibrated neutral and the original
    neutral, then add that to the existing config offsets.
    """

    def pct_to_pwm(pct, min_val, max_val):
        return min_val + (max_val - min_val) * (pct - 1.0) / 99.0

    # PWM at calibrated positions
    cal_lh = pct_to_pwm(50.0 - height_offset,
                         servoctl.leftUpHeight, servoctl.leftDownHeight)
    cal_rh = pct_to_pwm(50.0 + height_offset,
                         servoctl.rightUpHeight, servoctl.rightDownHeight)
    cal_ls = pct_to_pwm(50.0 - swing_offset,
                         servoctl.forwardLeftLeg, servoctl.backLeftLeg)
    cal_rs = pct_to_pwm(50.0 - swing_offset,
                         servoctl.forwardRightLeg, servoctl.backRightLeg)

    # PWM at current neutral (50 %)
    neut_lh = servoctl.leftNeutralHeight
    neut_rh = servoctl.rightNeutralHeight
    neut_ls = servoctl.neutralLeftLeg
    neut_rs = servoctl.neutralRightLeg

    # PWM deltas
    d_lh = cal_lh - neut_lh
    d_rh = cal_rh - neut_rh
    d_ls = cal_ls - neut_ls
    d_rs = cal_rs - neut_rs

    # New config offsets (additive on top of whatever is already set)
    #   Left height / swing offsets are ADDED to base values
    #   Right height offset is SUBTRACTED from base values
    #   Right swing offset is ADDED to base values
    new_lh = servoctl.perfectLeftHeightOffset  + int(round(d_lh))
    new_rh = servoctl.perfectRightHeightOffset - int(round(d_rh))
    new_ls = servoctl.perfectLeftLegOffset     + int(round(d_ls))
    new_rs = servoctl.perfectRightLegOffset    + int(round(d_rs))

    return new_lh, new_rh, new_ls, new_rs


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TARS IMU self-calibration — automatically level the legs")
    parser.add_argument("--save", action="store_true",
                        help="Write computed offsets to config.ini")
    parser.add_argument("--dry", action="store_true",
                        help="Read current tilt only, no servo movement")
    args = parser.parse_args()

    config = load_config()
    imu_cfg  = config.get("IMU", {})
    imu_addr = int(imu_cfg.get("imu_address", "0x68"), 16)

    print()
    print("=" * 60)
    print("  TARS IMU Self-Calibration")
    print("=" * 60)

    # ── Hardware init ────────────────────────────────────────────────────

    print()
    print("[1/6] Initializing hardware...")

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c, address=0x40)
        pca.frequency = 50
        servoctl.pca = pca
        print("       PCA9685 servo driver  ... ok")
    except Exception as e:
        print(f"       PCA9685 servo driver  ... FAILED: {e}")
        sys.exit(1)

    try:
        imu = IMUReader(address=imu_addr)
        print(f"       MPU6050 IMU (0x{imu_addr:02X})   ... ok")
    except Exception as e:
        print(f"       MPU6050 IMU           ... FAILED: {e}")
        sys.exit(1)

    # ── Dry-run mode ─────────────────────────────────────────────────────

    if args.dry:
        print()
        print("[DRY] Reading current tilt (no servo movement)...")
        time.sleep(0.5)
        r = imu.read_averaged()
        if r is None:
            print("       Could not get a stable reading.")
        else:
            roll, pitch = compute_roll_pitch(*r)
            tilt = total_tilt(*r)
            print(f"       Roll  (side-to-side): {roll:+.1f} deg")
            print(f"       Pitch (front-back):   {pitch:+.1f} deg")
            print(f"       Total tilt:           {tilt:.1f} deg")
        imu.close()
        print()
        return

    # ── Interactive prompt ───────────────────────────────────────────────

    print()
    print("  Place TARS on a flat, level surface.")
    print("  Do not touch it during calibration.")
    print()
    input("  Press Enter to begin... ")
    print()

    # ── Move to neutral ──────────────────────────────────────────────────

    print("[2/6] Moving to neutral position...")
    servoctl.move_legs(50, 50, 50, 50, 0.4)
    time.sleep(SETTLE_TIME * 1.5)
    print("       At neutral (50, 50, 50, 50)")

    # ── Baseline tilt ────────────────────────────────────────────────────

    print()
    print("[3/6] Reading baseline tilt...")
    base = imu.read_averaged()
    if base is None:
        print("       ERROR: cannot get stable IMU reading")
        servoctl.disable_all_servos()
        imu.close()
        sys.exit(1)

    init_roll, init_pitch = compute_roll_pitch(*base)
    init_tilt = total_tilt(*base)
    print(f"       Roll  (side-to-side): {init_roll:+.1f} deg")
    print(f"       Pitch (front-back):   {init_pitch:+.1f} deg")
    print(f"       Total tilt:           {init_tilt:.1f} deg")

    needs_roll  = abs(init_roll)  >= TOLERANCE
    needs_pitch = abs(init_pitch) >= TOLERANCE

    if not needs_roll and not needs_pitch:
        print()
        print("       Already level — no correction needed.")
        servoctl.disable_all_servos()
        imu.close()
        print()
        return

    # ── Correct each axis ────────────────────────────────────────────────

    print()
    print("[4/6] Correcting roll (leg heights)..." if needs_roll else
          "[4/6] Roll within tolerance, skipping")

    height_offset = 0.0
    if needs_roll:
        height_offset = _correct_axis(imu, "roll")

    print()
    print("[5/6] Correcting pitch (leg swing)..." if needs_pitch else
          "[5/6] Pitch within tolerance, skipping")

    swing_offset = 0.0
    if needs_pitch:
        swing_offset = _correct_axis(imu, "pitch")

    # ── Final measurement ────────────────────────────────────────────────

    print()
    print("[6/6] Final measurement...")
    time.sleep(SETTLE_TIME)
    final = imu.read_averaged()

    if final:
        fin_roll, fin_pitch = compute_roll_pitch(*final)
        fin_tilt = total_tilt(*final)
        print(f"       Roll:  {init_roll:+.1f} -> {fin_roll:+.1f} deg")
        print(f"       Pitch: {init_pitch:+.1f} -> {fin_pitch:+.1f} deg")
        print(f"       Total: {init_tilt:.1f} -> {fin_tilt:.1f} deg")

    # ── Compute and display config offsets ────────────────────────────────

    new_lh, new_rh, new_ls, new_rs = _compute_config_offsets(
        height_offset, swing_offset)

    print()
    print("  Computed config.ini offsets:")
    print(f"    perfectLeftHeightOffset  = {new_lh}")
    print(f"    perfectRightHeightOffset = {new_rh}")
    print(f"    perfectLeftLegOffset     = {new_ls}")
    print(f"    perfectRightLegOffset    = {new_rs}")

    negligible = abs(height_offset) < 0.5 and abs(swing_offset) < 0.5

    if negligible:
        print()
        print("  Offsets are negligible — no config changes needed.")
    elif args.save:
        print()
        print("  Saving to config.ini...")
        from modules.module_config import update_config_from_web_ui
        result = update_config_from_web_ui({
            "SERVO": {
                "perfectLeftHeightOffset":  str(new_lh),
                "perfectRightHeightOffset": str(new_rh),
                "perfectLeftLegOffset":     str(new_ls),
                "perfectRightLegOffset":    str(new_rs),
            }
        })
        if result["success"]:
            print("       Saved successfully.")
        else:
            print(f"       Save failed: {result['message']}")
    else:
        print()
        print("  Run with --save to write these to config.ini")

    # ── Cleanup ──────────────────────────────────────────────────────────

    servoctl.disable_all_servos()
    imu.close()

    print()
    print("  Calibration complete.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
