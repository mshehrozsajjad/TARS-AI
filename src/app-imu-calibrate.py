#!/usr/bin/env python3
"""
app-imu-calibrate.py

IMU-based self-calibration tool for TARS-AI.

Uses the MPU6050 accelerometer to automatically level TARS by iteratively
adjusting leg servo positions.  Computes optimal perfectXOffset values
for config.ini so every subsequent movement starts from a level stance.

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
MAX_ITERATIONS  = 20     # per axis before giving up
GAIN            = 0.5    # degrees-of-tilt → percent servo adjustment
MIN_STEP        = 0.3    # smallest meaningful adjustment (prevents stalling)
MAX_OFFSET      = 35.0   # max accumulated offset from 50 % (safety clamp)
SETTLE_TIME     = 1.0    # seconds to wait after a servo move
NUM_SAMPLES     = 50     # IMU readings to average per measurement
SAMPLE_INTERVAL = 0.02   # seconds between samples (50 Hz)
DIRECTION_PROBE = 4      # percent offset used to detect correction direction


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

    # ── single read ──────────────────────────────────────────────────────

    def read_accel(self):
        """Burst-read accelerometer → (ax, ay, az) in g, or None if corrupt."""
        data = self.bus.read_i2c_block_data(self.address, REG_ACCEL_XOUT_H, 6)
        raws = [(data[i] << 8 | data[i + 1]) for i in range(0, 6, 2)]
        signed = [(v - 0x10000) if v >= 0x8000 else v for v in raws]
        ax, ay, az = [s / ACCEL_SCALE for s in signed]

        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if mag > 4.0 or mag < 0.1:
            return None                        # I2C bus noise / corruption
        return ax, ay, az

    # ── averaged read ────────────────────────────────────────────────────

    def read_averaged(self, n=NUM_SAMPLES):
        """Take *n* good readings and return their mean (ax, ay, az), or None."""
        good = []
        for _ in range(n + 20):                # extra attempts for discards
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
#  Direction detection
# ═══════════════════════════════════════════════════════════════════════════════
#
# The sign mapping from "servo % change" to "tilt change" depends on IMU
# mounting and servo wiring, which can vary between builds.  A one-shot probe
# move determines the correct sign for each axis automatically.

def _detect_sign(imu, axis):
    """Return +1 or −1: how a positive servo offset maps to tilt change.

    Makes a small test move, measures the response, returns to neutral.
    """
    base = imu.read_averaged()
    if base is None:
        return 1
    base_roll, base_pitch = compute_roll_pitch(*base)
    baseline = base_roll if axis == "roll" else base_pitch

    # Probe move
    if axis == "roll":
        servoctl.move_legs(50 - DIRECTION_PROBE, 50 + DIRECTION_PROBE,
                           None, None, 0.5)
    else:
        servoctl.move_legs(None, None,
                           50 - DIRECTION_PROBE, 50 - DIRECTION_PROBE, 0.5)
    time.sleep(SETTLE_TIME)

    probe = imu.read_averaged()

    # Return to neutral
    servoctl.move_legs(50, 50, 50, 50, 0.5)
    time.sleep(SETTLE_TIME)

    if probe is None:
        return 1
    probe_roll, probe_pitch = compute_roll_pitch(*probe)
    probed = probe_roll if axis == "roll" else probe_pitch

    # If the probe reduced the absolute error, the sign is correct (+1)
    return 1 if abs(probed) < abs(baseline) else -1


# ═══════════════════════════════════════════════════════════════════════════════
#  Correction loop (proportional controller with oscillation damping)
# ═══════════════════════════════════════════════════════════════════════════════

def _correct_axis(imu, axis, sign):
    """Iteratively zero out one tilt axis.  Returns the final offset (%)."""
    label = "roll" if axis == "roll" else "pitch"
    offset = 0.0
    prev_error = None
    gain = GAIN

    for i in range(MAX_ITERATIONS):
        r = imu.read_averaged()
        if r is None:
            print(f"  #{i + 1:2d}  IMU read failed, retrying...")
            time.sleep(0.5)
            continue

        roll, pitch = compute_roll_pitch(*r)
        error = roll if axis == "roll" else pitch

        print(f"  #{i + 1:2d}  {label} = {error:+6.2f} deg   offset = {offset:+5.1f}%")

        if abs(error) < TOLERANCE:
            print(f"       Level (within {TOLERANCE} deg)")
            break

        # Dampen gain if the error flips sign (oscillating)
        if prev_error is not None and error * prev_error < 0:
            gain *= 0.6

        adj = error * gain * sign
        if 0 < abs(adj) < MIN_STEP:
            adj = MIN_STEP if adj > 0 else -MIN_STEP

        offset += adj
        offset = max(-MAX_OFFSET, min(MAX_OFFSET, offset))

        # Apply to servos
        if axis == "roll":
            lh = max(1.0, min(99.0, 50.0 - offset))
            rh = max(1.0, min(99.0, 50.0 + offset))
            servoctl.move_legs(lh, rh, None, None, 0.4)
        else:
            ls = max(1.0, min(99.0, 50.0 - offset))
            rs = max(1.0, min(99.0, 50.0 - offset))
            servoctl.move_legs(None, None, ls, rs, 0.4)

        time.sleep(SETTLE_TIME)
        prev_error = error
    else:
        print("       Max iterations reached")

    return offset


# ═══════════════════════════════════════════════════════════════════════════════
#  Offset computation — convert calibrated % position → config.ini offsets
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_config_offsets(height_offset, swing_offset):
    """Convert percentage offsets to config.ini perfectXOffset values.

    The servo controller applies offsets as:
        leftUpHeight     = base + perfectLeftHeightOffset
        rightUpHeight    = base − perfectRightHeightOffset
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

    # ── Detect correction directions ─────────────────────────────────────

    print()
    print("[4/6] Detecting correction directions...")

    roll_sign  = 1
    pitch_sign = 1

    if needs_roll:
        roll_sign = _detect_sign(imu, "roll")
        direction = "standard" if roll_sign == 1 else "inverted"
        print(f"       Roll  correction: {direction}")

    if needs_pitch:
        pitch_sign = _detect_sign(imu, "pitch")
        direction = "standard" if pitch_sign == 1 else "inverted"
        print(f"       Pitch correction: {direction}")

    # ── Correction loops ─────────────────────────────────────────────────

    print()
    print("[5/6] Correcting...")

    height_offset = 0.0
    swing_offset  = 0.0

    if needs_roll:
        print()
        print("  Roll correction (adjusting leg heights)")
        print("  " + "-" * 46)
        height_offset = _correct_axis(imu, "roll", roll_sign)

    if needs_pitch:
        print()
        print("  Pitch correction (adjusting leg swing)")
        print("  " + "-" * 46)
        swing_offset = _correct_axis(imu, "pitch", pitch_sign)

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
