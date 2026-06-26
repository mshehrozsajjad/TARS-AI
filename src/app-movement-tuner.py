#!/usr/bin/env python3
"""
app-movement-tuner.py

Movement tuning tool for TARS-AI.

Uses the MPU6050 IMU to score movement stability, then systematically
tries parameter variations to find the best-performing sequence.
Saves optimized parameters so movement functions use them at full speed
with zero runtime overhead.

Usage:
    python3 app-movement-tuner.py step_forward         # tune step_forward
    python3 app-movement-tuner.py walk_forward          # tune walk_forward
    python3 app-movement-tuner.py step_forward --runs 5 # more profiling runs
    python3 app-movement-tuner.py --list                # show tunable movements
    python3 app-movement-tuner.py --profile step_forward  # profile only, no tuning
"""

import argparse
import sys
import time

import board
import busio
from adafruit_pca9685 import PCA9685

from modules.module_config import load_config
import modules.module_servoctl as servoctl
from modules.module_movement_tuner import (
    DEFAULTS,
    get_default_steps,
    load_tuned_params,
    save_tuned_params,
    invalidate_cache,
    profile_movement,
    generate_variations,
    find_worst_steps,
    score_movement,
    IMURecorder,
    score_per_step,
)


# ── Display helpers ──────────────────────────────────────────────────────────

def print_step_table(steps, step_scores):
    """Print a formatted table of steps with their stability scores."""
    print()
    print(f"  {'Step':>4}  {'LH':>3} {'RH':>3} {'LS':>3} {'RS':>3} {'Spd':>4}  "
          f"{'MaxTilt':>8} {'Wobble':>7} {'Yaw':>7}  Status")
    print("  " + "-" * 68)

    for i, step in enumerate(steps):
        lh, rh, ls, rs, speed = step
        score = step_scores[i] if i < len(step_scores) else None

        if score and score["readings"] > 0:
            mt = score["max_tilt"]
            wb = score["wobble"]
            yaw = score.get("avg_yaw", 0)

            if mt > 20:
                status = "!! tilt"
            elif abs(yaw) > 10:
                status = "!! drift"
            elif mt > 12 or abs(yaw) > 5:
                status = "!  bad"
            else:
                status = "   ok"

            print(f"  {i + 1:>4}  {lh:>3} {rh:>3} {ls:>3} {rs:>3} {speed:>4.1f}  "
                  f"{mt:>7.1f}° {wb:>6.1f}° {yaw:>+6.1f}°/s  {status}")
        else:
            print(f"  {i + 1:>4}  {lh:>3} {rh:>3} {ls:>3} {rs:>3} {speed:>4.1f}  "
                  f"{'--':>8} {'--':>7} {'--':>7}  no data")


def print_comparison(old_steps, old_scores, new_steps, new_scores, step_idx):
    """Print before/after for a single optimized step."""
    old = old_steps[step_idx]
    new = new_steps[step_idx]
    os_data = old_scores[step_idx] if step_idx < len(old_scores) else None
    ns_data = new_scores[step_idx] if step_idx < len(new_scores) else None

    old_mt = os_data["max_tilt"] if os_data else "?"
    new_mt = ns_data["max_tilt"] if ns_data else "?"

    print(f"    Step {step_idx + 1}: "
          f"({old[0]},{old[1]},{old[2]},{old[3]} @{old[4]:.1f}) -> "
          f"({new[0]},{new[1]},{new[2]},{new[3]} @{new[4]:.1f})")
    print(f"    Max tilt: {old_mt}° -> {new_mt}°")


# ── IMU initialization ──────────────────────────────────────────────────────

def init_imu(config):
    """Initialize IMU for the tuner.  Returns True if successful."""
    imu_cfg = config.get("IMU", {})
    imu_addr = int(imu_cfg.get("imu_address", "0x68"), 16)

    # Try to use the full IMUManager (best: proven bus contention handling)
    try:
        from modules.module_imu import IMUManager, get_imu_manager
        if get_imu_manager() is None:
            mgr = IMUManager(config)
            if mgr.sensor_initialized:
                mgr.start()
                print(f"       MPU6050 IMU (0x{imu_addr:02X})   ... ok (IMUManager)")
                return True
        else:
            print(f"       MPU6050 IMU (0x{imu_addr:02X})   ... ok (already running)")
            return True
    except Exception as e:
        print(f"       IMUManager failed ({e}), trying direct...")

    # Fallback: the IMURecorder's _DirectIMUReader will handle it
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        bus.read_byte_data(imu_addr, 0x75)
        bus.close()
        print(f"       MPU6050 IMU (0x{imu_addr:02X})   ... ok (direct)")
        return True
    except Exception as e:
        print(f"       MPU6050 IMU           ... FAILED: {e}")
        return False


# ── Profile-only mode ────────────────────────────────────────────────────────

def do_profile(movement_name, steps, num_runs):
    """Profile a movement and display results without optimizing."""
    print(f"\n  Profiling '{movement_name}' ({num_runs} runs)...")
    print("  TARS will move. Do not touch it.\n")

    avg_scores, overall, per_run = profile_movement(steps, num_runs=num_runs)

    for run_idx, (run_steps, run_score) in enumerate(per_run):
        worst_step = max(run_steps, key=lambda s: s["max_tilt"]) if run_steps else None
        worst_info = (f"max_tilt={worst_step['max_tilt']:.1f}° at step {worst_step['step'] + 1}"
                      if worst_step else "no data")
        print(f"  Run {run_idx + 1}: score={run_score:>3}  {worst_info}")

    print(f"\n  Average score: {overall}/100")
    print_step_table(steps, avg_scores)

    return avg_scores, overall


# ── Optimization ─────────────────────────────────────────────────────────────

def _run_single_trial(var_steps):
    """Run a movement sequence once with IMU recording.  Returns (step_scores, overall_score)."""
    servoctl.move_legs(50, 50, 50, 50, 0.5)
    time.sleep(2.0)  # longer settle — let I2C bus recover from prior movement

    recorder = IMURecorder()
    if not recorder.start():
        return [], 0

    servoctl.MOVING = True
    try:
        for i, step in enumerate(var_steps):
            recorder.mark_step(i)
            servoctl.move_legs(step[0], step[1], step[2], step[3], step[4])
        time.sleep(0.5)  # capture settling after last step
    finally:
        servoctl.MOVING = False

    recordings, markers = recorder.stop()
    step_scores = score_per_step(recordings, markers)
    overall = score_movement(step_scores)
    return step_scores, overall


def do_optimize(movement_name, steps, num_runs, max_steps_to_fix=3):
    """Profile, fix drift with global swing bias, fix worst steps, save."""

    # Phase 1: Profile baseline
    print(f"\n[1/5] Profiling baseline ({num_runs} runs)...")
    print("       TARS will move. Do not touch it.\n")

    avg_scores, baseline_score, per_run = profile_movement(
        steps, num_runs=num_runs)

    for run_idx, (run_steps, run_score) in enumerate(per_run):
        worst_step = max(run_steps, key=lambda s: s["max_tilt"]) if run_steps else None
        worst_info = (f"max_tilt={worst_step['max_tilt']:.1f}° at step {worst_step['step'] + 1}"
                      if worst_step else "no data")
        print(f"  Run {run_idx + 1}: score={run_score:>3}  {worst_info}")

    print(f"\n  Baseline score: {baseline_score}/100")
    print_step_table(steps, avg_scores)

    current_steps = [list(s) for s in steps]  # deep copy

    # Phase 2: Fix drift with global swing bias
    # Adjusts left vs right swing across ALL steps at once to counteract
    # mechanical asymmetry that makes TARS turn instead of walking straight.
    print(f"\n[2/5] Correcting drift (global swing bias)...")

    # Check if there's meaningful yaw drift
    avg_yaw = sum(s.get("avg_yaw", 0) for s in avg_scores) / max(len(avg_scores), 1)
    print(f"       Average yaw rate: {avg_yaw:+.1f}°/s", end="")

    if abs(avg_yaw) < 3.0:
        print(" — minimal drift, skipping")
    else:
        drift_dir = "right" if avg_yaw > 0 else "left"
        print(f" — drifting {drift_dir}")
        print()

        best_bias = 0
        best_score = baseline_score

        for bias in [-6, -4, -2, +2, +4, +6]:
            # Apply swing bias: shift left swing by +bias, right swing by -bias
            biased_steps = []
            for step in current_steps:
                lh, rh, ls, rs, speed = step
                new_ls = max(1, min(99, ls + bias))
                new_rs = max(1, min(99, rs - bias))
                biased_steps.append([lh, rh, new_ls, new_rs, speed])

            trial_scores, trial_overall = _run_single_trial(biased_steps)

            # Get yaw from this trial
            trial_yaw = sum(s.get("avg_yaw", 0) for s in trial_scores) / max(len(trial_scores), 1)

            improved = trial_overall > best_score
            marker = " *best*" if improved else ""
            print(f"    bias L{bias:+d} R{-bias:+d}"
                  f" .... score={trial_overall:>3}  yaw={trial_yaw:+.1f}°/s{marker}")

            if improved:
                best_score = trial_overall
                best_bias = bias

        if best_bias != 0:
            print(f"\n    -> Best bias: L{best_bias:+d} R{-best_bias:+d}")
            for i, step in enumerate(current_steps):
                current_steps[i][2] = max(1, min(99, step[2] + best_bias))
                current_steps[i][3] = max(1, min(99, step[3] - best_bias))
        else:
            print(f"\n    -> No bias improved the score, keeping symmetric")

    # Phase 3: Identify and fix worst steps (per-step tuning)
    print(f"\n[3/5] Optimizing worst steps...")

    # Re-profile with current (possibly bias-corrected) steps
    avg_scores, current_score, _ = profile_movement(
        current_steps, num_runs=2, reset_pause=2.0)

    worst_indices = find_worst_steps(avg_scores, threshold=8.0)
    if not worst_indices:
        print("       No problematic steps found (all below 8° max tilt).")
    else:
        worst_indices = worst_indices[:max_steps_to_fix]

        for target_idx in worst_indices:
            target_score = avg_scores[target_idx]
            step_vals = current_steps[target_idx]
            print(f"\n  Step {target_idx + 1} "
                  f"({step_vals[0]},{step_vals[1]},{step_vals[2]},{step_vals[3]} "
                  f"@{step_vals[4]:.1f}) — avg max_tilt={target_score['max_tilt']:.1f}°")

            variations = generate_variations(current_steps, target_idx)
            best_variation = None
            best_overall = current_score

            for desc, var_steps in variations:
                trial_scores, trial_overall = _run_single_trial(var_steps)

                var_tilt = 999
                if target_idx < len(trial_scores):
                    var_tilt = trial_scores[target_idx]["max_tilt"]

                improved = trial_overall > best_overall
                marker = " *best*" if improved else ""
                print(f"    {desc:.<30} tilt={var_tilt:>5.1f}° score={trial_overall:>3}{marker}")

                if improved:
                    best_overall = trial_overall
                    best_variation = var_steps

            if best_variation:
                new_step = best_variation[target_idx]
                current_steps = best_variation
                print(f"    -> Best: ({new_step[0]},{new_step[1]},{new_step[2]},"
                      f"{new_step[3]} @{new_step[4]:.1f})")
            else:
                print(f"    -> No improvement found, keeping original")

    # Phase 4: Validate optimized sequence
    print(f"\n[4/5] Validating optimized sequence ({num_runs} runs)...")

    val_scores, val_overall, val_runs = profile_movement(
        current_steps, num_runs=num_runs)

    for run_idx, (run_steps, run_score) in enumerate(val_runs):
        print(f"  Run {run_idx + 1}: score={run_score:>3}")

    print(f"\n  Optimized score: {val_overall}/100 (was {baseline_score}/100)")
    print_step_table(current_steps, val_scores)

    # Phase 5: Save
    print(f"\n[5/5] Saving...")

    if val_overall >= baseline_score:
        save_tuned_params(movement_name, current_steps, score=val_overall)
        print(f"       Saved to character movement_params.json")
        print(f"       TARS will use these for '{movement_name}' from now on.")
    else:
        print(f"       Optimized version scored WORSE ({val_overall} < {baseline_score}).")
        print(f"       Keeping original parameters.")
        current_steps = steps

    return current_steps, val_overall


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TARS Movement Tuner — optimize movements using IMU feedback")
    parser.add_argument("movement", nargs="?",
                        help="Movement name to tune (e.g., step_forward)")
    parser.add_argument("--list", action="store_true",
                        help="List all tunable movements")
    parser.add_argument("--profile", metavar="NAME",
                        help="Profile a movement without optimizing")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of profiling runs (default: 3)")
    parser.add_argument("--reset", metavar="NAME",
                        help="Delete tuned params for a movement (revert to defaults)")
    args = parser.parse_args()

    # ── List mode ────────────────────────────────────────────────────────
    if args.list:
        print("\nTunable movements:")
        for name in sorted(DEFAULTS.keys()):
            tuned = load_tuned_params(name)
            status = " (tuned)" if tuned else ""
            print(f"  {name}{status}")
        print()
        return

    # ── Reset mode ───────────────────────────────────────────────────────
    if args.reset:
        invalidate_cache()
        import json, os
        from modules.module_movement_tuner import _params_path
        path = _params_path()
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if args.reset in data:
                del data[args.reset]
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Reset '{args.reset}' to default parameters.")
            else:
                print(f"'{args.reset}' has no tuned parameters.")
        except FileNotFoundError:
            print("No tuned parameters file exists.")
        return

    # ── Determine movement name ──────────────────────────────────────────
    movement_name = args.movement or args.profile
    if not movement_name:
        parser.print_help()
        return

    steps = get_default_steps(movement_name)
    if steps is None:
        print(f"\nERROR: Unknown movement '{movement_name}'")
        print(f"Available: {', '.join(sorted(DEFAULTS.keys()))}")
        sys.exit(1)

    # Check if already tuned
    tuned = load_tuned_params(movement_name)
    if tuned:
        print(f"\nNote: '{movement_name}' has tuned parameters. "
              f"Using those as baseline.")
        print(f"      Run with --reset {movement_name} first to start from defaults.")
        steps = tuned

    # ── Hardware init ────────────────────────────────────────────────────
    config = load_config()

    print()
    print("=" * 60)
    print(f"  TARS Movement Tuner — {movement_name}")
    print("=" * 60)
    print()
    print("  Initializing hardware...")

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c, address=0x40)
        pca.frequency = 50
        servoctl.pca = pca
        print("       PCA9685 servo driver  ... ok")
    except Exception as e:
        print(f"       PCA9685 servo driver  ... FAILED: {e}")
        sys.exit(1)

    if not init_imu(config):
        sys.exit(1)

    print()
    print("  Place TARS on a flat surface. It will move repeatedly.")
    print("  Stand it back up if it falls during a trial.")
    print()
    input("  Press Enter to begin... ")

    # ── Profile or Optimize ──────────────────────────────────────────────
    if args.profile:
        do_profile(movement_name, steps, args.runs)
    else:
        do_optimize(movement_name, steps, args.runs)

    # ── Cleanup ──────────────────────────────────────────────────────────
    servoctl.move_legs(50, 50, 50, 50, 0.5)
    time.sleep(0.5)
    servoctl.disable_all_servos()

    print()
    print("  Done.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
