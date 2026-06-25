"""
Servo Power Profiling via INA219
---------------------------------
Fires each leg servo individually while sampling INA219 at 20 Hz.
Then fires all 4 together. Produces a power baseline report.

Uses raw smbus2 for both PCA9685 and INA219 — zero extra dependencies.

Run from Pi:  python3 tools/profile_servo_power.py
"""

import time
import threading
import smbus2
import sys
import os

# ─── I2C addresses ───────────────────────────────────────────────
PCA_ADDR = 0x40
INA_ADDR = 0x41

# ─── PCA9685 registers ──────────────────────────────────────────
PCA_MODE1     = 0x00
PCA_PRESCALE  = 0xFE

# ─── INA219 registers ───────────────────────────────────────────
INA_CONFIG      = 0x00
INA_SHUNT_V     = 0x01
INA_BUS_V       = 0x02
INA_POWER       = 0x03
INA_CURRENT     = 0x04
INA_CALIBRATION = 0x05

SHUNT_R = 0.1          # 100 mΩ shunt resistor
CURRENT_LSB_MA = 0.1   # matches calibration value 4096

# ─── Servo channel layout (legs only, no arms) ──────────────────
# Physical wiring (ch1/ch2 swapped from config comments):
#   Ch0 = Left Height,  Ch1 = Left Swing,  Ch2 = Right Height,  Ch3 = Right Swing
SERVOS = {
    0: {"name": "Left Height",  "neutral": 360, "min": 230, "max": 470},  # 350+10 offset
    1: {"name": "Left Swing",   "neutral": 329, "min": 189, "max": 489},  # 300+29 offset
    2: {"name": "Right Height", "neutral": 270, "min": 140, "max": 380},  # 350-80 offset
    3: {"name": "Right Swing",  "neutral": 280, "min": 140, "max": 440},  # 300-20 offset
}


# ═════════════════════════════════════════════════════════════════
#  PCA9685 raw driver
# ═════════════════════════════════════════════════════════════════

def pca_init(bus):
    """Initialize PCA9685 at 50 Hz."""
    # Reset — set sleep bit
    bus.write_byte_data(PCA_ADDR, PCA_MODE1, 0x10)
    time.sleep(0.005)
    # Set prescale for 50 Hz: prescale = round(25MHz / (4096 * 50)) - 1 = 121
    bus.write_byte_data(PCA_ADDR, PCA_PRESCALE, 121)
    # Wake up
    bus.write_byte_data(PCA_ADDR, PCA_MODE1, 0x00)
    time.sleep(0.005)
    # Auto-increment
    bus.write_byte_data(PCA_ADDR, PCA_MODE1, 0x20)
    time.sleep(0.005)


def pca_set_pwm(bus, channel, pulse_value):
    """
    Set servo PWM using the same formula as module_servoctl.py:
      pulse_us = 500 + (pulse_value / 600) * 2000
      duty_4096 = int((pulse_us / 20000) * 4096)
    """
    pulse_us = 500.0 + (pulse_value / 600.0) * 2000.0
    off_count = int((pulse_us / 20000.0) * 4096.0)
    on_count = 0

    reg = 0x06 + 4 * channel
    bus.write_i2c_block_data(PCA_ADDR, reg, [
        on_count & 0xFF, (on_count >> 8) & 0xFF,
        off_count & 0xFF, (off_count >> 8) & 0xFF,
    ])


def pca_disable_channel(bus, channel):
    """Turn off a servo channel (no PWM output)."""
    reg = 0x06 + 4 * channel
    bus.write_i2c_block_data(PCA_ADDR, reg, [0, 0, 0, 0])


def pca_disable_all(bus):
    for ch in range(16):
        pca_disable_channel(bus, ch)


# ═════════════════════════════════════════════════════════════════
#  INA219 raw driver
# ═════════════════════════════════════════════════════════════════

def ina_init(bus):
    """Configure INA219: 32V, ±320mV shunt, 12-bit, continuous."""
    config = 0x399F
    bus.write_i2c_block_data(INA_ADDR, INA_CONFIG, [(config >> 8) & 0xFF, config & 0xFF])
    time.sleep(0.01)
    cal = 4096
    bus.write_i2c_block_data(INA_ADDR, INA_CALIBRATION, [(cal >> 8) & 0xFF, cal & 0xFF])
    time.sleep(0.01)


def ina_read(bus):
    """Read voltage (V), current (mA), power (mW) from INA219."""
    # Bus voltage
    data = bus.read_i2c_block_data(INA_ADDR, INA_BUS_V, 2)
    raw_bus = (data[0] << 8) | data[1]
    voltage = (raw_bus >> 3) * 0.004

    # Shunt voltage → current
    data = bus.read_i2c_block_data(INA_ADDR, INA_SHUNT_V, 2)
    raw_shunt = (data[0] << 8) | data[1]
    if raw_shunt >= 0x8000:
        raw_shunt -= 0x10000
    shunt_mv = raw_shunt * 0.01

    # Calibrated current
    data = bus.read_i2c_block_data(INA_ADDR, INA_CURRENT, 2)
    raw_current = (data[0] << 8) | data[1]
    if raw_current >= 0x8000:
        raw_current -= 0x10000
    current_ma = raw_current * CURRENT_LSB_MA

    # Power
    data = bus.read_i2c_block_data(INA_ADDR, INA_POWER, 2)
    raw_power = (data[0] << 8) | data[1]
    power_mw = raw_power * 20 * CURRENT_LSB_MA

    return voltage, current_ma, power_mw, shunt_mv


# ═════════════════════════════════════════════════════════════════
#  Sampling thread
# ═════════════════════════════════════════════════════════════════

class PowerSampler:
    """Background INA219 sampler at ~20 Hz."""

    def __init__(self, bus):
        self._bus = bus
        self._samples = []      # list of (timestamp, voltage, current_ma, power_mw)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._phase = "idle"    # current label for the phase

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def set_phase(self, label):
        with self._lock:
            self._phase = label

    def _loop(self):
        while self._running:
            try:
                v, i, p, _ = ina_read(self._bus)
                t = time.time()
                with self._lock:
                    self._samples.append((t, v, i, p, self._phase))
            except Exception:
                pass
            time.sleep(0.05)   # 20 Hz

    def get_samples(self, phase=None):
        with self._lock:
            if phase is None:
                return list(self._samples)
            return [(t, v, i, p, ph) for t, v, i, p, ph in self._samples if ph == phase]

    def clear(self):
        with self._lock:
            self._samples.clear()

    def get_phase_stats(self, phase):
        samples = self.get_samples(phase)
        if not samples:
            return None
        voltages  = [s[1] for s in samples]
        currents  = [s[2] for s in samples]
        powers    = [s[3] for s in samples]
        return {
            "count":   len(samples),
            "v_avg":   sum(voltages) / len(voltages),
            "v_min":   min(voltages),
            "v_max":   max(voltages),
            "i_avg":   sum(currents) / len(currents),
            "i_min":   min(currents),
            "i_max":   max(currents),
            "p_avg":   sum(powers) / len(powers),
            "p_max":   max(powers),
        }


# ═════════════════════════════════════════════════════════════════
#  Servo movement helpers
# ═════════════════════════════════════════════════════════════════

def sweep_servo(bus, channel, start, end, speed=0.5):
    """
    Sweep a servo from start to end PWM value.
    speed: 0.0 = slow, 1.0 = fast (matches module_servoctl convention).
    """
    step = 1 if end > start else -1
    delay = 0.02 * (1.0 - speed)
    pos = start
    while pos != end:
        pca_set_pwm(bus, channel, pos)
        pos += step
        time.sleep(delay)
    pca_set_pwm(bus, channel, end)


def move_to_neutral(bus, channel):
    info = SERVOS[channel]
    current = info.get("_current", info["neutral"])
    sweep_servo(bus, channel, current, info["neutral"], speed=0.4)
    info["_current"] = info["neutral"]


# ═════════════════════════════════════════════════════════════════
#  Main profiling sequence
# ═════════════════════════════════════════════════════════════════

def main():
    bus = smbus2.SMBus(1)

    print("=" * 70)
    print("  SERVO POWER PROFILING")
    print("=" * 70)
    print()

    # ── Init hardware ────────────────────────────────────────────
    print("Initializing PCA9685...", end=" ", flush=True)
    pca_init(bus)
    print("OK")

    print("Initializing INA219...", end=" ", flush=True)
    ina_init(bus)
    v, i, _, _ = ina_read(bus)
    print(f"OK  (bus: {v:.2f}V, current: {i:.0f}mA)")
    print()

    # Start all servos at neutral (engages them so baseline includes holding torque)
    print("Moving all servos to neutral...", flush=True)
    for ch, info in SERVOS.items():
        pca_set_pwm(bus, ch, info["neutral"])
        info["_current"] = info["neutral"]
        time.sleep(0.1)
    time.sleep(1.0)

    sampler = PowerSampler(bus)
    sampler.start()

    # ── Phase 1: idle baseline (servos engaged at neutral) ───────
    print("\n[Phase 1] Idle baseline — servos holding neutral (5 seconds)")
    sampler.set_phase("idle_engaged")
    time.sleep(5.0)

    idle_stats = sampler.get_phase_stats("idle_engaged")
    if idle_stats:
        print(f"  Voltage:  {idle_stats['v_avg']:.3f}V  (min {idle_stats['v_min']:.3f}, max {idle_stats['v_max']:.3f})")
        print(f"  Current:  {idle_stats['i_avg']:.1f}mA  (min {idle_stats['i_min']:.1f}, max {idle_stats['i_max']:.1f})")
    else:
        print("  ERROR: No samples collected!")
        sampler.stop()
        bus.close()
        return

    # ── Phase 2: profile each servo individually ─────────────────
    print(f"\n[Phase 2] Individual servo profiles")
    print("-" * 70)

    servo_results = {}

    for ch, info in SERVOS.items():
        phase_name = f"servo_{ch}"
        label = f"Ch{ch} {info['name']}"
        print(f"\n  Testing {label}...")
        print(f"    Range: {info['min']} → {info['max']}  (neutral: {info['neutral']})")

        # Small settle before this servo
        time.sleep(0.5)

        sampler.set_phase(phase_name)

        # Sweep neutral → min
        print(f"    Sweeping neutral → min ({info['neutral']} → {info['min']})...", flush=True)
        sweep_servo(bus, ch, info["neutral"], info["min"], speed=0.5)
        info["_current"] = info["min"]
        time.sleep(0.3)

        # Sweep min → max (full range)
        print(f"    Sweeping min → max ({info['min']} → {info['max']})...", flush=True)
        sweep_servo(bus, ch, info["min"], info["max"], speed=0.5)
        info["_current"] = info["max"]
        time.sleep(0.3)

        # Sweep max → neutral
        print(f"    Returning to neutral ({info['max']} → {info['neutral']})...", flush=True)
        sweep_servo(bus, ch, info["max"], info["neutral"], speed=0.5)
        info["_current"] = info["neutral"]

        # Phase done — switch to settling
        sampler.set_phase(f"settle_{ch}")
        time.sleep(1.5)

        stats = sampler.get_phase_stats(phase_name)
        if stats:
            delta_i = stats["i_avg"] - idle_stats["i_avg"]
            delta_v = idle_stats["v_avg"] - stats["v_avg"]
            peak_i  = stats["i_max"] - idle_stats["i_avg"]
            servo_results[ch] = {
                "name": info["name"],
                "stats": stats,
                "delta_i_avg": delta_i,
                "delta_i_peak": peak_i,
                "delta_v": delta_v,
            }
            print(f"    ΔI avg: {delta_i:+.1f}mA  |  ΔI peak: {peak_i:+.1f}mA  |  ΔV: {delta_v*1000:+.1f}mV")
        else:
            print(f"    WARNING: No samples for {label}")

    # ── Phase 3: all servos simultaneously ───────────────────────
    print(f"\n[Phase 3] All 4 servos simultaneous sweep")
    print("-" * 70)

    time.sleep(1.0)
    sampler.set_phase("all_servos")

    # Sweep all to min
    print("  All → min...", flush=True)
    threads = []
    for ch, info in SERVOS.items():
        t = threading.Thread(target=sweep_servo, args=(bus, ch, info["neutral"], info["min"], 0.5))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for ch, info in SERVOS.items():
        info["_current"] = info["min"]
    time.sleep(0.3)

    # Sweep all to max
    print("  All → max...", flush=True)
    threads = []
    for ch, info in SERVOS.items():
        t = threading.Thread(target=sweep_servo, args=(bus, ch, info["min"], info["max"], 0.5))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for ch, info in SERVOS.items():
        info["_current"] = info["max"]
    time.sleep(0.3)

    # Return all to neutral
    print("  All → neutral...", flush=True)
    threads = []
    for ch, info in SERVOS.items():
        t = threading.Thread(target=sweep_servo, args=(bus, ch, info["max"], info["neutral"], 0.5))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for ch, info in SERVOS.items():
        info["_current"] = info["neutral"]

    sampler.set_phase("final_settle")
    time.sleep(2.0)

    all_stats = sampler.get_phase_stats("all_servos")

    # ── Phase 4: servos disengaged baseline ──────────────────────
    print(f"\n[Phase 4] Disengaged baseline — servos off (3 seconds)")
    pca_disable_all(bus)
    sampler.set_phase("idle_disengaged")
    time.sleep(3.0)
    disengaged_stats = sampler.get_phase_stats("idle_disengaged")

    sampler.stop()

    # ═════════════════════════════════════════════════════════════
    #  Summary report
    # ═════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 70)
    print("  POWER PROFILE SUMMARY")
    print("=" * 70)

    print(f"\n  Idle (servos engaged at neutral):")
    print(f"    Voltage:  {idle_stats['v_avg']:.3f}V")
    print(f"    Current:  {idle_stats['i_avg']:.1f}mA")
    print(f"    Power:    {idle_stats['p_avg']:.0f}mW")

    if disengaged_stats:
        print(f"\n  Idle (servos disengaged):")
        print(f"    Voltage:  {disengaged_stats['v_avg']:.3f}V")
        print(f"    Current:  {disengaged_stats['i_avg']:.1f}mA")
        print(f"    Power:    {disengaged_stats['p_avg']:.0f}mW")
        hold_current = idle_stats['i_avg'] - disengaged_stats['i_avg']
        print(f"    Holding torque overhead: {hold_current:+.1f}mA")

    print(f"\n  Individual servo draw (above idle baseline):")
    print(f"    {'Servo':<20}  {'ΔI avg':>10}  {'ΔI peak':>10}  {'ΔV avg':>10}")
    print(f"    {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    for ch in sorted(servo_results):
        r = servo_results[ch]
        print(f"    Ch{ch} {r['name']:<14}  {r['delta_i_avg']:>+9.1f}mA"
              f"  {r['delta_i_peak']:>+9.1f}mA  {r['delta_v']*1000:>+9.1f}mV")

    if all_stats:
        delta_all_i = all_stats["i_avg"] - idle_stats["i_avg"]
        delta_all_peak = all_stats["i_max"] - idle_stats["i_avg"]
        delta_all_v = idle_stats["v_avg"] - all_stats["v_avg"]
        print(f"\n  All 4 servos simultaneous:")
        print(f"    ΔI avg:   {delta_all_i:+.1f}mA")
        print(f"    ΔI peak:  {delta_all_peak:+.1f}mA")
        print(f"    ΔV avg:   {delta_all_v*1000:+.1f}mV")

    # Estimated per-servo average for battery module
    if servo_results:
        avg_per_servo = sum(r["delta_i_avg"] for r in servo_results.values()) / len(servo_results)
        peak_per_servo = max(r["delta_i_peak"] for r in servo_results.values())
        print(f"\n  ── Recommended constants for battery module ──")
        print(f"    SERVO_AVG_CURRENT_MA  = {max(0, avg_per_servo):.0f}   # average per active servo")
        print(f"    SERVO_PEAK_CURRENT_MA = {max(0, peak_per_servo):.0f}   # worst-case peak per servo")
        if all_stats:
            print(f"    ALL_SERVOS_AVG_MA     = {max(0, delta_all_i):.0f}   # all 4 servos moving")
            print(f"    ALL_SERVOS_PEAK_MA    = {max(0, delta_all_peak):.0f}   # all 4 peak")

    print(f"\n  Total samples collected: {len(sampler.get_samples())}")
    print()

    # ── CSV dump ─────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_power_log.csv")
    try:
        with open(csv_path, "w") as f:
            f.write("timestamp,voltage,current_ma,power_mw,phase\n")
            for t, v, i, p, ph in sampler.get_samples():
                f.write(f"{t:.3f},{v:.4f},{i:.2f},{p:.1f},{ph}\n")
        print(f"  Raw data saved to: {csv_path}")
    except Exception as e:
        print(f"  WARNING: Could not save CSV: {e}")

    pca_disable_all(bus)
    bus.close()
    print("\nDone. Servos disengaged.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Disabling servos...")
        try:
            bus = smbus2.SMBus(1)
            pca_disable_all(bus)
            bus.close()
        except Exception:
            pass
        print("Done.")
