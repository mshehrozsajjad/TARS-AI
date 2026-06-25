"""
INA219 Voltage & Current Test
-----------------------------
Prints bus voltage (V), shunt voltage (mV), current (mA), and power (mW)
in a loop at ~2 Hz.

Setup notes:
  - INA219 is at I2C address 0x41
  - Sits on the Pi's buck converter output (does NOT see servo current)
  - Expected idle readings: ~5V bus, ~400-800 mA depending on Pi load

Instructions:
1. Run this script and confirm voltage matches expected (5V from buck).
2. Confirm current reads something sensible for Pi idle.
3. Start/stop a heavy process (e.g., stress test) and watch current change.
4. If voltage reads 0 or current stays at 0, check wiring.

Press Ctrl+C to stop.
"""

import time
import smbus2

# INA219 registers
INA_ADDR = 0x41
REG_CONFIG = 0x00
REG_SHUNT_VOLTAGE = 0x01
REG_BUS_VOLTAGE = 0x02
REG_POWER = 0x03
REG_CURRENT = 0x04
REG_CALIBRATION = 0x05

# Default shunt resistor on most INA219 breakout boards
SHUNT_RESISTOR_OHMS = 0.1  # 100 mΩ (R100)


def read_word_signed(bus, addr, reg):
    """Read a signed 16-bit big-endian value."""
    data = bus.read_i2c_block_data(addr, reg, 2)
    value = (data[0] << 8) | data[1]
    if value >= 0x8000:
        value -= 0x10000
    return value


def read_word_unsigned(bus, addr, reg):
    """Read an unsigned 16-bit big-endian value."""
    data = bus.read_i2c_block_data(addr, reg, 2)
    return (data[0] << 8) | data[1]


def configure_ina219(bus):
    """
    Configure INA219 for 32V range, ±320mV shunt range, 12-bit, continuous.

    Config register bits (MSB first):
      [15]    RST = 0
      [14-13] BRNG = 01 (32V bus range)
      [12-11] PG = 11 (±320mV shunt range — widest, good for discovery)
      [10-9]  BADC = 11 (12-bit bus ADC)
      [8-6]   SADC = 011 (12-bit shunt ADC)
      [5-3]   MODE = 111 (continuous shunt + bus)

    = 0b0_01_11_0011_011_111 = 0x399F
    """
    config = 0x399F
    bus.write_i2c_block_data(INA_ADDR, REG_CONFIG, [(config >> 8) & 0xFF, config & 0xFF])
    time.sleep(0.01)

    # Calibration register: CAL = trunc(0.04096 / (current_LSB * R_shunt))
    # With R_shunt = 0.1Ω and current_LSB = 0.1 mA:
    #   CAL = trunc(0.04096 / (0.0001 * 0.1)) = trunc(4096) = 4096
    cal = 4096
    bus.write_i2c_block_data(INA_ADDR, REG_CALIBRATION, [(cal >> 8) & 0xFF, cal & 0xFF])
    time.sleep(0.01)


def main():
    bus = smbus2.SMBus(1)

    # Quick sanity: read config register to confirm chip responds
    raw_config = read_word_unsigned(bus, INA_ADDR, REG_CONFIG)
    print(f"INA219 at 0x{INA_ADDR:02X} — config register: 0x{raw_config:04X}")

    configure_ina219(bus)
    print("Configured: 32V range, ±320mV shunt, 12-bit, continuous\n")

    print("INA219 Power Monitor — ~2 Hz")
    print("=" * 70)
    print(f"{'Bus (V)':>10}  {'Shunt (mV)':>12}  {'Current (mA)':>14}  {'Power (mW)':>12}")
    print("-" * 70)

    # current_LSB = 0.1 mA (matches calibration above)
    CURRENT_LSB_MA = 0.1
    # power_LSB = 20 * current_LSB = 2 mW
    POWER_LSB_MW = 20 * CURRENT_LSB_MA

    try:
        while True:
            # Bus voltage: bits [15:3] contain the voltage, LSB = 4 mV
            raw_bus = read_word_unsigned(bus, INA_ADDR, REG_BUS_VOLTAGE)
            bus_voltage = (raw_bus >> 3) * 0.004  # 4 mV per LSB

            # Shunt voltage: signed, LSB = 10 µV
            raw_shunt = read_word_signed(bus, INA_ADDR, REG_SHUNT_VOLTAGE)
            shunt_voltage_mv = raw_shunt * 0.01  # 10 µV → mV

            # Current from calibrated register
            raw_current = read_word_signed(bus, INA_ADDR, REG_CURRENT)
            current_ma = raw_current * CURRENT_LSB_MA

            # Power from calibrated register
            raw_power = read_word_unsigned(bus, INA_ADDR, REG_POWER)
            power_mw = raw_power * POWER_LSB_MW

            # Also compute current from shunt voltage directly (for comparison)
            current_from_shunt = (shunt_voltage_mv / 1000.0) / SHUNT_RESISTOR_OHMS * 1000  # mA

            print(f"{bus_voltage:>10.3f}  {shunt_voltage_mv:>12.3f}  "
                  f"{current_ma:>14.1f}  {power_mw:>12.1f}", end="\r")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nFinal readings:")
        raw_bus = read_word_unsigned(bus, INA_ADDR, REG_BUS_VOLTAGE)
        bus_voltage = (raw_bus >> 3) * 0.004
        raw_shunt = read_word_signed(bus, INA_ADDR, REG_SHUNT_VOLTAGE)
        shunt_mv = raw_shunt * 0.01
        raw_current = read_word_signed(bus, INA_ADDR, REG_CURRENT)
        current_ma = raw_current * CURRENT_LSB_MA

        print(f"  Bus voltage:    {bus_voltage:.3f} V")
        print(f"  Shunt voltage:  {shunt_mv:.3f} mV")
        print(f"  Current:        {current_ma:.1f} mA")
        print(f"  Shunt resistor: {SHUNT_RESISTOR_OHMS} Ω (assumed — adjust if different)")
        print("\nIf voltage is ~5V, the INA219 is on the Pi buck output (as expected).")
        print("If current reads ~0 or negative, the shunt resistor value may be wrong,")
        print("or the sensor is on the high side without load through the shunt.")
        print("Done.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
