"""
Module: BATTERY MONITOR - V3
Author: Charles-Olivier Dion (AtomikSpace)
Contact: atomikspace.labs@gmail.com
Copyright (c) 2026 Charles-Olivier Dion

This file is authored by Charles-Olivier Dion and is dual-licensed.

Non-Commercial License:
This file is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0).
You may use, modify, and redistribute this file for NON-COMMERCIAL purposes only, with attribution.

Commercial License:
Commercial use (including selling products, paid services, SaaS, subscriptions, Patreon rewards, or derivatives)
requires a separate written license from Charles-Olivier Dion (AtomikSpace).

This license applies only to this file and does not override licenses of other files in the repository.
"""
import time
import threading
import smbus2
from collections import deque
from modules.module_config import load_config

CONFIG = load_config()

# INA219 register addresses
_INA219_REG_CONFIG      = 0x00
_INA219_REG_SHUNT_V     = 0x01
_INA219_REG_BUS_V       = 0x02
_INA219_REG_POWER       = 0x03
_INA219_REG_CURRENT     = 0x04
_INA219_REG_CALIBRATION = 0x05

# INA219 constants
_INA219_ADDR        = 0x41
_INA219_SHUNT_R     = 0.1     # 100 mΩ shunt resistor
_INA219_CURRENT_LSB = 0.1     # mA per LSB (matches calibration value 4096)
_INA219_POWER_LSB   = 2.0     # mW per LSB (20 * current_LSB)


class BatteryModule:
    def __init__(self,
                 battery_capacity_mAh=CONFIG['BATTERY']['battery_capacity_mAh'],
                 battery_initial_voltage=CONFIG['BATTERY']['battery_initial_voltage'],
                 battery_cutoff_voltage=CONFIG['BATTERY']['battery_cutoff_voltage'],
                 auto_shutdown=CONFIG['BATTERY']['battery_auto_shutdown'],
                 smoothing_window=10):
        self.battery_capacity_mAh = battery_capacity_mAh
        self.battery_initial_voltage = battery_initial_voltage
        self.battery_cutoff_voltage = battery_cutoff_voltage
        self.auto_shutdown = auto_shutdown
        self.smoothing_window = smoothing_window
        self.current = 0.0
        self.voltage = 0.0
        self.power = 0.0
        self.battery_percentage = 0.0
        self.normalized_percentage = 0
        self.percentage_history = deque(maxlen=smoothing_window)
        self.is_running = False
        self.thread = None
        self.zero_percent_start_time = None
        self.shutdown_delay_seconds = 60

        self.voltage_history = deque(maxlen=15)
        self.baseline_voltage = None
        self.charging_state = "DISCHARGING"
        self.last_printed_state = None

        self.last_servo_activity_time = 0
        self.servo_cooldown_seconds = 10
        self.verbose = False

        try:
            self._bus = smbus2.SMBus(1)
            self._ina219_configure()
            # Verify sensor responds with a test read
            self._ina219_read_bus_voltage()
            self.sensor_initialized = True
            print("INA219 sensor detected")
        except Exception as e:
            print(f"INA219 sensor not detected: {e}")
            self._bus = None
            self.sensor_initialized = False

    def _ina219_configure(self):
        """Configure INA219: 32V bus range, ±320mV shunt, 12-bit, continuous."""
        # Config: BRNG=01(32V), PG=11(±320mV), BADC=0011(12-bit), SADC=011(12-bit), MODE=111(continuous)
        config = 0x399F
        self._bus.write_i2c_block_data(_INA219_ADDR, _INA219_REG_CONFIG,
                                       [(config >> 8) & 0xFF, config & 0xFF])
        time.sleep(0.01)
        # Calibration: CAL = trunc(0.04096 / (current_LSB * R_shunt))
        # With current_LSB=0.0001A and R_shunt=0.1Ω: CAL = 4096
        cal = 4096
        self._bus.write_i2c_block_data(_INA219_ADDR, _INA219_REG_CALIBRATION,
                                       [(cal >> 8) & 0xFF, cal & 0xFF])
        time.sleep(0.01)

    def _ina219_read_signed(self, reg):
        """Read a signed 16-bit big-endian value from INA219."""
        data = self._bus.read_i2c_block_data(_INA219_ADDR, reg, 2)
        value = (data[0] << 8) | data[1]
        if value >= 0x8000:
            value -= 0x10000
        return value

    def _ina219_read_unsigned(self, reg):
        """Read an unsigned 16-bit big-endian value from INA219."""
        data = self._bus.read_i2c_block_data(_INA219_ADDR, reg, 2)
        return (data[0] << 8) | data[1]

    def _ina219_read_bus_voltage(self):
        """Read bus voltage in volts. LSB = 4mV, bits [15:3]."""
        raw = self._ina219_read_unsigned(_INA219_REG_BUS_V)
        return (raw >> 3) * 0.004

    def _ina219_read_current(self):
        """Read calibrated current in mA."""
        raw = self._ina219_read_signed(_INA219_REG_CURRENT)
        return raw * _INA219_CURRENT_LSB

    def _ina219_read_power(self):
        """Read calibrated power in mW."""
        raw = self._ina219_read_unsigned(_INA219_REG_POWER)
        return raw * _INA219_POWER_LSB

    def signal_servo_activity(self):
        self.last_servo_activity_time = time.time()
        self.voltage_history.clear()

    def set_verbose(self, enabled):
        self.verbose = enabled

    def _is_servo_cooldown_active(self):
        return (time.time() - self.last_servo_activity_time) < self.servo_cooldown_seconds

    def calculate_battery_percentage(self, current_voltage):
        if current_voltage > self.battery_initial_voltage:
            current_voltage = self.battery_initial_voltage  
        elif current_voltage < self.battery_cutoff_voltage:
            current_voltage = self.battery_cutoff_voltage  
        percentage = ((current_voltage - self.battery_cutoff_voltage) / 
                (self.battery_initial_voltage - self.battery_cutoff_voltage)) * 100
        return round(percentage, 1)

    def normalize_percentage(self, percentage):
        self.percentage_history.append(percentage)
        if len(self.percentage_history) > 0:
            normalized = sum(self.percentage_history) / len(self.percentage_history)
            return int(normalized)  
        return int(percentage)

    def _get_smoothed_voltage(self):
        """Average of last 10 voltage readings to filter ±30mV noise."""
        if len(self.voltage_history) < 10:
            return None
        return sum(list(self.voltage_history)[-10:]) / 10

    def _update_charging_state(self):
        if self._is_servo_cooldown_active():
            return

        self.voltage_history.append(self.voltage)

        smoothed = self._get_smoothed_voltage()
        if smoothed is None:
            return

        # Initialize baseline from first smoothed reading
        if self.baseline_voltage is None:
            self.baseline_voltage = smoothed
            return

        was_charging = self.charging_state == "CHARGING"

        # Baseline tracks actual voltage via EMA, but only when NOT charging.
        # This keeps baseline at the "no-charger" voltage level.
        # When charging, baseline freezes so elevation stays high.
        if not was_charging:
            self.baseline_voltage += 0.02 * (smoothed - self.baseline_voltage)

        elevation = (smoothed - self.baseline_voltage) * 1000  # mV

        # Charging detection via elevation above frozen baseline.
        # Real charging: ~120mV above baseline (from profiling data).
        # Normal noise after smoothing: ~±10mV.
        # Enter charging at 80mV, exit at 30mV (hysteresis).
        if elevation > 80:
            self.charging_state = "CHARGING"
        elif was_charging and elevation > 30:
            self.charging_state = "CHARGING"
        elif self.current > 50:
            self.charging_state = "DISCHARGING"
        else:
            self.charging_state = "IDLE"

        if self.last_printed_state != self.charging_state:
            self.last_printed_state = self.charging_state

    def _monitoring_loop(self):
        print("Battery monitoring started")
        while self.is_running and self.sensor_initialized:
            try:
                self.voltage = self._ina219_read_bus_voltage()
                self.current = self._ina219_read_current()
                self.power = self._ina219_read_power()
                self.battery_percentage = self.calculate_battery_percentage(self.voltage)
                self.normalized_percentage = self.normalize_percentage(self.battery_percentage)
                
                self._update_charging_state()
                
                if self.verbose:
                    self.print_debug()

                if self.auto_shutdown and self.sensor_initialized:
                    if self.normalized_percentage <= 0:
                        if self.zero_percent_start_time is None:
                            self.zero_percent_start_time = time.time()
                        else:
                            elapsed = time.time() - self.zero_percent_start_time
                            if elapsed >= self.shutdown_delay_seconds:
                                self._initiate_shutdown()
                                break  
                    else:
                        if self.zero_percent_start_time is not None:
                            self.zero_percent_start_time = None

                time.sleep(0.5)

            except Exception as e:
                print(f"Battery monitoring error: {e}")
                time.sleep(5)

    def _initiate_shutdown(self):
        import subprocess
        import os
        try:
            subprocess.Popen(['sudo', 'shutdown', 'now'])
        except Exception as e:
            print(f"Failed to shutdown system: {e}")
        time.sleep(2)
        os._exit(0)

    def start(self):
        if not self.sensor_initialized:
            print("Cannot start battery monitoring: sensor not initialized")
            return False
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.thread.start()
            return True
        return False

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.thread:
                self.thread.join(timeout=2.0)
            return True
        return False

    def is_charging(self):
        return self.charging_state == "CHARGING"

    def get_battery_status(self):
        return {
            'current': self.current,  
            'voltage': self.voltage,  
            'power': self.power,  
            'percentage': self.battery_percentage,  
            'normalized_percentage': self.normalized_percentage,  
            'capacity': self.battery_capacity_mAh,  
            'is_charging': self.is_charging(),
            'charging_state': self.charging_state,
            'sensor_initialized': self.sensor_initialized  
        }

    def get_battery_percentage(self):
        return self.battery_percentage

    def get_normalized_percentage(self):
        return self.normalized_percentage

    def print_debug(self):
        baseline_str = f"{self.baseline_voltage:.3f}V" if self.baseline_voltage else "---"
        smoothed = self._get_smoothed_voltage()
        if smoothed and self.baseline_voltage:
            elevation = (smoothed - self.baseline_voltage) * 1000
        else:
            elevation = 0
        cooldown = "COOLDOWN" if self._is_servo_cooldown_active() else ""
        smooth_str = f"{smoothed:.3f}" if smoothed else "---"
        print(f"V: {self.voltage:.3f} (smooth: {smooth_str}, base: {baseline_str}, {elevation:+.0f}mV)  |  "
              f"I: {self.current:+.0f}mA  |  {self.charging_state} {cooldown}")


# Module-level singleton accessor for dashboard/chatui
_battery_instance = None

def set_battery_instance(instance):
    global _battery_instance
    _battery_instance = instance

def get_battery_status():
    if _battery_instance is not None:
        return _battery_instance.get_battery_status()
    return {'sensor_initialized': False}