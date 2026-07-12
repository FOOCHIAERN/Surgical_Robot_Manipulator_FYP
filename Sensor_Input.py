#!/usr/bin/env python3
"""
sensor_reader.py (Diagnostic Edition)
=====================================
"""

import time
import sys
import traceback

# --- CRITICAL SYSTEM CHECK ---
try:
    import smbus2
except ImportError:
    print("\nCRITICAL ERROR: The 'smbus2' library is not installed.")
    print("Please run this command in your terminal first:")
    print("  pip3 install smbus2\n")
    sys.exit(1)

# --- I2C Addresses ---
I2C_BUS       = 1
PCA9548A_ADDR = 0x70
AS5600_ADDR   = 0x36
ADS1115_ADDR  = 0x48  

AS5600_RAW_ANGLE_HIGH = 0x0C  

MUX_BASE     = 0
MUX_SHOULDER = 1
MUX_ELBOW    = 2
MUX_ADS1115  = 3

ACS712_SENSITIVITY = 0.185
ACS712_ZERO_VOLTAGE = 2.50  

class SensorTree:
    def __init__(self):
        print(f"Attempting to open I2C Bus {I2C_BUS}...")
        try:
            self.bus = smbus2.SMBus(I2C_BUS)
            print("I2C Bus opened successfully.")
        except PermissionError:
            print("\nCRITICAL ERROR: Permission denied accessing I2C.")
            print("Try running the script with sudo: sudo python3 sensor_reader.py\n")
            raise
        except FileNotFoundError:
            print("\nCRITICAL ERROR: I2C Bus 1 not found hardware-wise.")
            print("This usually means I2C is disabled in raspi-config!\n")
            raise

    def select_mux_channel(self, channel: int) -> bool:
        """Attempts to switch the MUX channel. Returns True if successful."""
        try:
            self.bus.write_byte(PCA9548A_ADDR, 1 << channel)
            time.sleep(0.001)  # Settling time
            return True
        except OSError:
            # Catch the Remote I/O error so the script keeps running
            return False

    def read_encoder_angle(self, channel: int) -> float:
        if not self.select_mux_channel(channel):
            return -999.0  # Use -999 to clearly signify a MUX communication failure
        try:
            data = self.bus.read_i2c_block_data(AS5600_ADDR, AS5600_RAW_ANGLE_HIGH, 2)
            high, low = data[0], data[1]
            raw_counts = ((high & 0x0F) << 8) | low
            return (raw_counts / 4096.0) * 360.0
        except Exception:
            return -999.0

    def read_current_sensor(self, ain_pin: int) -> float:
        if not self.select_mux_channel(MUX_ADS1115):
            return -999.0
        mux_bits = {0: 0x4000, 1: 0x5000, 2: 0x6000}[ain_pin]
        config = 0x8000 | mux_bits | 0x0000 | 0x0100 | 0x0083
        config_swapped = ((config & 0xFF) << 8) | ((config >> 8) & 0xFF)
        
        try:
            self.bus.write_word_data(ADS1115_ADDR, 0x01, config_swapped)
            time.sleep(0.015)  
            raw_word = self.bus.read_word_data(ADS1115_ADDR, 0x00)
            raw_adc = ((raw_word & 0xFF) << 8) | ((raw_word >> 8) & 0xFF)
            if raw_adc > 32767:
                raw_adc -= 65536
            voltage = (raw_adc / 32767.0) * 6.144
            return (voltage - ACS712_ZERO_VOLTAGE) / ACS712_SENSITIVITY
        except Exception:
            return -999.0

    def close(self):
        try:
            self.bus.write_byte(PCA9548A_ADDR, 0x00)
        except Exception:
            pass
        self.bus.close()


# --- Main Wrapper ---
if __name__ == "__main__":
    sensors = None
    try:
        print("Starting Sensor Tree System...")
        sensors = SensorTree()
        print("Reading Sensors... Press Ctrl+C to stop.\n")
        
        while True:
            base_deg     = sensors.read_encoder_angle(MUX_BASE)
            shoulder_deg = sensors.read_encoder_angle(MUX_SHOULDER)
            elbow_deg    = sensors.read_encoder_angle(MUX_ELBOW)
            
            base_amps     = sensors.read_current_sensor(0)  
            shoulder_amps = sensors.read_current_sensor(1)  
            elbow_amps    = sensors.read_current_sensor(2)  
            
            print(f"BASE     | Pos: {base_deg:6.2f}° | Cur: {base_amps:5.2f} A")
            print(f"SHOULDER | Pos: {shoulder_deg:6.2f}° | Cur: {shoulder_amps:5.2f} A")
            print(f"ELBOW    | Pos: {elbow_deg:6.2f}° | Cur: {elbow_amps:5.2f} A")
            print("-" * 45)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print("\n!!! SCRIPT CRASHED DURING RUNTIME !!!")
        traceback.print_exc()
    finally:
        if sensors is not None:
            sensors.close()
        sys.exit(0)