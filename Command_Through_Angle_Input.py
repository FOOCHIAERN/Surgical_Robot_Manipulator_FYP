#!/usr/bin/env python3
"""
stepper_motor_controller.py
============================
Standalone Python Script: 3-DOF RRR Manipulator
Hardware: Raspberry Pi 3+/4 (64-bit OS), TMC2209 drivers

Pin Mapping (from hardware config):
  Motor 1 (Base)     : STEP=GPIO27, DIR=GPIO17, EN=GPIO22 | Gear Ratio=27, µstep=16
  Motor 2 (Shoulder) : STEP=GPIO24, DIR=GPIO25, EN=GPIO23 | Gear Ratio=19, µstep=16
  Motor 3 (Elbow)    : STEP=GPIO6,  DIR=GPIO13, EN=GPIO5  | Gear Ratio=14, µstep=16
python3 Stepper_Motor_Control+Homing.py
Terminal Input:
  Manual entry of degrees (Base, Shoulder, Elbow) for direct testing.
"""

import RPi.GPIO as GPIO
import time
import threading
import sys

# ──────────────────────────────────────────────
# HARDWARE CONSTANTS
# ──────────────────────────────────────────────

MICROSTEPS = 16
STEPS_PER_REV = 200  # Standard 1.8° stepper → 200 full steps/rev

MOTORS = {
    "base": {
        "step_pin": 27,
        "dir_pin":  17,
        "en_pin":   22,
        "gear_ratio": 27,
    },
    "shoulder": {
        "step_pin": 24,
        "dir_pin":  25,
        "en_pin":   23,
        "gear_ratio": 19,
    },
    "elbow": {
        "step_pin": 6,
        "dir_pin":  20,
        "en_pin":   5,
        "gear_ratio": 14,
    },
}

STEP_PULSE_WIDTH = 2e-6   # 2 µs HIGH pulse
STEP_PERIOD_MIN  = 50e-6  # Minimum step period


# ──────────────────────────────────────────────
# STEPPER MOTOR DRIVER
# ──────────────────────────────────────────────

class StepperMotor:
    """
    Software step/direction driver for one TMC2209-driven stepper motor.
    Operates in pure open-loop by tracking cumulative steps.
    """

    def __init__(self, name: str, config: dict):
        self.name        = name
        self.step_pin    = config["step_pin"]
        self.dir_pin     = config["dir_pin"]
        self.en_pin      = config["en_pin"]
        self.gear_ratio  = config["gear_ratio"]

        # Calculate steps required for 1 degree of output shaft rotation
        self.steps_per_deg = (STEPS_PER_REV * MICROSTEPS * self.gear_ratio) / 360.0
        
        # Internal state tracking
        self.current_steps = 0  
        self.target_steps  = 0
        self._lock         = threading.Lock()

        # Configure GPIO
        GPIO.setup(self.step_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.dir_pin,  GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.en_pin,   GPIO.OUT, initial=GPIO.HIGH)  # HIGH = disabled

    def enable(self):
        GPIO.output(self.en_pin, GPIO.LOW)
        time.sleep(1e-3)

    def disable(self):
        GPIO.output(self.en_pin, GPIO.HIGH)

    def _step_pulse(self):
        GPIO.output(self.step_pin, GPIO.HIGH)
        time.sleep(STEP_PULSE_WIDTH)
        GPIO.output(self.step_pin, GPIO.LOW)

    def move_one_step(self, direction: int, step_delay: float = 500e-6):
        """Executes a single microstep in the specified direction."""
        dir_val = GPIO.HIGH if direction > 0 else GPIO.LOW
        GPIO.output(self.dir_pin, dir_val)
        time.sleep(5e-6) # Setup time for DIR pin

        self._step_pulse()
        time.sleep(max(step_delay - STEP_PULSE_WIDTH, STEP_PERIOD_MIN))
        
        with self._lock:
            self.current_steps += direction

    def set_target_deg(self, target_deg: float):
        """Set a new target position in degrees."""
        with self._lock:
            self.target_steps = int(round(target_deg * self.steps_per_deg))


# ──────────────────────────────────────────────
# MAIN CONTROLLER
# ──────────────────────────────────────────────

class RobotController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Initialize Motors
        self._motor_list = [
            StepperMotor("base", MOTORS["base"]),
            StepperMotor("shoulder", MOTORS["shoulder"]),
            StepperMotor("elbow", MOTORS["elbow"]),
        ]

        self.step_delay = 0.0005  # Speed control (lower is faster)
        self.running = True

        # Enable Motors
        for motor in self._motor_list:
            motor.enable()
        print("All motors enabled.")

        # Start Motion Background Thread
        self._motion_thread = threading.Thread(target=self._motion_loop, daemon=True)
        self._motion_thread.start()

    def _motion_loop(self):
        """Continuously steps motors towards their targets in the background."""
        while self.running:
            stepped_any = False
            for motor in self._motor_list:
                with motor._lock:
                    current = motor.current_steps
                    target = motor.target_steps

                delta = target - current
                if delta != 0:
                    direction = 1 if delta > 0 else -1
                    motor.move_one_step(direction, step_delay=self.step_delay)
                    stepped_any = True

            # If no motors need to move, sleep briefly to avoid 100% CPU usage
            if not stepped_any:
                time.sleep(0.01)

    def run_cli(self):
        """Allows user to type degrees directly into the terminal."""
        print("Ready. Enter degrees for [Base Shoulder Elbow] separated by spaces.")
        print("Type 'exit' or press Ctrl+C to stop.\n")
        
        while self.running:
            try:
                user_input = input("Target Degrees: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                parts = user_input.strip().split()
                if len(parts) == 3:
                    deg1, deg2, deg3 = map(float, parts)
                    self._motor_list[0].set_target_deg(deg1)
                    self._motor_list[1].set_target_deg(deg2)
                    self._motor_list[2].set_target_deg(deg3)
                    print(f"Moving to -> Base: {deg1}°, Shoulder: {deg2}°, Elbow: {deg3}°\n")
                else:
                    print("Invalid input. Please enter exactly 3 numbers (e.g., '90 45 -30').\n")
            except ValueError:
                print("Invalid input. Please enter numbers only.\n")
            except (EOFError, KeyboardInterrupt):
                break

    def shutdown(self):
        print("\nShutting down — disabling all motors.")
        self.running = False
        for motor in self._motor_list:
            motor.disable()
        GPIO.cleanup()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    controller = RobotController()
    try:
        controller.run_cli()
    finally:
        controller.shutdown()
        sys.exit(0)