#!/usr/bin/env python3
"""
stepper_motor_controller.py
============================
Standalone Python Script: 3-DOF RRR Manipulator with Safe Homing Routine
"""

import time
import threading
import sys
import math

# Wrapper to safely catch initialization errors before RPi.GPIO loads
try:
    import RPi.GPIO as GPIO
except ImportError:
    print("CRITICAL ERROR: RPi.GPIO library not found. Please install it.")
    sys.exit(1)

# ──────────────────────────────────────────────
# HARDWARE CONSTANTS
# ──────────────────────────────────────────────

MICROSTEPS = 16
STEPS_PER_REV = 200  

MOTORS = {
    "base": {
        "step_pin": 27,
        "dir_pin":  17,
        "en_pin":   22,
        "gear_ratio": 27,
        "limit_pin": None,
    },
    "shoulder": {
        "step_pin": 24,
        "dir_pin":  25,
        "en_pin":   23,
        "gear_ratio": 19,
        "limit_pin": 26,
    },
    "elbow": {
        "step_pin": 6,
        "dir_pin":  13,
        "en_pin":   5,
        "gear_ratio": 14,
        "limit_pin": 16,
    },
}

STEP_PULSE_WIDTH = 2e-6   
STEP_PERIOD_MIN  = 50e-6  

# Homing parameters
HOMING_SEARCH_DELAY = 0.0025   # slow search speed (larger delay = slower)
HOMING_BACKOFF_DELAY = 0.0015  # slow reverse speed
HOMING_SEARCH_TIMEOUT = 10.0    # seconds

# Per-joint backoff distance (degrees) -> zero position
HOMING_BACKOFF_DEG_SHOULDER = 90
HOMING_BACKOFF_DEG_ELBOW    = 90

# ──────────────────────────────────────────────
# DENAVIT–HARTENBERG PARAMETERS  (Table 4.2)
#   row i: (a_i [mm], alpha_i [rad], d_i [mm])  -- theta_i is the joint variable
# ──────────────────────────────────────────────
DH_PARAMS = [
    # a_i,    alpha_i,            d_i
    (49.23,   math.radians(90.0), 231.9),   # Joint 1 (Base):     L1 = 231.9
    (160.00,  0.0,                0.0),      # Joint 2 (Shoulder): L2 = 160.00
    (210.87,  0.0,                3.0),      # Joint 3 (Elbow):    L3 = 210.87, L4 = 3
]

# Offset between COMMANDED angle (0 = home pose) and DH joint angle.
#   q_DH = commanded_angle + HOME_OFFSET
# Home pose: shoulder vertical-up (+90 from DH-zero), elbow folded flat (-90).
# Flip a sign here if a joint's physical rotation runs opposite to its frame +Z.
HOME_OFFSET_DEG = [
    0.0,    # Base
    90.0,   # Shoulder
    -90.0,  # Elbow
]


# ──────────────────────────────────────────────
# STEPPER MOTOR DRIVER
# ──────────────────────────────────────────────

class StepperMotor:
    def __init__(self, name: str, config: dict):
        self.name         = name
        self.step_pin    = config["step_pin"]
        self.dir_pin     = config["dir_pin"]
        self.en_pin      = config["en_pin"]
        self.gear_ratio  = config["gear_ratio"]
        self.limit_pin   = config["limit_pin"]

        self.steps_per_deg = (STEPS_PER_REV * MICROSTEPS * self.gear_ratio) / 360.0
        
        self.current_steps = 0  
        self.target_steps  = 0
        self._lock         = threading.Lock()

        GPIO.setup(self.step_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.dir_pin,  GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.en_pin,   GPIO.OUT, initial=GPIO.HIGH)  

        if self.limit_pin is not None:
            GPIO.setup(self.limit_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def is_limit_pressed(self) -> bool:
        if self.limit_pin is None:
            return False
        return GPIO.input(self.limit_pin) == GPIO.LOW

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
        dir_val = GPIO.HIGH if direction > 0 else GPIO.LOW
        GPIO.output(self.dir_pin, dir_val)
        time.sleep(5e-6) 

        self._step_pulse()
        time.sleep(max(step_delay - STEP_PULSE_WIDTH, STEP_PERIOD_MIN))
        
        with self._lock:
            self.current_steps += direction

    def set_target_deg(self, target_deg: float):
        with self._lock:
            self.target_steps = int(round(target_deg * self.steps_per_deg))

    def reset_position(self):
        with self._lock:
            self.current_steps = 0
            self.target_steps = 0


# ──────────────────────────────────────────────
# MAIN CONTROLLER
# ──────────────────────────────────────────────

class RobotController:
    def __init__(self):
        print("Initializing GPIO system...")
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        print("Setting up motor drivers...")
        self.base_motor     = StepperMotor("base", MOTORS["base"])
        self.shoulder_motor = StepperMotor("shoulder", MOTORS["shoulder"])
        self.elbow_motor    = StepperMotor("elbow", MOTORS["elbow"])

        self._motor_list = [self.base_motor, self.shoulder_motor, self.elbow_motor]
        self.step_delay = 0.0005  
        self.running = True

        for motor in self._motor_list:
            motor.enable()
        print("All motors enabled successfully.")

        # Print current live state of switch pins to terminal
        print(f"DEBUG - Shoulder Limit (GPIO 26): {'PRESSED (LOW)' if self.shoulder_motor.is_limit_pressed() else 'OPEN (HIGH)'}")
        print(f"DEBUG - Elbow Limit (GPIO 16): {'PRESSED (LOW)' if self.elbow_motor.is_limit_pressed() else 'OPEN (HIGH)'}")

        self.home_robot()

        self._motion_thread = threading.Thread(target=self._motion_loop, daemon=True)
        self._motion_thread.start()

    def home_robot(self):
        print("\n=== Starting Homing Sequence ===")

        # Home shoulder and elbow simultaneously using their limit switches.
        # Each entry: (motor, home_direction, backoff_degrees)
        self._home_joints_with_switch(
            [
                (self.elbow_motor, 1, HOMING_BACKOFF_DEG_ELBOW),
                (self.shoulder_motor, -1, HOMING_BACKOFF_DEG_SHOULDER),
            ]
        )

        print("Homing Base: Forced soft reset to 0° position.")
        self.base_motor.reset_position()
        print("=== Homing Sequence Complete! ===\n")

    def _home_joints_with_switch(self, joints):
        """
        Home multiple switched joints at once.

        joints: list of (motor, home_direction, backoff_deg) tuples.

        Two-phase homing:
          Phase A: Slow search toward the switch until pressed (timeout-bounded).
          Phase B: Slowly reverse off the switch by that joint's backoff_deg
                   degrees, and zero the position there.

        All joints are stepped together each loop iteration, so they search
        for their limit switches simultaneously.
        """
        names = ", ".join(m.name.upper() for m, _, _ in joints)
        print(f"Homing {names} simultaneously...")

        # ── Phase A: Slow Search (timeout-bounded) ──
        print("  Phase A: Slow searching for limit switches...")
        found = {m.name: False for m, _, _ in joints}
        start_time = time.time()
        while not all(found.values()):
            if time.time() - start_time > HOMING_SEARCH_TIMEOUT:
                for m, _, _ in joints:
                    if not found[m.name]:
                        print(f"  [TIMEOUT] {m.name.upper()} failed to find switch within {HOMING_SEARCH_TIMEOUT}s.")
                break
            for motor, home_dir, _ in joints:
                if found[motor.name]:
                    continue
                if motor.is_limit_pressed():
                    found[motor.name] = True
                    print(f"  [{motor.name.upper()}] Limit tripped.")
                else:
                    motor.move_one_step(home_dir, step_delay=HOMING_SEARCH_DELAY)
        time.sleep(0.1)

        # ── Phase B: Slow Backoff by per-joint backoff_deg -> zero position ──
        print("  Phase B: Backing off to set zero position...")
        backoff = []
        for motor, home_dir, backoff_deg in joints:
            if not found[motor.name]:
                continue  # skip joints that never found their switch
            steps = int(round(backoff_deg * motor.steps_per_deg))
            print(f"  [{motor.name.upper()}] Backing off {backoff_deg}°.")
            backoff.append([motor, -home_dir, steps])  # reverse direction

        done = False
        while not done:
            done = True
            for entry in backoff:
                motor, direction, remaining = entry
                if remaining > 0:
                    motor.move_one_step(direction, step_delay=HOMING_BACKOFF_DELAY)
                    entry[2] = remaining - 1
                    if entry[2] > 0:
                        done = False
        time.sleep(0.1)

        for motor, _, _ in joints:
            if found[motor.name]:
                motor.reset_position()
                print(f"  [{motor.name.upper()}] Successfully homed and zeroed.")

    def _motion_loop(self):
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

            if not stepped_any:
                time.sleep(0.01)

    @staticmethod
    def _dh_matrix(theta, d, a, alpha):
        """Standard DH homogeneous transform for one link."""
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return [
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,      sa,       ca,      d],
            [0.0,     0.0,      0.0,    1.0],
        ]

    @staticmethod
    def _matmul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

    @staticmethod
    def _rotation_to_rpy(R):
        """
        Extract roll-pitch-yaw (ZYX Euler) angles, in degrees, from a 3x3
        rotation matrix R.

        Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
          yaw   -> rotation about Z
          pitch -> rotation about Y
          roll  -> rotation about X

        Handles the gimbal-lock case (pitch = +/-90 deg) by setting roll = 0
        and folding the remaining DOF into yaw.
        """
        r00, r01, r02 = R[0]
        r10, r11, r12 = R[1]
        r20, r21, r22 = R[2]

        sy = math.hypot(r00, r10)  # = cos(pitch)
        if sy > 1e-9:
            yaw   = math.atan2(r10, r00)
            pitch = math.atan2(-r20, sy)
            roll  = math.atan2(r21, r22)
        else:
            # Gimbal lock: pitch ~ +/-90 deg
            yaw   = math.atan2(-r01, r11)
            pitch = math.atan2(-r20, sy)
            roll  = 0.0

        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    def forward_kinematics(self, commanded_deg):
        """
        Compute the end-effector pose from COMMANDED joint angles (degrees),
        where 0,0,0 is the physical home pose.

        Converts commanded -> DH angles via HOME_OFFSET_DEG, then chains the
        DH transforms from Table 4.2.

        Returns (position, orientation, T):
          position    = (x, y, z) in mm
          orientation = (roll, pitch, yaw) in degrees (ZYX Euler)
          T           = the full 4x4 homogeneous transform
        """
        T = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]  # identity
        for i, (cmd_deg, (a, alpha, d)) in enumerate(zip(commanded_deg, DH_PARAMS)):
            q = math.radians(cmd_deg + HOME_OFFSET_DEG[i])
            T = self._matmul(T, self._dh_matrix(q, d, a, alpha))

        position = (T[0][3], T[1][3], T[2][3])
        R = [row[:3] for row in T[:3]]          # 3x3 rotation part
        orientation = self._rotation_to_rpy(R)  # (roll, pitch, yaw) in degrees
        return position, orientation, T

    def run_cli(self):
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
                    print(f"Moving to -> Base: {deg1}°, Shoulder: {deg2}°, Elbow: {deg3}°")
                    (x, y, z), (roll, pitch, yaw), _ = self.forward_kinematics([deg1, deg2, deg3])
                    print(f"  End-effector position:    X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm")
                    print(f"  End-effector orientation: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}° (ZYX)\n")
                else:
                    print("Invalid input. Please enter exactly 3 numbers.\n")
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
# ENTRY POINT WITH ERROR LOGGING
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    try:
        controller = RobotController()
        controller.run_cli()
    except Exception as e:
        print("\n!!! SCRIPT CRASHED DURING RUNTIME !!!")
        print("Error details:")
        traceback.print_exc()
    finally:
        try:
            controller.shutdown()
        except NameError:
            pass # Controller wasn't initialized completely
        sys.exit(0)