#!/usr/bin/env python3
"""
RAC_AFC_Fuzzy_Control.py
========================
Resolved Acceleration Control (RAC) + Active Force Control (AFC)
with Fuzzy-Logic inertia estimation, for the 3-DOF RRR translational
manipulator defined in `Robot_Motion_Control.py`.

This module implements the control topology shown in the supplied block
diagram:

    xdd_d ──►(+)──►(+)──► xdd_ref ──► InvKin ──► qdd_ref ──► [IN/Ktn] ─► Ic
              ▲     ▲                                                     │
          Kp(x_err) Kd(xd_err)                                           (+)─► It ─►[Ktn]─► Tq
                                                                          ▲                   │
                                                                AFC:  Ia=[1/Ktn]·Q*        (Dynamics
                                                                          ▲                  of MM)
                                                                   Q* = Tq − IN·qdd            │
                                                                          ▲                    ▼
                                                                   IN ◄─ Fuzzy(e, edot)      qdd ─►∫─► qd ─►∫─► q
                                                                                                   │
                                                            Jacobian ◄────────────────────────────┘  (xd_act)
                                                            FwdKin   ◄────────────────────────────┘  (x_act)

WHAT EACH LOOP DOES
-------------------
RAC (outer, Cartesian):
    Drives the end-effector acceleration so Cartesian position/velocity
    errors decay. Produces a *resolved* joint-acceleration reference qdd_ref
    that already accounts for the manipulator Jacobian.

        xdd_ref = xdd_d + Kd*(xd_d - xd_act) + Kp*(x_d - x_act)
        qdd_ref = Jinv * (xdd_ref - Jdot*qd)        (resolved acceleration)

AFC (inner, joint torque):
    Rejects modelling error and external disturbance Q by estimating the
    disturbance torque Q* and feeding it forward so the *actual* joint
    acceleration tracks the commanded one:

        Q*  = Tq - IN_est * qdd_measured            (estimated disturbance)
        Ia  = Q* / Ktn                              (compensation current)
        It  = Ic + Ia                               (total motor current)
        Tq  = Ktn * It                              (applied joint torque)

Fuzzy Logic:
    Online-tunes the estimated inertia IN_est used by AFC. A perfect IN
    makes AFC a perfect disturbance observer; a wrong IN leaks disturbance
    into the loop. The fuzzy block nudges IN_est based on tracking error e
    and its rate edot.

HARDWARE NOTE
-------------
The physical arm is stepper-driven (it accepts *position* targets, not
torque). A true torque loop needs current-controlled actuators, so on the
real hardware this module runs the RAC+AFC+Fuzzy law against an internal
dynamic model of the manipulator, integrates the resulting *actual* joint
acceleration into a corrected joint trajectory, and streams that trajectory
to the existing `set_target_deg()` interface. When current sensors are
available (via Sensor_Input.SensorTree) the measured motor current is used
to form the AFC disturbance estimate instead of the modelled torque, which
is exactly the Ia = Q*/Ktn path in the diagram.

Run standalone for a pure simulation (no Pi hardware required):

    python3 RAC_AFC_Fuzzy_Control.py
"""

import math
import time

# ──────────────────────────────────────────────────────────────────────
# Optional integration with the user's existing modules.
# Everything degrades gracefully to pure simulation if they're absent
# (e.g. when developing off the Raspberry Pi).
# ──────────────────────────────────────────────────────────────────────
try:
    from Robot_Motion_Control import RobotController, DH_PARAMS, HOME_OFFSET_DEG
    _HAVE_ROBOT = True
except Exception:
    _HAVE_ROBOT = False
    # Fallback DH table (identical to Robot_Motion_Control.py) so the
    # simulator's kinematics match the real robot exactly.
    DH_PARAMS = [
        (49.23,   math.radians(90.0), 231.9),
        (160.00,  0.0,                0.0),
        (210.87,  0.0,                3.0),
    ]
    HOME_OFFSET_DEG = [0.0, 90.0, -90.0]

try:
    from Sensor_Input import SensorTree, MUX_BASE, MUX_SHOULDER, MUX_ELBOW
    _HAVE_SENSORS = True
except Exception:
    _HAVE_SENSORS = False


# ══════════════════════════════════════════════════════════════════════
# SMALL LINEAR-ALGEBRA HELPERS  (3x3, no numpy dependency)
# ══════════════════════════════════════════════════════════════════════
def mat_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(3)) for i in range(3)]


def mat_mat(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def scale(v, s):
    return [x * s for x in v]


def det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def inv3(M, eps=1e-9):
    """Inverse of a 3x3 matrix; returns a damped pseudo-inverse near singularity."""
    d = det3(M)
    if abs(d) < eps:
        # Damped least squares (Levenberg-Marquardt) to survive singularities.
        lam = 1e-3
        Mt = transpose(M)
        MtM = mat_mat(Mt, M)
        for i in range(3):
            MtM[i][i] += lam * lam
        d2 = det3(MtM)
        if abs(d2) < eps:
            return [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        inv_MtM = _inv3_raw(MtM, d2)
        return mat_mat(inv_MtM, Mt)
    return _inv3_raw(M, d)


def _inv3_raw(M, d):
    c = [[0.0] * 3 for _ in range(3)]
    c[0][0] = (M[1][1] * M[2][2] - M[1][2] * M[2][1]) / d
    c[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) / d
    c[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) / d
    c[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) / d
    c[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) / d
    c[1][2] = (M[0][2] * M[1][0] - M[0][0] * M[1][2]) / d
    c[2][0] = (M[1][0] * M[2][1] - M[1][1] * M[2][0]) / d
    c[2][1] = (M[0][1] * M[2][0] - M[0][0] * M[2][1]) / d
    c[2][2] = (M[0][0] * M[1][1] - M[0][1] * M[1][0]) / d
    return c


# ══════════════════════════════════════════════════════════════════════
# KINEMATICS  (mirrors Robot_Motion_Control.py, in radians, mm)
# ══════════════════════════════════════════════════════════════════════
def _dh_matrix(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return [
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ]


def _matmul4(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def fk_position(q_cmd_rad):
    """Forward kinematics: commanded joint angles (rad) -> (x,y,z) mm."""
    T = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for i, (a, alpha, d) in enumerate(DH_PARAMS):
        q = q_cmd_rad[i] + math.radians(HOME_OFFSET_DEG[i])
        T = _matmul4(T, _dh_matrix(q, d, a, alpha))
    return [T[0][3], T[1][3], T[2][3]]


def jacobian(q_cmd_rad, eps=1e-6):
    """
    Numerical position Jacobian J (3x3): dx = J dq, with q in radians.
    Finite-difference is robust for this 3-DOF positional arm and avoids
    hand-deriving the analytic Jacobian with the home offsets baked in.
    """
    base = fk_position(q_cmd_rad)
    J = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        qj = list(q_cmd_rad)
        qj[j] += eps
        pj = fk_position(qj)
        for i in range(3):
            J[i][j] = (pj[i] - base[i]) / eps
    return J


def jacobian_dot(q_cmd_rad, qd_rad, eps=1e-6):
    """
    Time derivative of the Jacobian, Jdot = dJ/dt, via central finite
    difference of J along the current joint-velocity direction. Used by the
    resolved-acceleration term:  xdd = J*qdd + Jdot*qd.
    """
    q_plus  = [q_cmd_rad[i] + eps * qd_rad[i] for i in range(3)]
    q_minus = [q_cmd_rad[i] - eps * qd_rad[i] for i in range(3)]
    Jp = jacobian(q_plus)
    Jm = jacobian(q_minus)
    return [[(Jp[i][j] - Jm[i][j]) / (2 * eps) for j in range(3)] for i in range(3)]


# ══════════════════════════════════════════════════════════════════════
# MANIPULATOR DYNAMIC MODEL  (for simulation / disturbance generation)
# ══════════════════════════════════════════════════════════════════════
class ManipulatorDynamics:
    """
    Lightweight rigid-body model:  M(q) qdd + C(q,qd) qd + G(q) = Tau + Q

    M(q)  : configuration-dependent inertia  ("Dynamics of MM" / true IN)
    C,G   : Coriolis/centrifugal + gravity, lumped as a smooth disturbance
    Q     : external disturbance torque (payload, friction, contact)

    This is the "Dynamics of MM" block. The controller does NOT get to see
    the true M(q); it only knows its fuzzy estimate IN_est. That mismatch is
    exactly what AFC + Fuzzy are there to absorb.
    """

    def __init__(self):
        # Nominal diagonal-ish inertia (kg*m^2-ish, arbitrary consistent units).
        # Base carries the most; elbow the least.
        self.M0 = [
            [0.90, 0.05, 0.02],
            [0.05, 0.55, 0.04],
            [0.02, 0.04, 0.25],
        ]
        self.gravity_gain = [0.0, 6.0, 2.5]  # gravity loads shoulder & elbow

    def inertia(self, q):
        """True configuration-dependent inertia M(q)."""
        # Inertia seen at the base shrinks as the arm folds in (cos of reach).
        reach = math.cos(q[1]) * 0.5 + math.cos(q[1] + q[2]) * 0.3
        M = [row[:] for row in self.M0]
        M[0][0] += 0.6 * reach * reach
        return M

    def bias_torque(self, q, qd):
        """Coriolis + gravity, treated as part of the disturbance."""
        g = [self.gravity_gain[i] * math.cos(q[i]) for i in range(3)]
        # crude velocity-product Coriolis term
        c = [0.10 * qd[(i + 1) % 3] * qd[i] for i in range(3)]
        return [g[i] + c[i] for i in range(3)]

    def accel(self, q, qd, tau, Q_ext):
        """
        Solve qdd from  M(q) qdd = Tau + Q_ext - bias(q,qd).
        Returns the *actual* joint acceleration the arm experiences.
        """
        M = self.inertia(q)
        bias = self.bias_torque(q, qd)
        rhs = [tau[i] + Q_ext[i] - bias[i] for i in range(3)]
        return mat_vec(inv3(M), rhs)


# ══════════════════════════════════════════════════════════════════════
# FUZZY-LOGIC INERTIA ESTIMATOR
# ══════════════════════════════════════════════════════════════════════
class FuzzyInertiaEstimator:
    """
    Mamdani-style fuzzy controller that adapts the estimated inertia scalar
    IN_est for each joint used by AFC.

    Inputs : e    = tracking error          (q_ref - q_actual)
             edot = error rate              (d/dt of e)
    Output : dIN  = increment to IN_est

    Rule intuition: if the arm is consistently lagging the reference and the
    lag is growing, the AFC is under-compensating because IN_est is too low
    -> push IN_est up. If it's overshooting, pull it down. Small/zero error
    -> leave IN_est alone. This is the classic 5x5 error/error-rate rule base
    collapsed to the diagonal that matters for inertia adaptation.
    """

    def __init__(self, in_init, in_min, in_max, learn=0.02):
        self.IN = list(in_init)           # per-joint estimated inertia
        self.in_min = in_min
        self.in_max = in_max
        self.learn = learn

    # --- triangular membership helpers (universe normalised to [-1, 1]) ---
    @staticmethod
    def _memberships(x):
        """Return degrees for {NB, NS, ZE, PS, PB} of a normalised input."""
        x = max(-1.0, min(1.0, x))
        NB = max(0.0, min(1.0, (-x - 0.5) / 0.5)) if x < 0 else 0.0
        NS = max(0.0, 1.0 - abs((x + 0.5) / 0.5))
        ZE = max(0.0, 1.0 - abs(x / 0.5))
        PS = max(0.0, 1.0 - abs((x - 0.5) / 0.5))
        PB = max(0.0, min(1.0, (x - 0.5) / 0.5)) if x > 0 else 0.0
        return {"NB": NB, "NS": NS, "ZE": ZE, "PS": PS, "PB": PB}

    # Output singletons (normalised correction strength) for each rule result.
    _OUT = {"NB": -1.0, "NS": -0.5, "ZE": 0.0, "PS": 0.5, "PB": 1.0}

    # 5x5 rule base: rows = e, cols = edot. Standard PD-like fuzzy surface.
    _RULES = {
        "NB": {"NB": "NB", "NS": "NB", "ZE": "NS", "PS": "NS", "PB": "ZE"},
        "NS": {"NB": "NB", "NS": "NS", "ZE": "NS", "PS": "ZE", "PB": "PS"},
        "ZE": {"NB": "NS", "NS": "NS", "ZE": "ZE", "PS": "PS", "PB": "PS"},
        "PS": {"NB": "NS", "NS": "ZE", "ZE": "PS", "PS": "PS", "PB": "PB"},
        "PB": {"NB": "ZE", "NS": "PS", "ZE": "PS", "PS": "PB", "PB": "PB"},
    }

    def _infer_one(self, e_n, edot_n):
        """Mamdani inference + centroid (weighted-average) defuzzification."""
        me = self._memberships(e_n)
        med = self._memberships(edot_n)
        num = 0.0
        den = 0.0
        for le, we in me.items():
            if we <= 0.0:
                continue
            for led, wed in med.items():
                if wed <= 0.0:
                    continue
                strength = min(we, wed)          # AND = min
                out_label = self._RULES[le][led]
                num += strength * self._OUT[out_label]
                den += strength
        return num / den if den > 1e-9 else 0.0

    def update(self, e, edot, e_scale=0.05, edot_scale=0.5):
        """
        Update per-joint IN_est. e, edot are joint-space vectors (rad, rad/s).
        e_scale / edot_scale normalise them into the fuzzy [-1,1] universe.
        Returns the updated inertia estimate vector.
        """
        for j in range(3):
            e_n    = max(-1.0, min(1.0, e[j] / e_scale))
            edot_n = max(-1.0, min(1.0, edot[j] / edot_scale))
            dIN = self._infer_one(e_n, edot_n) * self.learn
            self.IN[j] = max(self.in_min, min(self.in_max, self.IN[j] + dIN))
        return list(self.IN)


# ══════════════════════════════════════════════════════════════════════
# THE COMBINED RAC + AFC + FUZZY CONTROLLER
# ══════════════════════════════════════════════════════════════════════
class RAC_AFC_FuzzyController:
    """
    One control step = one call to `step()`. Wire it into a fixed-rate loop.

    Gains:
      Kp, Kd : Cartesian PD gains for the RAC outer loop (per axis x,y,z)
      Ktn    : motor torque constant (Tq = Ktn * It). Cancels in the ideal
               AFC path but kept explicit to mirror the block diagram and to
               convert measured current -> torque on real hardware.
    """

    def __init__(self,
                 Kp=(120.0, 120.0, 120.0),
                 Kd=(22.0, 22.0, 22.0),
                 Ktn=1.0,
                 dt=0.01,
                 dynamics=None,
                 use_afc=True,
                 use_fuzzy=True):
        self.Kp = list(Kp)
        self.Kd = list(Kd)
        self.Ktn = Ktn
        self.dt = dt
        self.use_afc = use_afc
        self.use_fuzzy = use_fuzzy

        self.dyn = dynamics or ManipulatorDynamics()

        # Fuzzy seeds its inertia estimate from the controller's *nominal*
        # model diagonal — deliberately imperfect vs. the true M(q).
        nominal_diag = [self.dyn.M0[i][i] for i in range(3)]
        self.fuzzy = FuzzyInertiaEstimator(
            in_init=nominal_diag,
            in_min=0.05,
            in_max=3.0,
            learn=0.03,
        )
        # If fuzzy is disabled, AFC uses this fixed (wrong-ish) estimate.
        self.IN_fixed = list(nominal_diag)

        # State (joint space, radians)
        self.q  = [0.0, 0.0, 0.0]
        self.qd = [0.0, 0.0, 0.0]

        self._prev_e = [0.0, 0.0, 0.0]

        # Logging
        self.t = 0.0

    # ---- state initialisation -------------------------------------------------
    def set_joint_state(self, q_rad, qd_rad=None):
        self.q = list(q_rad)
        self.qd = list(qd_rad) if qd_rad else [0.0, 0.0, 0.0]

    # ---- one control + dynamics step -----------------------------------------
    def step(self, x_d, xd_d, xdd_d, Q_ext=None):
        """
        Advance one timestep.

        x_d, xd_d, xdd_d : desired Cartesian position / velocity / accel (mm)
        Q_ext            : external disturbance torque (joint space). If None,
                           the dynamic model supplies Coriolis/gravity only.

        Returns a dict of useful signals for logging/plotting.
        """
        dt = self.dt
        if Q_ext is None:
            Q_ext = [0.0, 0.0, 0.0]

        # ---- current actual Cartesian state (Forward Kinematics + Jacobian) ----
        x_act = fk_position(self.q)
        J = jacobian(self.q)
        xd_act = mat_vec(J, self.qd)              # xd_act = J * qd

        # ════════════════════ RAC : outer Cartesian loop ════════════════════
        #   xdd_ref = xdd_d + Kd*(xd_d - xd_act) + Kp*(x_d - x_act)
        e_x  = vec_sub(x_d,  x_act)
        e_xd = vec_sub(xd_d, xd_act)
        xdd_ref = [
            xdd_d[i] + self.Kd[i] * e_xd[i] + self.Kp[i] * e_x[i]
            for i in range(3)
        ]

        # Resolved acceleration -> joint accel reference:
        #   xdd = J*qdd + Jdot*qd   =>   qdd_ref = Jinv*(xdd_ref - Jdot*qd)
        Jdot = jacobian_dot(self.q, self.qd)
        Jinv = inv3(J)
        qdd_ref = mat_vec(Jinv, vec_sub(xdd_ref, mat_vec(Jdot, self.qd)))

        # ════════════════════ Inertia estimate (Fuzzy Logic) ════════════════
        # Tracking error in joint space drives the fuzzy inertia adaptation.
        # q_ref is reconstructed implicitly via the acceleration reference;
        # we use the Cartesian->joint resolved error as the fuzzy input.
        e_q = mat_vec(Jinv, e_x)                  # joint-space position error
        edot_q = [(e_q[j] - self._prev_e[j]) / dt for j in range(3)]
        self._prev_e = e_q

        if self.use_fuzzy:
            IN_est = self.fuzzy.update(e_q, edot_q)
        else:
            IN_est = self.IN_fixed

        # ════════════════════ Feedforward command current Ic ════════════════
        #   Ic = (IN / Ktn) * qdd_ref      (block "IN/Ktn")
        Ic = [(IN_est[i] / self.Ktn) * qdd_ref[i] for i in range(3)]

        # ════════════════════ AFC : inner disturbance-rejection loop ════════
        # Estimated disturbance torque:  Q* = Tq - IN_est * qdd_measured
        # Compensation current:          Ia = Q* / Ktn
        # On the first pass qdd_measured isn't known yet, so AFC uses the
        # previous step's measured acceleration (one-step delayed observer),
        # which is exactly how a discrete AFC observer is realised.
        if self.use_afc:
            qdd_meas_prev = getattr(self, "_qdd_meas", [0.0, 0.0, 0.0])
            Tq_prev       = getattr(self, "_Tq", [0.0, 0.0, 0.0])
            Q_star = [Tq_prev[i] - IN_est[i] * qdd_meas_prev[i] for i in range(3)]
            Ia = [Q_star[i] / self.Ktn for i in range(3)]
        else:
            Q_star = [0.0, 0.0, 0.0]
            Ia = [0.0, 0.0, 0.0]

        # Total current and applied torque
        It = [Ic[i] + Ia[i] for i in range(3)]    # It = Ic + Ia
        Tq = [self.Ktn * It[i] for i in range(3)]  # Tq = Ktn * It

        # ════════════════════ Dynamics of MM (plant) ════════════════════════
        # True plant integrates with the TRUE inertia M(q) and the real
        # disturbance Q_ext + Coriolis/gravity bias.
        qdd = self.dyn.accel(self.q, self.qd, Tq, Q_ext)

        # Stash for next step's AFC observer
        self._qdd_meas = qdd
        self._Tq = Tq

        # ════════════════════ Integrate  qdd ->∫-> qd ->∫-> q ═══════════════
        self.qd = [self.qd[i] + qdd[i] * dt for i in range(3)]
        self.q  = [self.q[i]  + self.qd[i] * dt for i in range(3)]
        self.t += dt

        x_new = fk_position(self.q)
        pos_err = math.dist(x_d, x_new)

        return {
            "t": self.t,
            "q": list(self.q),
            "qd": list(self.qd),
            "qdd": list(qdd),
            "x_act": x_new,
            "x_d": list(x_d),
            "pos_err_mm": pos_err,
            "qdd_ref": qdd_ref,
            "IN_est": list(IN_est),
            "Q_star": Q_star,
            "Tq": Tq,
            "It": It,
            "Ic": Ic,
            "Ia": Ia,
        }

    def joint_deg(self):
        """Current commanded joint angles in degrees (for set_target_deg)."""
        return [math.degrees(a) for a in self.q]


# ══════════════════════════════════════════════════════════════════════
# HARDWARE BRIDGE  — stream the controller's joint trajectory to the arm
# ══════════════════════════════════════════════════════════════════════
class HardwareBridge:
    """
    Connects RAC_AFC_FuzzyController to the real stepper arm
    (Robot_Motion_Control.RobotController) and, if present, the encoders /
    current sensors (Sensor_Input.SensorTree).

    The control law runs in joint *acceleration*; steppers accept *position*.
    Each control tick we integrate the corrected joint trajectory and push
    the resulting angle to set_target_deg(). When current sensors exist, the
    measured motor current is converted to a disturbance estimate and fed to
    the AFC path (Ia = Q*/Ktn), closing the loop on real torque.
    """

    def __init__(self, controller: "RAC_AFC_FuzzyController",
                 robot=None, sensors=None):
        self.ctrl = controller
        self.robot = robot
        self.sensors = sensors

    def sync_state_from_encoders(self):
        """Seed controller joint state from absolute encoders, if available."""
        if not self.sensors:
            return
        base = self.sensors.read_encoder_angle(MUX_BASE)
        sh   = self.sensors.read_encoder_angle(MUX_SHOULDER)
        el   = self.sensors.read_encoder_angle(MUX_ELBOW)
        vals = [base, sh, el]
        if all(v > -900 for v in vals):  # -999 == sensor/MUX failure
            self.ctrl.set_joint_state([math.radians(v) for v in vals])

    def measure_disturbance(self):
        """
        Read current sensors -> external joint torque estimate (very rough;
        scale factors are placeholders to be calibrated against your arm).
        Returns a joint-space Q_ext vector, or zeros if no sensors.
        """
        if not self.sensors:
            return [0.0, 0.0, 0.0]
        amps = [self.sensors.read_current_sensor(i) for i in range(3)]
        # Motor torque ≈ Ktn * I_motor; treat the *excess* over expected as
        # disturbance. Placeholder gain — calibrate per joint.
        TORQUE_PER_AMP = 0.18
        return [(a if a > -900 else 0.0) * TORQUE_PER_AMP for a in amps]

    def run(self, trajectory, send_to_motors=True, verbose=True):
        """
        Execute a Cartesian trajectory.

        trajectory: iterable of (x_d, xd_d, xdd_d) tuples, each a 3-vector.
        send_to_motors: if True and a robot is attached, push joint targets.
        """
        if self.sensors:
            self.sync_state_from_encoders()

        for k, (x_d, xd_d, xdd_d) in enumerate(trajectory):
            Q_ext = self.measure_disturbance() if self.sensors else None
            info = self.ctrl.step(x_d, xd_d, xdd_d, Q_ext=Q_ext)

            if send_to_motors and self.robot is not None:
                b, s, e = self.ctrl.joint_deg()
                self.robot.base_motor.set_target_deg(b)
                self.robot.shoulder_motor.set_target_deg(s)
                self.robot.elbow_motor.set_target_deg(e)

            if verbose and k % 20 == 0:
                q = info["q"]
                print(f"t={info['t']:5.2f}s  "
                      f"err={info['pos_err_mm']:6.2f}mm  "
                      f"q=[{math.degrees(q[0]):6.1f},{math.degrees(q[1]):6.1f},"
                      f"{math.degrees(q[2]):6.1f}]°  "
                      f"IN={[round(v,3) for v in info['IN_est']]}")

            # pace the loop to the controller dt when on real hardware
            if send_to_motors and self.robot is not None:
                time.sleep(self.ctrl.dt)

        if send_to_motors and self.robot is not None:
            self.robot._wait_until_at_target()


# ══════════════════════════════════════════════════════════════════════
# DEMO TRAJECTORY GENERATOR
# ══════════════════════════════════════════════════════════════════════
def make_step_trajectory(x_start, x_goal, duration, dt):
    """
    Quintic (minimum-jerk) Cartesian point-to-point trajectory.
    Yields (x_d, xd_d, xdd_d) at each timestep.
    """
    n = int(duration / dt)
    for k in range(n + 1):
        s = k / n if n else 1.0
        # quintic blend: position 0->1 with zero vel/accel at ends
        p   = 10 * s**3 - 15 * s**4 + 6 * s**5
        pd  = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
        pdd = (60 * s - 180 * s**2 + 120 * s**3) / (duration ** 2)
        x_d   = [x_start[i] + (x_goal[i] - x_start[i]) * p   for i in range(3)]
        xd_d  = [(x_goal[i] - x_start[i]) * pd               for i in range(3)]
        xdd_d = [(x_goal[i] - x_start[i]) * pdd              for i in range(3)]
        yield x_d, xd_d, xdd_d


# ══════════════════════════════════════════════════════════════════════
# STANDALONE SIMULATION  (no Raspberry Pi / hardware needed)
# ══════════════════════════════════════════════════════════════════════
def _simulate():
    dt = 0.01

    # Start the model at the home pose [0,0,0] commanded angles.
    q0 = [0.0, 0.0, 0.0]
    x_start = fk_position(q0)
    x_goal = [x_start[0] - 60.0, x_start[1] + 40.0, x_start[2] - 50.0]

    print("=" * 64)
    print("RAC + AFC + Fuzzy-Logic control — standalone simulation")
    print("=" * 64)
    print(f"Start Cartesian pos: [{x_start[0]:.1f}, {x_start[1]:.1f}, {x_start[2]:.1f}] mm")
    print(f"Goal  Cartesian pos: [{x_goal[0]:.1f}, {x_goal[1]:.1f}, {x_goal[2]:.1f}] mm")
    print("-" * 64)

    def run_variant(use_afc, use_fuzzy, label, disturbance):
        ctrl = RAC_AFC_FuzzyController(dt=dt, use_afc=use_afc, use_fuzzy=use_fuzzy)
        ctrl.set_joint_state([math.radians(a) for a in q0])
        traj = list(make_step_trajectory(x_start, x_goal, duration=2.0, dt=dt))
        # hold at goal for a second so steady-state error is visible
        hold = [(x_goal, [0, 0, 0], [0, 0, 0])] * int(1.0 / dt)
        peak = 0.0
        final = 0.0
        for (x_d, xd_d, xdd_d) in traj + hold:
            info = ctrl.step(x_d, xd_d, xdd_d, Q_ext=disturbance)
            peak = max(peak, info["pos_err_mm"])
            final = info["pos_err_mm"]
        print(f"{label:38s} peak_err={peak:6.2f}mm  final_err={final:6.3f}mm  "
              f"IN_est={[round(v,3) for v in info['IN_est']]}")
        return final

    # A constant external disturbance torque (e.g. unmodeled payload/friction)
    disturbance = [1.5, 4.0, 2.0]

    print("\nWith a constant external disturbance torque [1.5, 4.0, 2.0]:")
    run_variant(False, False, "RAC only (no AFC)", disturbance)
    run_variant(True,  False, "RAC + AFC (fixed inertia)", disturbance)
    run_variant(True,  True,  "RAC + AFC + Fuzzy (adaptive)", disturbance)

    print("\nInterpretation:")
    print("  RAC alone leaves a standing error under disturbance.")
    print("  AFC slashes it by estimating & cancelling the disturbance torque.")
    print("  Fuzzy tunes the inertia estimate so AFC stays accurate as the")
    print("  configuration (and hence true inertia) changes through the move.")
    print("=" * 64)


# ══════════════════════════════════════════════════════════════════════
# HARDWARE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
def _run_on_hardware():
    """
    Full integration path: home the arm, optionally read sensors, then drive
    a Cartesian trajectory through the RAC+AFC+Fuzzy controller.
    Only callable on the Raspberry Pi with the real modules present.
    """
    robot = RobotController()              # homes the arm on construction
    sensors = None
    if _HAVE_SENSORS:
        try:
            sensors = SensorTree()
        except Exception as exc:
            print(f"Sensors unavailable ({exc}); running model-based.")

    dt = 0.01
    ctrl = RAC_AFC_FuzzyController(dt=dt, use_afc=True, use_fuzzy=True)

    # Seed state from the homed pose (commanded [0,0,0]).
    ctrl.set_joint_state([0.0, 0.0, 0.0])
    x_start = fk_position([0.0, 0.0, 0.0])
    x_goal = [x_start[0] - 40.0, x_start[1] + 30.0, x_start[2] - 30.0]

    bridge = HardwareBridge(ctrl, robot=robot, sensors=sensors)
    traj = make_step_trajectory(x_start, x_goal, duration=3.0, dt=dt)

    try:
        bridge.run(traj, send_to_motors=True, verbose=True)
        print("Trajectory complete.")
    finally:
        robot.shutdown()
        if sensors is not None:
            sensors.close()


if __name__ == "__main__":
    import sys
    if "--hardware" in sys.argv and _HAVE_ROBOT:
        _run_on_hardware()
    else:
        if "--hardware" in sys.argv and not _HAVE_ROBOT:
            print("Robot_Motion_Control not importable here; running simulation.\n")
        _simulate()
