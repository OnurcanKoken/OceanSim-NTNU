"""
BlueROV2 Heavy 6-DOF Hydrodynamic Dynamics Model

Ported from MarineGym's UnderwaterVehicle (PyTorch/GPU) to NumPy for single-environment use.
Parameters from Benzon et al. (experimentally validated).

Implements Fossen's equations of motion for underwater vehicles:
    M·ν̇ + C(ν)·ν + D(ν)·ν + g(η) = τ

where:
    M   = added mass matrix
    C   = Coriolis and centripetal matrix
    D   = damping matrix (linear + quadratic)
    g   = hydrostatic restoring forces (buoyancy)
    τ   = external forces (thrusters, control inputs)
"""

import numpy as np
import yaml


def quat_to_euler(quat):
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def quat_rotate_inverse(quat, vec):
    """Rotate a vector by the inverse of a quaternion [w, x, y, z]."""
    w, x, y, z = quat
    q_vec = np.array([x, y, z])
    a = vec * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, vec) * 2.0 * w
    c = q_vec * (np.dot(q_vec, vec) * 2.0)
    return a - b + c


class BlueROVDynamics:
    """
    Computes hydrodynamic forces and torques for a BlueROV2 Heavy.

    Call `compute_hydrodynamics(quat, lin_vel, ang_vel, dt)` each physics step.
    Returns (force, torque) in the **body frame** to be applied via PhysxForceAPI.
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        vehicle = cfg['vehicle']
        hydro = cfg['hydro_coefficients']
        current = cfg['ocean_current']

        self.mass = vehicle['mass']
        self.volume = vehicle['volume']
        self.water_density = vehicle['water_density']
        self.gravity = vehicle['gravity']

        # Center of gravity / buoyancy vectors
        self.cg = np.array(vehicle['cg'], dtype=np.float64)
        self.cb = np.array(vehicle['cb'], dtype=np.float64)
        # Buoyancy offset: vector from CG to CB (used for restoring moments)
        self.r_bg = self.cb - self.cg

        self.added_mass_diag = np.array(hydro['added_mass'], dtype=np.float64)
        self.linear_damping_diag = np.array(hydro['linear_damping'], dtype=np.float64)
        self.quadratic_damping_diag = np.array(hydro['quadratic_damping'], dtype=np.float64)

        # 6x6 diagonal matrices
        self.M_added = np.diag(self.added_mass_diag)
        self.D_linear = np.diag(self.linear_damping_diag)
        self.D_quadratic = np.diag(self.quadratic_damping_diag)

        # Ocean current
        self.max_flow_vel = np.array(current['max_velocity'], dtype=np.float64)
        self.flow_noise_scale = np.array(current['noise_scale'], dtype=np.float64)
        self.flow_vel = np.random.rand(6) * self.max_flow_vel

        # Acceleration estimation (low-pass filtered finite difference)
        self.alpha = cfg.get('acceleration_filter_alpha', 0.3)
        self._prev_body_vel = np.zeros(6)
        self._prev_body_acc = np.zeros(6)

    def reset(self):
        """Reset internal state (call on scenario reset)."""
        self._prev_body_vel = np.zeros(6)
        self._prev_body_acc = np.zeros(6)
        self.flow_vel = np.random.rand(6) * self.max_flow_vel

    def compute_hydrodynamics(self, quat_wxyz, lin_vel_world, ang_vel_world, dt):
        """
        Compute hydrodynamic forces and torques.

        Args:
            quat_wxyz: Orientation quaternion [w, x, y, z] in world frame.
            lin_vel_world: Linear velocity [vx, vy, vz] in world frame (m/s).
            ang_vel_world: Angular velocity [wx, wy, wz] in world frame (rad/s).
            dt: Physics timestep (s).

        Returns:
            (force_body, torque_body): NumPy arrays of shape (3,) each, in body frame.
        """
        # --- Transform velocities to body frame ---
        lin_vel_body = quat_rotate_inverse(quat_wxyz, lin_vel_world)
        ang_vel_body = quat_rotate_inverse(quat_wxyz, ang_vel_world)
        body_vel = np.concatenate([lin_vel_body, ang_vel_body])

        # --- Subtract ocean current (in body frame) ---
        flow_world = self.flow_vel + np.random.randn(6) * self.flow_noise_scale
        flow_body = np.concatenate([
            quat_rotate_inverse(quat_wxyz, flow_world[:3]),
            quat_rotate_inverse(quat_wxyz, flow_world[3:])
        ])
        relative_vel = body_vel - flow_body

        # Apply MarineGym's sign convention for sway/heave/pitch/yaw
        relative_vel[[1, 2, 4, 5]] *= -1

        # --- Estimate body-frame acceleration ---
        body_acc = self._estimate_acceleration(relative_vel, dt)

        # --- Euler angles for buoyancy ---
        rpy = quat_to_euler(quat_wxyz)
        rpy[[1, 2]] *= -1  # MarineGym sign convention

        # --- Compute hydrodynamic terms ---
        damping = self._compute_damping(relative_vel)
        added_mass = self._compute_added_mass(body_acc)
        coriolis = self._compute_coriolis(relative_vel)
        buoyancy = self._compute_buoyancy(rpy)

        # Total hydrodynamic wrench (6-DOF)
        hydro = -(added_mass + coriolis + damping)
        # Undo sign convention before output
        hydro[[1, 2, 4, 5]] *= -1
        buoyancy[[1, 2, 4, 5]] *= -1

        total = hydro + buoyancy
        force_body = total[:3]
        torque_body = total[3:]

        return force_body, torque_body

    def _estimate_acceleration(self, body_vel, dt):
        """Low-pass filtered finite-difference acceleration."""
        if dt <= 0:
            return self._prev_body_acc.copy()
        raw_acc = (body_vel - self._prev_body_vel) / dt
        filtered_acc = (1.0 - self.alpha) * self._prev_body_acc + self.alpha * raw_acc
        self._prev_body_vel = body_vel.copy()
        self._prev_body_acc = filtered_acc.copy()
        return filtered_acc

    def _compute_damping(self, body_vel):
        """Linear + quadratic velocity-dependent damping: D(v) * v"""
        abs_vel_diag = np.diag(np.abs(body_vel))
        # Cross-coupling between sway-yaw and heave-pitch
        abs_vel_diag[1, 5] = abs(body_vel[5])
        abs_vel_diag[2, 4] = abs(body_vel[4])
        abs_vel_diag[4, 2] = abs(body_vel[2])
        abs_vel_diag[5, 1] = abs(body_vel[1])

        D = self.D_linear + self.D_quadratic @ abs_vel_diag
        return D @ body_vel

    def _compute_added_mass(self, body_acc):
        """Added mass force: M_a * a"""
        return self.M_added @ body_acc

    def _compute_coriolis(self, body_vel):
        """Coriolis force from added mass: C(v) * v"""
        Mv = self.M_added @ body_vel
        coriolis = np.zeros(6)
        coriolis[:3] = -np.cross(Mv[:3], body_vel[3:])
        coriolis[3:] = -(np.cross(Mv[:3], body_vel[:3]) + np.cross(Mv[3:], body_vel[3:]))
        return coriolis

    def _compute_buoyancy(self, rpy):
        """Hydrostatic restoring forces from buoyancy."""
        roll, pitch, _ = rpy
        B = self.water_density * self.gravity * self.volume  # buoyant force magnitude
        W = self.mass * self.gravity                         # weight

        # Distance from CG to CB along body z-axis
        d = abs(self.r_bg[2])

        buoyancy = np.zeros(6)
        buoyancy[0] = (B - W) * np.sin(pitch)                        # surge
        buoyancy[1] = -(B - W) * np.sin(roll) * np.cos(pitch)        # sway
        buoyancy[2] = -(B - W) * np.cos(roll) * np.cos(pitch)        # heave (net buoyancy)
        buoyancy[3] = -d * B * np.cos(pitch) * np.sin(roll)          # roll restoring moment
        buoyancy[4] = -d * B * np.sin(pitch)                         # pitch restoring moment
        return buoyancy
