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
    """Rotate a vector by the inverse of a quaternion [w, x, y, z].
    Transforms from world frame to body frame."""
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
    Returns (force, torque) in the **body frame** (Isaac Sim Z-up convention)
    to be applied via PhysxForceAPI with worldFrameEnabled=False.
    """

    # Maximum physically plausible body-frame acceleration for an ROV (m/s² and rad/s²)
    MAX_LINEAR_ACC = 50.0
    MAX_ANGULAR_ACC = 100.0

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

        # Buoyant force magnitude
        self.B = self.water_density * self.gravity * self.volume

        # Center of gravity / buoyancy vectors
        self.cg = np.array(vehicle['cg'], dtype=np.float64)
        self.cb = np.array(vehicle['cb'], dtype=np.float64)
        # Vector from CG to CB (used for restoring moments)
        self.r_bg = self.cb - self.cg

        self.added_mass_diag = np.array(hydro['added_mass'], dtype=np.float64)
        self.linear_damping_diag = np.array(hydro['linear_damping'], dtype=np.float64)
        self.quadratic_damping_diag = np.array(hydro['quadratic_damping'], dtype=np.float64)
        # Optional off-diagonal damping couplings. These default to zero because
        # reusing the diagonal sway/heave terms here can create large lateral or
        # vertical forces during pure yaw/pitch maneuvers.
        coupling_cfg = hydro.get('quadratic_coupling', {})
        self._quadratic_coupling = {
            'sway_from_yaw': float(coupling_cfg.get('sway_from_yaw', 0.0)),
            'heave_from_pitch': float(coupling_cfg.get('heave_from_pitch', 0.0)),
            'pitch_from_heave': float(coupling_cfg.get('pitch_from_heave', 0.0)),
            'yaw_from_sway': float(coupling_cfg.get('yaw_from_sway', 0.0)),
        }

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
            (force_body, torque_body): NumPy arrays of shape (3,) each,
            in Isaac Sim body frame (Z-up).
        """
        # --- Transform velocities to body frame (Isaac Sim Z-up convention) ---
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

        # --- Convert Isaac Sim body frame (X-fwd, Y-left, Z-up) to
        #     Fossen/MarineGym body frame (X-fwd, Y-right, Z-down) ---
        # Negate sway, heave, pitch, yaw to match NED convention
        vel_ned = relative_vel.copy()
        vel_ned[[1, 2, 4, 5]] *= -1

        # --- Estimate body-frame acceleration (in NED convention) ---
        body_acc = self._estimate_acceleration(vel_ned, dt)

        # --- Compute hydrodynamic terms (all in NED convention) ---
        damping = self._compute_damping(vel_ned)
        added_mass = self._compute_added_mass(body_acc)
        coriolis = self._compute_coriolis(vel_ned)

        # Hydrodynamic wrench in NED: τ_hydro = -(M_a·a + C·v + D·v)
        hydro_ned = -(added_mass + coriolis + damping)

        # --- Convert back to Isaac Sim body frame (negate sway, heave, pitch, yaw) ---
        hydro_isaac = hydro_ned.copy()
        hydro_isaac[[1, 2, 4, 5]] *= -1

        # --- Buoyancy: computed directly in Isaac Sim body frame via quaternion ---
        buoy_force, buoy_torque = self._compute_buoyancy(quat_wxyz)

        # --- Combine ---
        force_body = hydro_isaac[:3] + buoy_force
        torque_body = hydro_isaac[3:] + buoy_torque

        return force_body, torque_body

    def _estimate_acceleration(self, body_vel, dt):
        """Low-pass filtered finite-difference acceleration with magnitude clamping."""
        if dt <= 0:
            return self._prev_body_acc.copy()
        raw_acc = (body_vel - self._prev_body_vel) / dt

        # Clamp to physically plausible range to prevent feedback instability
        raw_acc[:3] = np.clip(raw_acc[:3], -self.MAX_LINEAR_ACC, self.MAX_LINEAR_ACC)
        raw_acc[3:] = np.clip(raw_acc[3:], -self.MAX_ANGULAR_ACC, self.MAX_ANGULAR_ACC)

        filtered_acc = (1.0 - self.alpha) * self._prev_body_acc + self.alpha * raw_acc
        self._prev_body_vel = body_vel.copy()
        self._prev_body_acc = filtered_acc.copy()
        return filtered_acc

    def _compute_damping(self, body_vel):
        """Linear + quadratic velocity-dependent damping: D(v) * v"""
        # Default to diagonal damping. Off-diagonal coupling is only applied
        # when explicitly configured, which keeps pure yaw/pitch commands from
        # generating unintended sway/heave forces.
        damping = self.linear_damping_diag * body_vel
        damping += self.quadratic_damping_diag * np.abs(body_vel) * body_vel

        damping[1] += self._quadratic_coupling['sway_from_yaw'] * abs(body_vel[5]) * body_vel[5]
        damping[2] += self._quadratic_coupling['heave_from_pitch'] * abs(body_vel[4]) * body_vel[4]
        damping[4] += self._quadratic_coupling['pitch_from_heave'] * abs(body_vel[2]) * body_vel[2]
        damping[5] += self._quadratic_coupling['yaw_from_sway'] * abs(body_vel[1]) * body_vel[1]

        return damping

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

    def _compute_buoyancy(self, quat_wxyz):
        """
        Hydrostatic buoyancy force and restoring torque.

        Computed directly in Isaac Sim body frame using the quaternion,
        avoiding Euler angle singularities and NED sign convention issues.

        PhysX handles gravity (W = m*g) separately, so this only applies
        the buoyant force B = ρgV (always world-Z upward).

        Returns:
            (force_body, torque_body): buoyancy force and restoring torque
            in Isaac Sim body frame (Z-up).
        """
        # Buoyancy is always [0, 0, +B] in world frame (upward)
        buoyancy_world = np.array([0.0, 0.0, self.B])

        # Rotate to body frame
        buoy_force_body = quat_rotate_inverse(quat_wxyz, buoyancy_world)

        # Restoring torque: τ = r_bg × F_buoyancy (in body frame)
        # r_bg = vector from CG to CB; this creates a moment that
        # restores the vehicle to upright orientation
        buoy_torque_body = np.cross(self.r_bg, buoy_force_body)

        return buoy_force_body, buoy_torque_body
