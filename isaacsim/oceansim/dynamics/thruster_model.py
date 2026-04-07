"""
BlueROV2 Heavy Thruster Allocation & T200 Thruster Model

Implements the full control pipeline:
    desired wrench τ (6-DOF) → allocation T⁺ → per-thruster commands V ∈ [-1,1]
    → Benzon polynomial → thrust forces f (N) → net wrench τ = T·f

Thruster geometry from Wu & Eng (Table 4.2, §4.4).
Thrust curve from Benzon et al. (9th-order polynomial, §4.1 eq. 18).
"""

import numpy as np
import yaml


def _benzon_polynomial(V, coeffs):
    """
    Benzon et al. 9th-order polynomial for T200 thrust.
    F(V) = c9·V⁹ + c7·V⁷ + c5·V⁵ + c3·V³ + c1·V

    Args:
        V: Normalized thruster input in [-1, 1], shape (n,).
        coeffs: [c9, c7, c5, c3, c1].

    Returns:
        Thrust force in Newtons, shape (n,).
    """
    c9, c7, c5, c3, c1 = coeffs
    V2 = V * V
    return (((c9 * V2 + c7) * V2 + c5) * V2 + c3) * V2 * V + c1 * V


class ThrusterAllocator:
    """
    Converts a desired 6-DOF wrench into 8 per-thruster commands,
    then computes individual thrust forces via the Benzon polynomial.

    Usage:
        allocator = ThrusterAllocator(config_path)
        thruster_forces, net_wrench = allocator.wrench_to_thrust(desired_wrench)
    """

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        tcfg = cfg['thruster']
        self.num_thrusters = tcfg['num_thrusters']
        self.max_thrust = tcfg['max_thrust_per_thruster']
        self.benzon_coeffs = tcfg['benzon_polynomial']

        # Build thrust configuration matrix T (6×8) from config
        self.T = np.array(tcfg['thrust_config_matrix'], dtype=np.float64)
        assert self.T.shape == (6, self.num_thrusters), \
            f"T matrix shape mismatch: expected (6, {self.num_thrusters}), got {self.T.shape}"

        # Precompute pseudoinverse: T⁺ = Tᵀ(TTᵀ)⁻¹
        self.T_pinv = np.linalg.pinv(self.T)  # shape (8, 6)

        # Linear approximation for inverse mapping (command from force)
        self._K_linear = 40.0  # Wu & Eng linear approximation

    def reset(self):
        """Reset internal state (stateless allocator, included for API consistency)."""
        pass

    def wrench_to_thrust(self, desired_wrench):
        """
        Convert a desired 6-DOF wrench to per-thruster forces.

        Args:
            desired_wrench: [Fx, Fy, Fz, Mx, My, Mz] in Newtons/Newton-meters.

        Returns:
            thruster_forces: Per-thruster forces in Newtons, shape (8,).
            net_wrench: Actual 6-DOF wrench produced (may differ due to saturation), shape (6,).
        """
        tau = np.asarray(desired_wrench, dtype=np.float64)

        # Step 1: Allocate desired wrench to per-thruster forces via pseudoinverse
        f_desired = self.T_pinv @ tau

        # Step 2: Convert desired forces to normalized commands V ∈ [-1, 1]
        V_commands = f_desired / self._K_linear

        # Step 3: Saturate commands to [-1, 1]
        V_commands = np.clip(V_commands, -1.0, 1.0)

        # Step 4: Compute actual thrust via Benzon polynomial
        thruster_forces = _benzon_polynomial(V_commands, self.benzon_coeffs)

        # Step 5: Compute actual net wrench: τ_actual = T · f_actual
        net_wrench = self.T @ thruster_forces

        return thruster_forces, net_wrench

    def get_thruster_commands(self, desired_wrench):
        """
        Get the normalized thruster commands V ∈ [-1, 1] for a desired wrench.
        Useful for logging/debugging.

        Args:
            desired_wrench: [Fx, Fy, Fz, Mx, My, Mz].

        Returns:
            commands: Normalized commands per thruster, shape (8,).
        """
        tau = np.asarray(desired_wrench, dtype=np.float64)
        f_desired = self.T_pinv @ tau
        V_commands = f_desired / self._K_linear
        return np.clip(V_commands, -1.0, 1.0)
