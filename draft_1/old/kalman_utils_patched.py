import numpy as np

# ===================== EKF =====================

class EKF:
    """6D EKF state in world frame:

        x = [x, y, z, vx, vy, vz]^T

    Notes about this EKF:
      - It does *not* estimate attitude. In your current pipeline you use PoseSensor
        to rotate IMU/DVL to world. That means gyro-related Allan parameters do not
        influence this EKF unless you also estimate attitude.
      - IMU acceleration is treated as a known input with uncertainty modeled via
        process noise Q (continuous white acceleration model).

    You can optionally enable innovation gating (NIS / chi-square) in the update calls.
    """

    def __init__(
        self,
        dt: float,
        q_pos_std: float = 0.02,
        q_vel_std: float = 0.2,
        p_pos_std_init: float = 1.0,
        p_vel_std_init: float = 1.0,
    ):
        self.dt = float(dt)

        # State mean
        self.x = np.zeros(6, dtype=float)

        # State covariance
        self.P = np.diag(
            [
                p_pos_std_init**2,
                p_pos_std_init**2,
                p_pos_std_init**2,
                p_vel_std_init**2,
                p_vel_std_init**2,
                p_vel_std_init**2,
            ]
        ).astype(float)

        # Default process noise (legacy / simple). You can override with
        # set_process_noise_from_accel().
        self.Q = np.diag(
            [
                q_pos_std**2,
                q_pos_std**2,
                q_pos_std**2,
                q_vel_std**2,
                q_vel_std**2,
                q_vel_std**2,
            ]
        ).astype(float)

        # Keep the latest NIS values for debugging/plots
        self.last_nis = None

    # ---------- Process noise helpers ----------

    def set_process_noise_from_accel(self, sigma_a: float):
        """Set Q using a *continuous white-acceleration* model.

        If accel measurement has white noise std-dev sigma_a [m/s^2] per sample
        (or the equivalent continuous model), then the discretized covariance
        for [pos; vel] with constant-velocity dynamics is:

            Q = σ_a^2 * [[dt^4/4 I, dt^3/2 I],
                        [dt^3/2 I, dt^2   I]]

        This is the standard 'constant-velocity with acceleration noise' model.
        """
        dt = self.dt
        s2 = float(sigma_a) ** 2

        Q_pos = (dt**4) / 4.0 * s2
        Q_pv  = (dt**3) / 2.0 * s2
        Q_vel = (dt**2) * s2

        Q = np.zeros((6, 6), dtype=float)
        Q[0:3, 0:3] = np.eye(3) * Q_pos
        Q[0:3, 3:6] = np.eye(3) * Q_pv
        Q[3:6, 0:3] = np.eye(3) * Q_pv
        Q[3:6, 3:6] = np.eye(3) * Q_vel
        self.Q = Q

    # ---------- Predict ----------

    def predict(self, a_world):
        """Prediction using world-frame acceleration (gravity already removed)."""
        dt = self.dt

        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        B = np.zeros((6, 3), dtype=float)
        B[0:3, :] = 0.5 * (dt**2) * np.eye(3)  # position integrates accel
        B[3:6, :] = dt * np.eye(3)             # velocity integrates accel

        a_world = np.asarray(a_world, dtype=float).reshape(3,)

        self.x = F @ self.x + B @ a_world
        self.P = F @ self.P @ F.T + self.Q

    # ---------- Updates + gating ----------

    @staticmethod
    def _nis(y, S):
        """Normalized Innovation Squared (scalar)."""
        return float(y.T @ np.linalg.inv(S) @ y)

    def update_linear(self, z, H, R, chi2_gate=None):
        """Standard linear KF update: z = Hx + noise (DVL, Depth).

        Returns:
            accepted (bool), nis (float)
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)

        x = self.x.reshape(-1, 1)
        y = z - H @ x
        S = H @ self.P @ H.T + R

        nis = self._nis(y, S)
        self.last_nis = nis

        if chi2_gate is not None and nis > float(chi2_gate):
            # Reject as outlier
            return False, nis

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = (x + K @ y).flatten()

        I = np.eye(self.P.shape[0], dtype=float)
        self.P = (I - K @ H) @ self.P
        return True, nis

    def update_range(self, z_range, beacon_pos, R_range, chi2_gate=None):
        """Nonlinear EKF update for range-only measurement:
            z_range ~ || position - beacon_pos ||

        Returns:
            accepted (bool), nis (float)
        """
        beacon_pos = np.asarray(beacon_pos, dtype=float).reshape(3,)
        px, py, pz = self.x[0:3]
        bx, by, bz = beacon_pos

        dx = px - bx
        dy = py - by
        dz = pz - bz
        dist_pred = float(np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9)

        # h(x)
        h = dist_pred

        # Jacobian wrt state [x,y,z,vx,vy,vz]
        H = np.zeros((1, 6), dtype=float)
        H[0, 0] = dx / dist_pred
        H[0, 1] = dy / dist_pred
        H[0, 2] = dz / dist_pred

        z = np.array([[float(z_range)]], dtype=float)
        Rm = np.array([[float(R_range)]], dtype=float)

        y = z - np.array([[h]], dtype=float)
        S = H @ self.P @ H.T + Rm

        nis = self._nis(y, S)
        self.last_nis = nis

        if chi2_gate is not None and nis > float(chi2_gate):
            return False, nis

        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = (self.x.reshape(-1, 1) + K @ y).flatten()
        I = np.eye(self.P.shape[0], dtype=float)
        self.P = (I - K @ H) @ self.P
        return True, nis


# ===================== Utilities =====================

def compute_rmse(true, est):
    """Return (total_rmse, per_axis_rmse) for arrays shape (N,3)."""
    true = np.asarray(true, dtype=float)
    est = np.asarray(est, dtype=float)
    err = est - true
    mse_axis = np.mean(err**2, axis=0)
    rmse_axis = np.sqrt(mse_axis)
    mse_total = np.mean(np.sum(err**2, axis=1))
    rmse_total = np.sqrt(mse_total)
    return rmse_total, rmse_axis
