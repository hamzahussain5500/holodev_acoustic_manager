import numpy as np

# ===================== EKF =====================

class EKF:
    """
    Simple 6D EKF state:
        x = [x, y, z, vx, vy, vz]^T  (world frame)
    """

    def __init__(
        self,
        dt,
        q_pos_std=0.02,
        q_vel_std=0.2,
        p_pos_std_init=1.0,
        p_vel_std_init=1.0,
    ):
        self.dt = dt

        # State
        self.x = np.zeros(6)

        # Covariance
        self.P = np.diag([
            p_pos_std_init**2,
            p_pos_std_init**2,
            p_pos_std_init**2,
            p_vel_std_init**2,
            p_vel_std_init**2,
            p_vel_std_init**2,
        ])

        # --- Process-noise parameters ("Q fix") ---
        # This EKF uses IMU acceleration as an *input* (control). Process noise should
        # therefore model uncertainty in that acceleration input.
        #
        # We interpret:
        #   q_vel_std -> sigma_a  (accel uncertainty std, m/s^2)
        #   q_pos_std -> optional position random-walk std (m / sqrt(s))
        #
        # NOTE: For most underwater cases, set q_pos_std ~= 0 and tune q_vel_std.
        self.q_pos_std = float(q_pos_std)
        self.sigma_a = float(q_vel_std)

        # Tiny numerical jitter to keep P well-conditioned
        self._eps = 1e-12

    def predict(self, a_world, sigma_a=None):
        """Prediction using world-frame acceleration (gravity already removed).

        Parameters
        ----------
        a_world : array_like shape (3,)
            World-frame linear acceleration.
        sigma_a : float, optional
            If provided, overrides the default accel uncertainty std used to build Q.
        """
        dt = self.dt

        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Control-input matrix for acceleration.
        # We use the standard constant-acceleration discretization:
        #   p_{k+1} = p_k + v_k dt + 0.5 a dt^2
        #   v_{k+1} = v_k + a dt
        B = np.zeros((6, 3))
        B[0, 0] = 0.5 * dt * dt
        B[1, 1] = 0.5 * dt * dt
        B[2, 2] = 0.5 * dt * dt
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt

        a_world = np.asarray(a_world).reshape(3,)

        self.x = F @ self.x + B @ a_world

        # Build Q from accel uncertainty: Q = B * Qa * B^T
        sa = float(self.sigma_a if sigma_a is None else sigma_a)
        Qa = (sa * sa) * np.eye(3)
        Q = B @ Qa @ B.T

        # Optional position random-walk (scaled with dt).
        # Interpreting q_pos_std as m/sqrt(s): var increment per step is q^2 * dt.
        if self.q_pos_std > 0:
            q2dt = (self.q_pos_std * self.q_pos_std) * dt
            Q += np.diag([q2dt, q2dt, q2dt, 0.0, 0.0, 0.0])

        self.P = F @ self.P @ F.T + Q + self._eps * np.eye(6)
        self.P = 0.5 * (self.P + self.P.T)

    def update_linear(self, z, H, R):
        """Standard linear KF update: z = Hx + noise (DVL, Depth)."""
        z = np.asarray(z).reshape(-1, 1)
        H = np.asarray(H)
        R = np.asarray(R)

        x = self.x.reshape(-1, 1)
        y = z - H @ x
        S = H @ self.P @ H.T + R
        # Use solve instead of explicit inverse for numerical stability
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        self.x = (x + K @ y).flatten()

        # Joseph stabilized covariance update (keeps P symmetric PSD)
        I = np.eye(self.P.shape[0])
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_range(self, z_range, beacon_pos, R_range):
        """
        Nonlinear EKF update for range-only measurement:
            z_range ~ || position - beacon_pos ||
        """
        beacon_pos = np.asarray(beacon_pos).reshape(3,)
        px, py, pz = self.x[0:3]
        bx, by, bz = beacon_pos

        dx = px - bx
        dy = py - by
        dz = pz - bz
        dist_pred = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9

        # h(x)
        h = dist_pred

        # Jacobian wrt state [x,y,z,vx,vy,vz]
        H = np.zeros((1, 6))
        H[0, 0] = dx / dist_pred
        H[0, 1] = dy / dist_pred
        H[0, 2] = dz / dist_pred

        z = np.array([[z_range]])
        Rm = np.array([[R_range]])

        y = z - np.array([[h]])
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        self.x = (self.x.reshape(-1, 1) + K @ y).flatten()

        I = np.eye(self.P.shape[0])
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)


# ===================== Utilities =====================

def compute_rmse(true, est):
    """Return (total_rmse, per_axis_rmse) for arrays shape (N,3)."""
    true = np.asarray(true)
    est = np.asarray(est)
    err = est - true
    mse_axis = np.mean(err**2, axis=0)
    rmse_axis = np.sqrt(mse_axis)
    mse_total = np.mean(np.sum(err**2, axis=1))
    rmse_total = np.sqrt(mse_total)
    return rmse_total, rmse_axis


def make_q_cv_3d(dt, sigma_a):
    """Discrete-time process-noise covariance for a 3D constant-velocity model.

    State: [x, y, z, vx, vy, vz]^T
    Assumption: unknown acceleration is white noise with std = sigma_a (m/s^2)
    Then: Q = B * (sigma_a^2 I) * B^T where
        B = [[0.5 dt^2 I],
             [dt I]]

    This is the same structure used inside EKF.predict() above.
    """
    dt = float(dt)
    sa2 = float(sigma_a) ** 2
    B = np.zeros((6, 3))
    B[0, 0] = 0.5 * dt * dt
    B[1, 1] = 0.5 * dt * dt
    B[2, 2] = 0.5 * dt * dt
    B[3, 0] = dt
    B[4, 1] = dt
    B[5, 2] = dt
    return B @ (sa2 * np.eye(3)) @ B.T
