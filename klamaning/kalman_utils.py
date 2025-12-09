import numpy as np

# ===================== EKF =====================

class EKF:
    """
    Simple 6D EKF state:
        x = [x, y, z, vx, vy, vz]^T  (world frame)
    """

    def __init__(self, dt,
                 q_pos_std=0.02,
                 q_vel_std=0.2,
                 p_pos_std_init=1.0,
                 p_vel_std_init=1.0):
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

        # Process noise
        self.Q = np.diag([
            q_pos_std**2,
            q_pos_std**2,
            q_pos_std**2,
            q_vel_std**2,
            q_vel_std**2,
            q_vel_std**2,
        ])

    def predict(self, a_world):
        """Prediction using world-frame acceleration (gravity already removed)."""
        dt = self.dt

        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        B = np.zeros((6, 3))
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt

        a_world = np.asarray(a_world).reshape(3,)

        self.x = F @ self.x + B @ a_world
        self.P = F @ self.P @ F.T + self.Q

    def update_linear(self, z, H, R):
        """Standard linear KF update: z = Hx + noise (DVL, Depth)."""
        z = np.asarray(z).reshape(-1, 1)
        H = np.asarray(H)
        R = np.asarray(R)

        x = self.x.reshape(-1, 1)
        y = z - H @ x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = (x + K @ y).flatten()
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

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
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = (self.x.reshape(-1, 1) + K @ y).flatten()
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P


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
