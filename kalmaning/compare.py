"""
EKF FUSION OF IMU + (OPTIONAL) DVL IN HOLOOCEAN, WITH KEYBOARD CONTROL
---------------------------------------------------------------------

This version can run:
- IMU-only EKF (prediction only)
- IMU + DVL EKF (prediction + velocity update)

It runs BOTH and compares:
- Trajectories
- Position error over time
- RMSE metrics

Keyboard mapping (same as before):
    i / k : forward / backward (all 4 main thrusters)
    j / l : yaw left / right
    w / s : up / down (vertical thrusters)
    a / d : roll left / right
"""

import holoocean
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard


# ============================================================
# TUNABLE PARAMETERS
# ============================================================

# Scenario & agent configuration
SCENARIO_NAME = "blue_rov"   # <-- set this to your scenario name
AUV_NAME = "auv"             # <-- set to your agent name ("auv", "auv0", etc.)

# Simulation timing
DEFAULT_TICKS_PER_SEC = 30
SIM_DURATION_SEC = 40.0      # total simulation length in seconds

# Keyboard/thruster control
BASE_THRUSTER_FORCE = 10.0   # magnitude of command applied by each key

# Gravity in WORLD frame (based on your IMU+Pose sample)
GRAVITY_WORLD = np.array([0.0, 0.0, 9.81])

# EKF process noise (Q)
Q_POS_STD = 0.02   # std dev for position noise per step (m)
Q_VEL_STD = 0.2    # std dev for velocity noise per step (m/s)

# EKF initial covariance (P)
P_POS_STD_INIT = 1.0   # initial pos uncertainty (m)
P_VEL_STD_INIT = 1.0   # initial vel uncertainty (m/s)

# DVL measurement noise (R)
DVL_VEL_STD = 0.05     # std dev for each velocity component (m/s)


# ============================================================
# EKF CLASS
# ============================================================

class EKF:
    """
    Simple 6D EKF:

        x = [x, y, z, vx, vy, vz]^T  (world frame)
    """

    def __init__(self, dt):
        self.dt = dt

        # State
        self.x = np.zeros(6)

        # Covariance
        self.P = np.diag([
            P_POS_STD_INIT**2,
            P_POS_STD_INIT**2,
            P_POS_STD_INIT**2,
            P_VEL_STD_INIT**2,
            P_VEL_STD_INIT**2,
            P_VEL_STD_INIT**2,
        ])

        # Process noise
        self.Q = np.diag([
            Q_POS_STD**2,
            Q_POS_STD**2,
            Q_POS_STD**2,
            Q_VEL_STD**2,
            Q_VEL_STD**2,
            Q_VEL_STD**2,
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
        """Standard linear KF update: z = Hx + noise."""
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


# ============================================================
# KEYBOARD CONTROL
# ============================================================

pressed_keys = set()   # currently pressed keys


def on_press(key):
    if hasattr(key, "char") and key.char is not None:
        pressed_keys.add(key.char)


def on_release(key):
    if hasattr(key, "char") and key.char is not None:
        pressed_keys.discard(key.char)


def parse_keys(keys, val):
    """
    Converts pressed keys into an 8D thruster command vector.
    """
    command = np.zeros(8)

    # forward/back
    if "i" in keys:
        command[0:4] += val
    if "k" in keys:
        command[0:4] -= val

    # yaw
    if "j" in keys:
        command[[4, 7]] += 0.25 * val
        command[[5, 6]] -= 0.25 * val
    if "l" in keys:
        command[[4, 7]] -= 0.25 * val
        command[[5, 6]] += 0.25 * val

    # vertical thrust
    if "w" in keys:
        command[4:8] += val
    if "s" in keys:
        command[4:8] -= val

    # roll
    if "a" in keys:
        command[[4, 6]] += val
        command[[5, 7]] -= val
    if "d" in keys:
        command[[4, 6]] -= val
        command[[5, 7]] += val

    return command


# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# ============================================================
# UTILITIES
# ============================================================

def compute_rmse(true, est):
    """Return (total_rmse, per_axis_rmse) for arrays shape (N,3)."""
    err = est - true
    mse_axis = np.mean(err**2, axis=0)
    rmse_axis = np.sqrt(mse_axis)
    mse_total = np.mean(np.sum(err**2, axis=1))
    rmse_total = np.sqrt(mse_total)
    return rmse_total, rmse_axis


# ============================================================
# MAIN EKF + CONTROL LOOP
# ============================================================

def run_ekf_with_control(use_dvl_update: bool):
    """
    Run one experiment:
    - use_dvl_update = False  -> IMU-only EKF
    - use_dvl_update = True   -> IMU + DVL EKF
    """
    true_positions = []
    est_positions = []
    true_velocities = []
    est_velocities = []
    times = []

    with holoocean.make(SCENARIO_NAME) as env:
        ticks_per_sec = getattr(env, "ticks_per_sec", DEFAULT_TICKS_PER_SEC)
        dt = 1.0 / float(ticks_per_sec)

        ekf = EKF(dt=dt)

        n_steps = int(SIM_DURATION_SEC * ticks_per_sec)
        prev_true_pos = None

        for k in range(n_steps):
            # 1) Apply keyboard control
            command = parse_keys(pressed_keys, BASE_THRUSTER_FORCE)
            env.act(AUV_NAME, command)

            # 2) Step environment
            state = env.tick()

            # 3) Read IMU & Pose (always needed)
            imu = state[AUV_NAME]["IMUSensor"]
            pose = state[AUV_NAME]["PoseSensor"]

            # DVL may be used or ignored depending on use_dvl_update
            dvl = state[AUV_NAME]["DVLSensor"]

            accel_meas = imu[0, :]
            accel_bias = imu[2, :] if imu.shape[0] >= 3 else np.zeros(3)
            accel_body = accel_meas - accel_bias

            T_ws = pose
            R_ws = T_ws[0:3, 0:3]   # body -> world
            true_pos = T_ws[0:3, 3]

            # 4) True velocity (for evaluation only)
            if prev_true_pos is None:
                true_vel = np.zeros(3)
            else:
                true_vel = (true_pos - prev_true_pos) / dt
            prev_true_pos = true_pos.copy()

            # 5) IMU prediction: body accel -> world, minus gravity
            a_world_raw = R_ws @ accel_body
            a_world = a_world_raw - GRAVITY_WORLD

            if k == 0:
                ekf.x[0:3] = true_pos.copy()
                ekf.x[3:6] = true_vel.copy()

            ekf.predict(a_world)

            # 6) Optional DVL update
            if use_dvl_update:
                v_body = dvl[0:3]
                v_world_meas = R_ws @ v_body

                H_dvl = np.array([
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ])

                R_dvl = np.diag([
                    DVL_VEL_STD**2,
                    DVL_VEL_STD**2,
                    DVL_VEL_STD**2,
                ])

                ekf.update_linear(v_world_meas, H_dvl, R_dvl)

            # 7) Logging
            t = k * dt
            times.append(t)
            true_positions.append(true_pos.copy())
            est_positions.append(ekf.x[0:3].copy())
            true_velocities.append(true_vel.copy())
            est_velocities.append(ekf.x[3:6].copy())

    times = np.array(times)
    true_positions = np.array(true_positions)
    est_positions = np.array(est_positions)
    true_velocities = np.array(true_velocities)
    est_velocities = np.array(est_velocities)

    return times, true_positions, est_positions, true_velocities, est_velocities


def main():
    # -------- RUN 1: IMU-only --------
    print("\nRunning EKF with IMU ONLY (no DVL update)...")
    (t_imu,
     true_pos_imu,
     est_pos_imu,
     true_vel_imu,
     est_vel_imu) = run_ekf_with_control(use_dvl_update=False)

    # -------- RUN 2: IMU + DVL --------
    print("\nRunning EKF with IMU + DVL...")
    (t_dvl,
     true_pos_dvl,
     est_pos_dvl,
     true_vel_dvl,
     est_vel_dvl) = run_ekf_with_control(use_dvl_update=True)

    # ----- METRICS -----
    pos_rmse_imu, pos_rmse_axis_imu = compute_rmse(true_pos_imu, est_pos_imu)
    vel_rmse_imu, vel_rmse_axis_imu = compute_rmse(true_vel_imu, est_vel_imu)
    final_pos_err_imu = np.linalg.norm(est_pos_imu[-1] - true_pos_imu[-1])

    pos_rmse_dvl, pos_rmse_axis_dvl = compute_rmse(true_pos_dvl, est_pos_dvl)
    vel_rmse_dvl, vel_rmse_axis_dvl = compute_rmse(true_vel_dvl, est_vel_dvl)
    final_pos_err_dvl = np.linalg.norm(est_pos_dvl[-1] - true_pos_dvl[-1])

    print("\n========== EKF PERFORMANCE COMPARISON ==========")
    print("IMU ONLY:")
    print(f"  Position RMSE total  : {pos_rmse_imu:.3f} m")
    print(f"    axes (x,y,z)       : {pos_rmse_axis_imu}")
    print(f"  Velocity RMSE total  : {vel_rmse_imu:.3f} m/s")
    print(f"    axes (x,y,z)       : {vel_rmse_axis_imu}")
    print(f"  Final position error : {final_pos_err_imu:.3f} m\n")

    print("IMU + DVL:")
    print(f"  Position RMSE total  : {pos_rmse_dvl:.3f} m")
    print(f"    axes (x,y,z)       : {pos_rmse_axis_dvl}")
    print(f"  Velocity RMSE total  : {vel_rmse_dvl:.3f} m/s")
    print(f"    axes (x,y,z)       : {vel_rmse_axis_dvl}")
    print(f"  Final position error : {final_pos_err_dvl:.3f} m")
    print("================================================\n")

    # ----- PLOTS -----

    # 1) XY trajectory comparison
    plt.figure()
    plt.plot(true_pos_imu[:, 0], true_pos_imu[:, 1],
             label="true (IMU run)", color="k", linewidth=2)
    plt.plot(est_pos_imu[:, 0], est_pos_imu[:, 1],
             "--", label="EKF IMU-only")
    plt.plot(est_pos_dvl[:, 0], est_pos_dvl[:, 1],
             "--", label="EKF IMU + DVL")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("XY trajectory: IMU-only vs IMU + DVL")
    plt.axis("equal")
    plt.legend()

    # 2) Z vs time comparison
    plt.figure()
    plt.plot(t_imu, true_pos_imu[:, 2], label="true z (IMU run)")
    plt.plot(t_imu, est_pos_imu[:, 2], "--", label="EKF IMU-only z")
    plt.plot(t_dvl, est_pos_dvl[:, 2], "--", label="EKF IMU + DVL z")
    plt.xlabel("time [s]")
    plt.ylabel("z [m]")
    plt.title("Z position vs time")
    plt.legend()

    # 3) Position error norm vs time
    pos_err_norm_imu = np.linalg.norm(est_pos_imu - true_pos_imu, axis=1)
    pos_err_norm_dvl = np.linalg.norm(est_pos_dvl - true_pos_dvl, axis=1)

    plt.figure()
    plt.plot(t_imu, pos_err_norm_imu, label="IMU-only")
    plt.plot(t_dvl, pos_err_norm_dvl, label="IMU + DVL")
    plt.xlabel("time [s]")
    plt.ylabel("||position error|| [m]")
    plt.title("Position error over time")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
