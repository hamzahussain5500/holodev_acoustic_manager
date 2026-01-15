"""
EKF FUSION (IMU + DVL + Depth + Acoustic_1) in Holoocean
---------------------------------------------------------

Runs a single configuration:
- IMU prediction
- DVL velocity update
- Depth (z) update
- Acoustic range update from USV_1

Outputs basic metrics and plots for the single run.
"""

import holoocean
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from uncertainty_utils import covariance_ellipse_points, covariance_ellipsoid_mesh
from kalman_utils import EKF, compute_rmse


# ============================================================
# TUNABLE PARAMETERS
# ============================================================

# Scenario & agent configuration
SCENARIO_NAME = "usv_auv"   # set this to your scenario name
AUV_NAME      = "auv"
USV_1_NAME    = "usv1"
USV_2_NAME    = "usv2"
USV_3_NAME    = "usv3"
USV_4_NAME    = "usv4"

# Acoustic beacon IDs (must match scenario config)
AUV_BEACON_ID    = 0
USV_BEACON_ID_1  = 1
USV_BEACON_ID_2  = 2
USV_BEACON_ID_3  = 3
USV_BEACON_ID_4  = 4
# Simulation timing
DEFAULT_TICKS_PER_SEC = 30
SIM_DURATION_SEC = 120.0

# Gravity in WORLD frame
GRAVITY_WORLD = np.array([0.0, 0.0, 9.81])

# EKF process noise (Q)
Q_POS_STD = 0.02
Q_VEL_STD = 0.2

# EKF initial covariance (P)
P_POS_STD_INIT = 1.0
P_VEL_STD_INIT = 1.0

# DVL measurement noise (R)
DVL_VEL_STD = 0.2626

# Depth measurement noise (R_depth)
DEPTH_STD = 0.2626

# Acoustic range noise (R_range)
ACOUSTIC_RANGE_STD = 0.5

# Acoustic ping interval (in ticks)
ACOUSTIC_UPDATE_PERIOD_TICKS = 15

# ===== Currents control =====
USE_CURRENTS = True
VEHICLES_FOR_CURRENTS = [AUV_NAME]
MAP_DIMENSIONS = [100, 100, 35]
DRAW_CURRENT_FIELD_STEP = 10

# ===== PID waypoint navigation (6-DOF targets) =====
# Desired constant depth (z) and yaw for the AUV. If world z is positive-down,
# set TARGET_Z accordingly.
TARGET_Z = -15.0
TARGET_YAW = 0.0  # radians

# Waypoints in XY to follow; lifted to [x, y, z, roll, pitch, yaw]
WAYPOINTS_XY = np.array([
    [ 15.0,  15.0],
    [-15.0,  15.0],
    [-15.0, -15.0],
    [ 15.0, -15.0],
    [ 0.0,  0.0],
    [ 10.0,  0.0],
    [ 0.0,  10.0],
    [-10.0,  0.0],
    [ 0.0, -20.0]
], dtype=float)

# Proximity threshold (meters) to switch to next waypoint
POS_TOL = 1.0


# ============================================================
# CURRENT FIELD
# ============================================================

def vortex_field(location):
    x, y, z = location
    cx, cy = 0.0, 0.0
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2) + 1e-6
    strength = 2.0
    v_theta = strength / r
    vx = -v_theta * dy
    vy =  v_theta * dx
    vz = 0.0
    return np.array([vx, vy, vz], dtype=float)


def apply_currents(env, state, clock):
    if not USE_CURRENTS:
        return
    if clock == DRAW_CURRENT_FIELD_STEP:
        env.draw_debug_vector_field(
            vortex_field,
            location=[0, 0, 0],
            vector_field_dimensions=MAP_DIMENSIONS,
            arrow_thickness=7,
            arrow_size=.25,
            spacing=3
        )
    for vehicle in VEHICLES_FOR_CURRENTS:
        if "LocationSensor" not in state[vehicle]:
            continue
        location = state[vehicle]["LocationSensor"]
        current_velocity = vortex_field(location)
        env.set_ocean_currents(vehicle, current_velocity)


# ============================================================
# SINGLE-RUN EKF LOOP (IMU + DVL + Depth + Acoustic_1)
# ============================================================

def run_ekf_acoustics():
    true_positions = []
    est_positions = []
    true_velocities = []
    est_velocities = []
    pos_covariances = []
    times = []

    with holoocean.make(SCENARIO_NAME, show_viewport=True, frames_per_sec=False) as env:
        ticks_per_sec = getattr(env, "ticks_per_sec", DEFAULT_TICKS_PER_SEC)
        dt = 1.0 / float(ticks_per_sec)

        ekf = EKF(dt=dt)

        n_steps = int(SIM_DURATION_SEC * ticks_per_sec)
        prev_true_pos = None
        clock = 0
        idx = 0

        # Visualize target waypoints at the chosen depth
        for xy in WAYPOINTS_XY:
            env.draw_point([xy[0], xy[1], TARGET_Z], lifetime=0)

        for k in range(n_steps):
            clock += 1
            r_meas_1 = None
            beacon_1_pos = None

            # Build 6-DOF PID target and step environment
            x, y = WAYPOINTS_XY[idx]
            target_6d = np.array([x, y, TARGET_Z, 0.0, 0.0, TARGET_YAW], dtype=float)
            state = env.step(target_6d)

            # Periodic acoustic ping from USV_1 -> AUV
            if k % ACOUSTIC_UPDATE_PERIOD_TICKS == 0:
                env.send_acoustic_message(AUV_BEACON_ID, USV_BEACON_ID_1, "MSG_REQX", "ping")
                # Advance one step to process the ping within the sim
                state = env.step(target_6d)

            # Apply currents
            apply_currents(env, state, clock)

            # Read sensors
            imu   = state[AUV_NAME]["IMUSensor"]
            pose  = state[AUV_NAME]["PoseSensor"]
            dvl   = state[AUV_NAME]["DVLSensor"]
            depth = state[AUV_NAME]["DepthSensor"]

            accel_meas = imu[0, :]
            accel_bias = imu[2, :] if imu.shape[0] >= 3 else np.zeros(3)
            accel_body = accel_meas - accel_bias

            T_ws = pose
            R_ws = T_ws[0:3, 0:3]
            true_pos = T_ws[0:3, 3]

            # True velocity (for evaluation only)
            if prev_true_pos is None:
                true_vel = np.zeros(3)
            else:
                true_vel = (true_pos - prev_true_pos) / dt
            prev_true_pos = true_pos.copy()

            # IMU prediction
            a_world_raw = R_ws @ accel_body
            a_world = a_world_raw - GRAVITY_WORLD

            if k == 0:
                ekf.x[0:3] = true_pos.copy()
                ekf.x[3:6] = true_vel.copy()

            ekf.predict(a_world)

            # DVL update (world-frame velocity)
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

            # Depth update (z)
            z_meas = float(depth[0])
            z_vec = np.array([z_meas])
            H_depth = np.array([[0, 0, 1, 0, 0, 0]])
            R_depth = np.array([[DEPTH_STD**2]])
            ekf.update_linear(z_vec, H_depth, R_depth)

            # Acoustic_1 range update (nonlinear)
            if "AcousticBeaconSensor" in state[AUV_NAME]:
                acoustic_msg = state[AUV_NAME]["AcousticBeaconSensor"]
                if acoustic_msg is not None and len(acoustic_msg) >= 6:
                    if acoustic_msg[0] == "MSG_RESPX" and acoustic_msg[1] == USV_BEACON_ID_1:
                        r_meas_1 = float(acoustic_msg[5])
                        print(r_meas_1)
                        # USV (beacon) position in world frame
                        if "PoseSensor" in state[USV_1_NAME]:
                            usv_1_pose = state[USV_1_NAME]["PoseSensor"]
                            beacon_1_pos = usv_1_pose[0:3, 3]
                        else:
                            beacon_1_pos = state[USV_1_NAME]["LocationSensor"]

            if (r_meas_1 is not None) and (beacon_1_pos is not None):
                ekf.update_range(r_meas_1, beacon_1_pos, ACOUSTIC_RANGE_STD**2)

            # Waypoint switching: stop after last waypoint is reached
            pos = state[AUV_NAME]["LocationSensor"][0:3]
            if np.linalg.norm(pos - np.array([x, y, TARGET_Z])) <= POS_TOL:
                if idx >= len(WAYPOINTS_XY) - 1:
                    print("Final waypoint reached. Ending simulation.")
                    break
                else:
                    idx += 1

            # Logging
            t = k * dt
            times.append(t)
            true_positions.append(true_pos.copy())
            est_positions.append(ekf.x[0:3].copy())
            true_velocities.append(true_vel.copy())
            est_velocities.append(ekf.x[3:6].copy())
            pos_covariances.append(ekf.P[0:3, 0:3].copy())



    times = np.array(times)
    true_positions = np.array(true_positions)
    est_positions = np.array(est_positions)
    true_velocities = np.array(true_velocities)
    est_velocities = np.array(est_velocities)
    pos_covariances = np.array(pos_covariances)

    return times, true_positions, est_positions, true_velocities, est_velocities, pos_covariances


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nRunning EKF with IMU + DVL + Depth + Acoustic_1...")
    (t,
     true_pos,
     est_pos,
     true_vel,
     est_vel,
     Ppos) = run_ekf_acoustics()

    # Metrics
    pos_rmse, pos_axis = compute_rmse(true_pos, est_pos)
    vel_rmse, vel_axis = compute_rmse(true_vel, est_vel)
    final_err = np.linalg.norm(est_pos[-1] - true_pos[-1])
    print("\n========== EKF PERFORMANCE ==========")
    print(f"  Position RMSE total  : {pos_rmse:.3f} m")
    print(f"    axes (x,y,z)       : {pos_axis}")
    print(f"  Velocity RMSE total  : {vel_rmse:.3f} m/s")
    print(f"    axes (x,y,z)       : {vel_axis}")
    print(f"  Final position error : {final_err:.3f} m")
    print("====================================\n")

    chi2_2d_95 = 5.991
    chi2_3d_95 = 7.815

    # XY trajectory + ellipse
    plt.figure()
    plt.plot(true_pos[:, 0], true_pos[:, 1], label="true", color="k", linewidth=2)
    plt.plot(est_pos[:, 0], est_pos[:, 1], "--", label="EKF", color="purple")
    idxs = np.linspace(0, len(est_pos) - 1, 4, dtype=int)
    for idx in idxs:
        P_xy = Ppos[idx][0:2, 0:2]
        mean_xy = est_pos[idx, 0:2]
        ex, ey = covariance_ellipse_points(P_xy, mean_xy, chi2_val=chi2_2d_95)
        plt.plot(ex, ey, color="purple", alpha=0.5, linewidth=1)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("XY trajectory with uncertainty ellipses")
    plt.axis("equal")
    plt.legend()

    # Z vs time
    plt.figure()
    plt.plot(t, true_pos[:, 2], label="true z", color="k")
    plt.plot(t, est_pos[:, 2], "--", label="EKF z", color="purple")
    plt.xlabel("time [s]")
    plt.ylabel("z [m]")
    plt.title("Z position vs time")
    plt.legend()

    # Position error vs time
    pos_err = np.linalg.norm(est_pos - true_pos, axis=1)
    plt.figure()
    plt.plot(t, pos_err, label="error", color="purple")
    plt.xlabel("time [s]")
    plt.ylabel("||position error|| [m]")
    plt.title("Position error over time")
    plt.legend()

    # 3D uncertainty bubble (final)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    idx = -1
    mean = est_pos[idx]
    P = Ppos[idx]
    X, Y, Z = covariance_ellipsoid_mesh(P, mean, chi2_val=chi2_3d_95, num_u=25, num_v=25)
    ax.plot_surface(X, Y, Z, alpha=0.3, color="purple", edgecolor="none")
    ax.scatter(true_pos[idx, 0], true_pos[idx, 1], true_pos[idx, 2], color="k", label="true pos")
    ax.scatter(mean[0], mean[1], mean[2], color="purple", label="EKF mean")
    ax.set_title("IMU + DVL + Depth + Acoustic_1: 95% uncertainty bubble")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
