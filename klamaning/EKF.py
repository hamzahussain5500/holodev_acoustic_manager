"""
EKF FUSION OF IMU, DVL, DEPTH, ACOUSTIC IN HOLOOCEAN
----------------------------------------------------

Compares four configurations:

1) IMU-only
2) IMU + DVL
3) IMU + DVL + Depth
4) IMU + DVL + Depth + Acoustic Range

Uses:
- PoseSensor  -> ground truth pose, rotation
- IMUSensor   -> acceleration (prediction)
- DVLSensor   -> velocity (update)
- DepthSensor -> z position (update)
- AcousticBeaconSensor (AUV + USV) + send_acoustic_message
               -> range-only measurement (nonlinear EKF update)
- LocationSensor on USV -> beacon position

Also:
- Keyboard control
- Optional currents
- XY trajectory with uncertainty ellipses
- Separate 3D uncertainty bubbles per configuration
"""

import holoocean
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from uncertainty_utils import covariance_ellipse_points, covariance_ellipsoid_mesh
from kalman_utils import EKF, compute_rmse


# ============================================================
# TUNABLE PARAMETERS
# ============================================================

# Scenario & agent configuration
SCENARIO_NAME = "blue_rov"   # <-- set this to your scenario name
AUV_NAME      = "auv"       # <-- AUV agent name
USV_NAME = "usv1"   # three surface beacons

# Acoustic beacon IDs (must match scenario config)
AUV_BEACON_ID = 0
USV_BEACON_ID = 1




# Simulation timing
DEFAULT_TICKS_PER_SEC = 30
SIM_DURATION_SEC = 20.0      # total simulation length in seconds

# Keyboard/thruster control
BASE_THRUSTER_FORCE = 15.0   # magnitude of command applied by each key

# Gravity in WORLD frame
GRAVITY_WORLD = np.array([0.0, 0.0, 9.81])

# EKF process noise (Q)
Q_POS_STD = 0.02   # std dev for position noise per step (m)
Q_VEL_STD = 0.2    # std dev for velocity noise per step (m/s)

# EKF initial covariance (P)
P_POS_STD_INIT = 1.0   # initial pos uncertainty (m)
P_VEL_STD_INIT = 1.0   # initial vel uncertainty (m/s)

# DVL measurement noise (R)
DVL_VEL_STD = 0.2626     # std dev for each velocity component (m/s)

# Depth measurement noise (R_depth)
DEPTH_STD = 0.2626       # depth sensor std dev (m)

# Acoustic range noise (R_range)
ACOUSTIC_RANGE_STD = 0.5  # std dev of range measurement (m), tune from DistanceSigma

# Acoustic ping interval (in ticks)
ACOUSTIC_UPDATE_PERIOD_TICKS = 30  # ~1 second if ticks_per_sec=30

# ===== Currents control =====
USE_CURRENTS = True
VEHICLES_FOR_CURRENTS = [AUV_NAME, USV_NAME]  # currents applied to both
MAP_DIMENSIONS = [100, 100, 35]
DRAW_CURRENT_FIELD_STEP = 100


# ============================================================
# CURRENT FIELD
# ============================================================

def vortex_field(location):
    """
    Example vortex-like current field in XY plane.
    Replace with your own current model if desired.
    """
    x, y, z = location

    cx, cy = 0.0, 0.0
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2) + 1e-6

    strength = 10
    v_theta = strength / r

    vx = -v_theta * dy
    vy =  v_theta * dx
    vz = 0.0

    return np.array([vx, vy, vz], dtype=float)


def apply_currents(env, state, clock):
    """Apply currents to vehicles if USE_CURRENTS is True."""
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
# EKF CLASS
# ============================================================



# ============================================================
# KEYBOARD CONTROL
# ============================================================

pressed_keys = set()


def on_press(key):
    if hasattr(key, "char") and key.char is not None:
        pressed_keys.add(key.char)


def on_release(key):
    if hasattr(key, "char") and key.char is not None:
        pressed_keys.discard(key.char)


def parse_keys(keys, val):
    """Convert pressed keys into 8D thruster command vector."""
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


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# ============================================================
# UTILITIES
# ============================================================

# compute_rmse imported from kalman_utils


# ============================================================
# MAIN EKF + CONTROL LOOP
# ============================================================

def run_ekf_with_control(use_dvl_update: bool,
                         use_depth_update: bool,
                         use_acoustic_update: bool):
    """
    Run one experiment with chosen sensor combo.

    Flags:
    - use_dvl_update      -> include DVL velocity corrections
    - use_depth_update    -> include Depth z update
    - use_acoustic_update -> include range update to USV
    """
    true_positions = []
    est_positions = []
    true_velocities = []
    est_velocities = []
    pos_covariances = []
    times = []

    with holoocean.make(SCENARIO_NAME, show_viewport = False, frames_per_sec= False) as env:
        ticks_per_sec = getattr(env, "ticks_per_sec", DEFAULT_TICKS_PER_SEC)
        dt = 1.0 / float(ticks_per_sec)

        ekf = EKF(dt=dt)

        n_steps = int(SIM_DURATION_SEC * ticks_per_sec)
        prev_true_pos = None
        clock = 0

        for k in range(n_steps):
            clock += 1

            # 1) Apply keyboard control to AUV
            command = parse_keys(pressed_keys, BASE_THRUSTER_FORCE)
            env.act(AUV_NAME, command)

            state = env.tick()


            # 2) Optionally send acoustic ping from USV -> AUV
            if use_acoustic_update and (k % ACOUSTIC_UPDATE_PERIOD_TICKS == 0):
                env.send_acoustic_message(AUV_BEACON_ID,
                                          USV_BEACON_ID,
                                          "MSG_REQX",
                                          "ping")
                state= env.tick()
                
                

            # 3) Step environment

            # 4) Apply currents
            apply_currents(env, state, clock)

            # 5) Read sensors
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

            # 6) IMU prediction
            a_world_raw = R_ws @ accel_body
            a_world = a_world_raw - GRAVITY_WORLD

            if k == 0:
                ekf.x[0:3] = true_pos.copy()
                ekf.x[3:6] = true_vel.copy()

            ekf.predict(a_world)

            # 7) DVL update
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

            # 8) Depth update (z)
            if use_depth_update:
                # If DepthSensor returns positive "depth", use: z_meas = -depth[0]
                z_meas = float(depth[0])
                z_vec = np.array([z_meas])
                H_depth = np.array([[0, 0, 1, 0, 0, 0]])
                R_depth = np.array([[DEPTH_STD**2]])
                ekf.update_linear(z_vec, H_depth, R_depth)

            # 9) Acoustic range update (nonlinear EKF)
            if use_acoustic_update:
                # Read last acoustic message at AUV
                if "AcousticBeaconSensor" in state[AUV_NAME]:
                    acoustic_msg = state[AUV_NAME]["AcousticBeaconSensor"]
                    # Expect: ["MSG_RESPX", from_id, payload, phi, theta, r, d]
                    if acoustic_msg is not None and len(acoustic_msg) >= 6:
                        msg_type = acoustic_msg[0]
                        if msg_type == "MSG_RESPX":
                            r_meas = float(acoustic_msg[5]) # r
                            print (r_meas)

                            # USV (beacon) position in world frame
                            if "PoseSensor" in state[USV_NAME]:
                                usv_pose = state[USV_NAME]["PoseSensor"]
                                beacon_pos = usv_pose[0:3, 3]
                            else:
                                beacon_pos = state[USV_NAME]["LocationSensor"]

                            ekf.update_range(r_meas,
                                            beacon_pos,
                                            ACOUSTIC_RANGE_STD**2)


            # 10) Logging
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


def main():

    # ----- RUN 4: IMU + DVL + Depth + Acoustic -----
    print("\nRunning EKF with IMU + DVL + Depth + Acoustic...")
    (t_dda,
     true_pos_dda,
     est_pos_dda,
     true_vel_dda,
     est_vel_dda,
     Ppos_dda) = run_ekf_with_control(
        use_dvl_update=True,
        use_depth_update=True,
        use_acoustic_update=True
    )
    # ----- RUN 1: IMU-only -----
    print("\nRunning EKF with IMU ONLY...")
    (t_imu,
     true_pos_imu,
     est_pos_imu,
     true_vel_imu,
     est_vel_imu,
     Ppos_imu) = run_ekf_with_control(
        use_dvl_update=False,
        use_depth_update=False,
        use_acoustic_update=False
    )

    # ----- RUN 2: IMU + DVL -----
    print("\nRunning EKF with IMU + DVL...")
    (t_dvl,
     true_pos_dvl,
     est_pos_dvl,
     true_vel_dvl,
     est_vel_dvl,
     Ppos_dvl) = run_ekf_with_control(
        use_dvl_update=True,
        use_depth_update=False,
        use_acoustic_update=False
    )

    # ----- RUN 3: IMU + DVL + Depth -----
    print("\nRunning EKF with IMU + DVL + Depth...")
    (t_dd,
     true_pos_dd,
     est_pos_dd,
     true_vel_dd,
     est_vel_dd,
     Ppos_dd) = run_ekf_with_control(
        use_dvl_update=True,
        use_depth_update=True,
        use_acoustic_update=False
    )



    # ===== METRICS =====
    def print_metrics(name, true_pos, est_pos, true_vel, est_vel):
        pos_rmse, pos_axis = compute_rmse(true_pos, est_pos)
        vel_rmse, vel_axis = compute_rmse(true_vel, est_vel)
        final_err = np.linalg.norm(est_pos[-1] - true_pos[-1])
        print(f"{name}:")
        print(f"  Position RMSE total  : {pos_rmse:.3f} m")
        print(f"    axes (x,y,z)       : {pos_axis}")
        print(f"  Velocity RMSE total  : {vel_rmse:.3f} m/s")
        print(f"    axes (x,y,z)       : {vel_axis}")
        print(f"  Final position error : {final_err:.3f} m\n")

    print("\n========== EKF PERFORMANCE COMPARISON ==========")
    print_metrics("IMU ONLY", true_pos_imu, est_pos_imu, true_vel_imu, est_vel_imu)
    print_metrics("IMU + DVL", true_pos_dvl, est_pos_dvl, true_vel_dvl, est_vel_dvl)
    print_metrics("IMU + DVL + Depth", true_pos_dd, est_pos_dd, true_vel_dd, est_vel_dd)
    print_metrics("IMU + DVL + Depth + Acoustic", true_pos_dda, est_pos_dda, true_vel_dda, est_vel_dda)
    print("================================================\n")

    chi2_2d_95 = 5.991
    chi2_3d_95 = 7.815

    # ===== 2D XY TRAJECTORY + ELLIPSES =====
    plt.figure()
    plt.plot(true_pos_imu[:, 0], true_pos_imu[:, 1],
             label="true (IMU run)", color="k", linewidth=2)
    plt.plot(est_pos_imu[:, 0], est_pos_imu[:, 1],
             "--", label="EKF IMU-only", color="red")
    plt.plot(est_pos_dvl[:, 0], est_pos_dvl[:, 1],
             "--", label="EKF IMU + DVL", color="green")
    plt.plot(est_pos_dd[:, 0], est_pos_dd[:, 1],
             "--", label="EKF IMU + DVL + Depth", color="blue")
    plt.plot(est_pos_dda[:, 0], est_pos_dda[:, 1],
             "--", label="EKF IMU+DVL+Depth+Acoustic", color="purple")

    num_ellipses = 4
    idxs_imu = np.linspace(0, len(est_pos_imu) - 1, num_ellipses, dtype=int)
    idxs_dvl = np.linspace(0, len(est_pos_dvl) - 1, num_ellipses, dtype=int)
    idxs_dd  = np.linspace(0, len(est_pos_dd)  - 1, num_ellipses, dtype=int)
    idxs_dda = np.linspace(0, len(est_pos_dda) - 1, num_ellipses, dtype=int)

    for idx in idxs_imu:
        P_xy = Ppos_imu[idx][0:2, 0:2]
        mean_xy = est_pos_imu[idx, 0:2]
        ex, ey = covariance_ellipse_points(P_xy, mean_xy, chi2_val=chi2_2d_95)
        plt.plot(ex, ey, color="red", alpha=0.3, linewidth=1)

    for idx in idxs_dvl:
        P_xy = Ppos_dvl[idx][0:2, 0:2]
        mean_xy = est_pos_dvl[idx, 0:2]
        ex, ey = covariance_ellipse_points(P_xy, mean_xy, chi2_val=chi2_2d_95)
        plt.plot(ex, ey, color="green", alpha=0.4, linewidth=1)

    for idx in idxs_dd:
        P_xy = Ppos_dd[idx][0:2, 0:2]
        mean_xy = est_pos_dd[idx, 0:2]
        ex, ey = covariance_ellipse_points(P_xy, mean_xy, chi2_val=chi2_2d_95)
        plt.plot(ex, ey, color="blue", alpha=0.4, linewidth=1)

    for idx in idxs_dda:
        P_xy = Ppos_dda[idx][0:2, 0:2]
        mean_xy = est_pos_dda[idx, 0:2]
        ex, ey = covariance_ellipse_points(P_xy, mean_xy, chi2_val=chi2_2d_95)
        plt.plot(ex, ey, color="purple", alpha=0.5, linewidth=1)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("XY trajectory with uncertainty ellipses")
    plt.axis("equal")
    plt.legend()

    # ===== Z vs TIME =====
    plt.figure()
    plt.plot(t_imu, true_pos_imu[:, 2], label="true z (IMU run)", color="k")
    plt.plot(t_imu, est_pos_imu[:, 2], "--", label="EKF IMU-only z", color="red")
    plt.plot(t_dvl, est_pos_dvl[:, 2], "--", label="EKF IMU + DVL z", color="green")
    plt.plot(t_dd,  est_pos_dd[:, 2],  "--", label="EKF IMU + DVL + Depth z", color="blue")
    plt.plot(t_dda, est_pos_dda[:, 2], "--", label="EKF IMU+DVL+Depth+Acoustic z", color="purple")
    plt.xlabel("time [s]")
    plt.ylabel("z [m]")
    plt.title("Z position vs time")
    plt.legend()

    # ===== POSITION ERROR vs TIME =====
    pos_err_imu = np.linalg.norm(est_pos_imu - true_pos_imu, axis=1)
    pos_err_dvl = np.linalg.norm(est_pos_dvl - true_pos_dvl, axis=1)
    pos_err_dd  = np.linalg.norm(est_pos_dd  - true_pos_dd,  axis=1)
    pos_err_dda = np.linalg.norm(est_pos_dda - true_pos_dda, axis=1)

    plt.figure()
    plt.plot(t_imu, pos_err_imu, label="IMU-only", color="red")
    plt.plot(t_dvl, pos_err_dvl, label="IMU + DVL", color="green")
    plt.plot(t_dd,  pos_err_dd,  label="IMU + DVL + Depth", color="blue")
    plt.plot(t_dda, pos_err_dda, label="IMU + DVL + Depth + Acoustic", color="purple")
    plt.xlabel("time [s]")
    plt.ylabel("||position error|| [m]")
    plt.title("Position error over time")
    plt.legend()

    # ===== 3D BUBBLES (SEPARATE FIGURES) =====
    def plot_bubble(title, est_pos, true_pos, Ppos, color):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        idx = -1
        mean = est_pos[idx]
        P = Ppos[idx]
        X, Y, Z = covariance_ellipsoid_mesh(P, mean,
                                            chi2_val=chi2_3d_95,
                                            num_u=25, num_v=25)
        ax.plot_surface(X, Y, Z, alpha=0.3, color=color, edgecolor="none")
        ax.scatter(true_pos[idx, 0], true_pos[idx, 1], true_pos[idx, 2],
                   color="k", label="true pos")
        ax.scatter(mean[0], mean[1], mean[2],
                   color=color, label="EKF mean")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend()

    plot_bubble("IMU-only 95% uncertainty bubble",
                est_pos_imu, true_pos_imu, Ppos_imu, "red")
    plot_bubble("IMU + DVL 95% uncertainty bubble",
                est_pos_dvl, true_pos_dvl, Ppos_dvl, "green")
    plot_bubble("IMU + DVL + Depth 95% uncertainty bubble",
                est_pos_dd, true_pos_dd, Ppos_dd, "blue")
    plot_bubble("IMU + DVL + Depth + Acoustic 95% uncertainty bubble",
                est_pos_dda, true_pos_dda, Ppos_dda, "purple")

    plt.show()


if __name__ == "__main__":
    main()
