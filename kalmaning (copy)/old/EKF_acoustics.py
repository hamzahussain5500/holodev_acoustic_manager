"""
Single-run EKF fusion in HoloOcean: IMU + DVL + Depth + Acoustic Range.

- Applies optional ocean currents (vortex_field + apply_currents)
- Runs one configuration only (no multi-run comparisons)
- Prints concise metrics (RMSEs and final position error)
"""

import holoocean
import numpy as np

from kalman_utils import EKF, compute_rmse


# ============================================================
# TUNABLE PARAMETERS
# ============================================================

# Scenario & agent configuration
SCENARIO_NAME = "blue_rov"
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
SIM_DURATION_SEC = 180.0

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
ACOUSTIC_UPDATE_PERIOD_TICKS = 30  # ~1 second if ticks_per_sec=30

# ===== Currents control =====
USE_CURRENTS = True
VEHICLES_FOR_CURRENTS = [AUV_NAME]
MAP_DIMENSIONS = [100, 100, 35]
DRAW_CURRENT_FIELD_STEP = 1


# ============================================================
# CURRENT FIELD (copied from EKF.py logic)
# ============================================================

def vortex_field(location):
    """Vortex field with vertical component (from currents.py logic)."""
    x, y, z = location
    if z > 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    strength = 10.0
    r_squared = x**2 + y**2 + 1e-5  # avoid divide by zero
    dx = -y / r_squared * strength
    dy =  x / r_squared * strength
    dz = 0.2 * np.cos(0.1 * r_squared)

    return np.array([3*dx, 3*dy, 3*dz], dtype=float)


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
            arrow_size=0.5,
            spacing=3,
            lifetime=0,
        )

    for vehicle in VEHICLES_FOR_CURRENTS:
        if "LocationSensor" not in state[vehicle]:
            continue
        location = state[vehicle]["LocationSensor"]
        current_velocity = vortex_field(location)
        env.set_ocean_currents(vehicle, current_velocity)


# ============================================================
# SINGLE-RUN EKF (IMU + DVL + Depth + Acoustic_1)
# ============================================================

def run_ekf_acoustics():
    """Run one EKF fusion with IMU + DVL + Depth + Acoustic range."""
    true_positions = []
    est_positions = []
    true_velocities = []
    est_velocities = []
    pos_covariances = []
    times = []

    use_dvl_update = True
    use_depth_update = True
    use_acoustic_update_1 = True

    with holoocean.make(SCENARIO_NAME, show_viewport=False, frames_per_sec=False) as env:
        ticks_per_sec = getattr(env, "ticks_per_sec", DEFAULT_TICKS_PER_SEC)
        dt = 1.0 / float(ticks_per_sec)

        ekf = EKF(dt=dt)

        n_steps = int(SIM_DURATION_SEC * ticks_per_sec)
        prev_true_pos = None
        clock = 0

        for k in range(n_steps):
            clock += 1
            r_meas_1 = None
            beacon_1_pos = None

            # Step env (initial tick)
            state = env.tick()

            # Optionally send acoustic ping from USV_1 -> AUV
            if use_acoustic_update_1 and (k % ACOUSTIC_UPDATE_PERIOD_TICKS == 0):
                env.send_acoustic_message(
                    AUV_BEACON_ID,
                    USV_BEACON_ID_1,
                    "MSG_REQX",
                    "ping",
                )
                state = env.tick()

            # Read last acoustic message at AUV
            if use_acoustic_update_1:
                if "AcousticBeaconSensor" in state[AUV_NAME]:
                    acoustic_msg = state[AUV_NAME]["AcousticBeaconSensor"]
                    # Expect: ["MSG_RESPX", from_id, payload, phi, theta, r, d]
                    if acoustic_msg is not None and len(acoustic_msg) >= 6:
                        msg_type = acoustic_msg[0]
                        if msg_type == "MSG_RESPX" and acoustic_msg[1] == USV_BEACON_ID_1:
                            r_meas_1 = float(acoustic_msg[5])
                            # USV (beacon) position in world frame
                            if "PoseSensor" in state[USV_1_NAME]:
                                usv_1_pose = state[USV_1_NAME]["PoseSensor"]
                                beacon_1_pos = usv_1_pose[0:3, 3]
                            else:
                                beacon_1_pos = state[USV_1_NAME]["LocationSensor"]

            # Apply currents (optional)
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

            # DVL update (DVL reports world-frame velocity)
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

            # Depth update (z)
            if use_depth_update:
                z_meas = float(depth[0])
                z_vec = np.array([z_meas])
                H_depth = np.array([[0, 0, 1, 0, 0, 0]])
                R_depth = np.array([[DEPTH_STD**2]])
                ekf.update_linear(z_vec, H_depth, R_depth)

            # Acoustic range update (nonlinear EKF) if valid measurement
            if use_acoustic_update_1 and (r_meas_1 is not None) and (beacon_1_pos is not None):
                ekf.update_range(
                    r_meas_1,
                    beacon_1_pos,
                    ACOUSTIC_RANGE_STD**2,
                )

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


def main():
    print("\nRunning EKF fusion: IMU + DVL + Depth + Acoustic_1...")
    (times,
     true_pos,
     est_pos,
     true_vel,
     est_vel,
     Ppos) = run_ekf_acoustics()

    pos_rmse, pos_axis = compute_rmse(true_pos, est_pos)
    vel_rmse, vel_axis = compute_rmse(true_vel, est_vel)
    final_err = np.linalg.norm(est_pos[-1] - true_pos[-1])

    print("\n========== EKF FUSION RESULTS ==========")
    print(f"  Position RMSE total  : {pos_rmse:.3f} m")
    print(f"    axes (x,y,z)       : {pos_axis}")
    print(f"  Velocity RMSE total  : {vel_rmse:.3f} m/s")
    print(f"    axes (x,y,z)       : {vel_axis}")
    print(f"  Final position error : {final_err:.3f} m")
    print("=======================================\n")


if __name__ == "__main__":
    main()

