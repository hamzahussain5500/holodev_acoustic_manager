import holoocean
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from kalman_utils import EKF, compute_rmse
from uncertainty_utils import covariance_ellipse_points, covariance_ellipsoid_mesh



# EKF fusion script: IMU + DVL + Depth + Acoustic range (optional range logging).

# ===== Scenario & agent configuration (global assignments) =====
SCENARIO_NAME = "usv_auv"   # <-- set this to your scenario name
AUV_NAME      = "auv"        # <-- AUV agent name
USV_1_NAME    = "usv1"       # <-- surface beacons
USV_2_NAME    = "usv2"
USV_3_NAME    = "usv3"
USV_4_NAME    = "usv4"

# Acoustic beacon IDs (must match scenario config)
AUV_BEACON_ID        = 0
USV_BEACON_ID_1      = 1
USV_BEACON_ID_2      = 2
USV_BEACON_ID_3      = 3
USV_BEACON_ID_4      = 4



# Simulation timing
DEFAULT_TICKS_PER_SEC = 30
SIM_DURATION_SEC = 30.0

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
ACOUSTIC_RANGE_STD = 0.2626

# Acoustic ping interval (in ticks)
ACOUSTIC_UPDATE_PERIOD_TICKS = 30  # ~1 second if ticks_per_sec=30

# ===== Currents control =====
USE_CURRENTS = True
VEHICLES_FOR_CURRENTS = [AUV_NAME]
MAP_DIMENSIONS = [100, 100, 35]
DRAW_CURRENT_FIELD_STEP = 100


def safe_tick(env, last_state_container, retries=5, delay=0.01, show_warn=True):
    """Call env.tick() with retries on intermittent ValueError.

    last_state_container: single-element list used to store last known good state.
    Returns the new state on success or last known state on persistent failure.
    """
    try:
        s = env.tick()
        last_state_container[0] = s
        return s
    except ValueError as e:
        # retry a few times with short sleeps
        for _ in range(retries):
            try:
                time.sleep(delay)
                s = env.tick()
                last_state_container[0] = s
                return s
            except ValueError:
                continue
        if show_warn:
            print(f"[WARN] env.tick() ValueError after {retries+1} attempts: {e} -- using last known state")
        return last_state_container[0]
    except Exception as e:
        print(f"[ERROR] env.tick() raised unexpected exception: {e}")
        return last_state_container[0]


# ===== Beacon helper utilities =====
def get_beacon_ids(env):
    """Return list of beacon IDs, or empty list if unavailable."""
    try:
        ids = getattr(env, "beacons_id", None)
        return list(ids) if ids is not None else []
    except Exception:
        return []


def get_beacon_statuses(env):
    """Return list of beacon modem statuses aligned with env.beacons_id, or empty list."""
    try:
        statuses = getattr(env, "beacons_status", None)
        return list(statuses) if statuses is not None else []
    except Exception:
        return []


def get_status_by_beacon_id(env, beacon_id):
    """Return modem status for the given beacon id if available, else None."""
    ids = get_beacon_ids(env)
    statuses = get_beacon_statuses(env)
    if not ids or not statuses:
        return None
    try:
        idx = ids.index(beacon_id)
        return statuses[idx] if idx < len(statuses) else None
    except ValueError:
        return None

def build_mappings_from_globals(env):
    """Build mappings using global names and IDs.

    Returns (id_to_agent, USV_IDS, AUV_ID, TICKS_PER_SEC)
    """
    TICKS_PER_SEC = getattr(env, "ticks_per_sec", 30.0) if hasattr(env, "ticks_per_sec") else 30.0

    id_to_agent = {
        AUV_BEACON_ID: AUV_NAME,
        USV_BEACON_ID_1: USV_1_NAME,
        USV_BEACON_ID_2: USV_2_NAME,
        USV_BEACON_ID_3: USV_3_NAME,
        USV_BEACON_ID_4: USV_4_NAME,
    }

    #USV_IDS = [USV_BEACON_ID_1, USV_BEACON_ID_2, USV_BEACON_ID_3, USV_BEACON_ID_4]
    USV_IDS = [USV_BEACON_ID_1]

    AUV_ID = AUV_BEACON_ID

    return id_to_agent, USV_IDS, AUV_ID, TICKS_PER_SEC


def run_round_robin(env, id_to_agent, USV_IDS, AUV_ID, TICKS_PER_SEC, max_wait_seconds=2, verbose=False):
    """Run a single round-robin MSG_REQX -> MSG_RESPX pass to all USVs.

    This function performs the send->wait->collect cycle for each target and returns
    a dict of collected ranges {beacon_id: distance}
    """
    tick_count = 0
    last_state = [{}]
    ranges = {}
    max_wait_ticks = int(max_wait_seconds * TICKS_PER_SEC)

    # helper local reference for speed
    local_safe_tick = lambda: safe_tick(env, last_state)

    # One round across USVs
    for target_id in USV_IDS:
        # Skip if target beacon does not exist in environment
        if target_id not in get_beacon_ids(env):
            print(f"\n[SKIP] Beacon {target_id} not present in env.beacons_id; skipping.")
            continue
        target_name = id_to_agent.get(target_id, f"id={target_id}")
        if verbose:
            print(f"\nRanging to {target_name} (beacon {target_id})")

        # Show current modem statuses (verbose)
        if verbose:
            auv_stat = get_status_by_beacon_id(env, AUV_ID)
            tgt_stat = get_status_by_beacon_id(env, target_id)
            print(f"    Modem statuses before send: AUV={auv_stat}, Target={tgt_stat}")

        # Tick once before sending
        # state = local_safe_tick()  # replaced 
        state = env.tick()
        tick_count += 1

        # Wait for Idle statuses before sending (best-effort)
        waited_ticks = 0
        for _ in range(max_wait_ticks):
            auv_stat = get_status_by_beacon_id(env, AUV_ID)
            tgt_stat = get_status_by_beacon_id(env, target_id)
            if auv_stat in (None, "Idle") and tgt_stat in (None, "Idle"):
                print (f"    Modems Idle after {waited_ticks} ticks")
                break
            # state = local_safe_tick()  # replaced 
            state = env.tick()
            tick_count += 1
            waited_ticks += 1
            if verbose:
                print (f'waiting for Idle {tick_count}')
        if verbose and waited_ticks:
            print(f"    Waited {waited_ticks} ticks for Idle")

        # Send request from AUV to target
        send_tick = tick_count
        env.send_acoustic_message(AUV_ID, target_id, "MSG_REQX", "range_req")

        # Wait for response
        got_response = False
        for _ in range(max_wait_ticks):
            # state = local_safe_tick()  # replaced with direct env.tick() per request
            state = env.tick()
            tick_count += 1
            

            auv_sensors = state.get(AUV_NAME, {})
            if isinstance(auv_sensors, dict) and "AcousticBeaconSensor" in auv_sensors:
                msg = auv_sensors["AcousticBeaconSensor"]
                if msg is not None and isinstance(msg, (list, tuple)) and len(msg) >= 3:
                    msg_type = msg[0]
                    from_id = msg[1]

                    if msg_type == "MSG_RESPX" and from_id == target_id:
                        # parse expected format: ["MSG_RESPX", from_sensor, payload, phi, theta, dist, depth]
                        dist = None
                        if len(msg) >= 7:
                            _, _, payload, phi, theta, dist, depth = msg
                        else:
                            dist = None

                        if dist is not None:
                            if verbose:
                                print(f"    r_modem (from msg) = {dist:.2f} m with tick time {tick_count} ({tick_count / TICKS_PER_SEC:.2f} s)")
                            ranges[target_id] = float(dist)
                        else:
                            if verbose:
                                print("    [WARN] response arrived but distance not present in message")

                        got_response = True
                        if verbose:
                            print (f'took tick {tick_count} to get response')
                        print (f'ranges received after {tick_count - send_tick} ticks')
                        break

        if not got_response:
            print(f"  [WARN] No MSG_RESPX received from {target_name} within timeout")

    if verbose:
        print( "sending ranges took total ticks ", tick_count)
    return ranges


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
    """Apply currents to vehicles if USE_CURRENTS is True (AUV only)."""
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
        sensors = state.get(vehicle, {})
        if isinstance(sensors, dict) and "LocationSensor" in sensors:
            location = sensors["LocationSensor"]
            current_velocity = vortex_field(location)
            env.set_ocean_currents(vehicle, current_velocity)


def run_ekf_acoustics(target_names=None, verbose=True):
    """Run one EKF fusion with IMU + DVL + Depth + Acoustic range."""
    true_positions = []
    est_positions = []
    true_velocities = []
    est_velocities = []
    pos_covariances = []
    times = []

    use_dvl_update = True
    use_depth_update = True
    use_acoustic_update_1 = False


    with holoocean.make(SCENARIO_NAME, show_viewport=False, frames_per_sec=False, verbose=False) as env:

        # Minimal startup diagnostics (verbose)
        if verbose:
            print(f"Scenario: {SCENARIO_NAME}")
            print("Beacons:", get_beacon_ids(env))
            statuses = get_beacon_statuses(env)
            if statuses:
                print("Statuses:", statuses)
        if use_acoustic_update_1:

            # Build maps and defaults from global configuration
            id_to_agent, USV_IDS, AUV_ID, _TICKS_PER_SEC = build_mappings_from_globals(env)

            if not USV_IDS:
                print("[ERROR] No USV_IDS detected. Aborting.")
                return
            if AUV_ID is None:
                print("[ERROR] No AUV_ID detected. Aborting.")
                return

            if verbose:
                print("Using beacon ID", AUV_ID, "as AUV sender (agent=", id_to_agent.get(AUV_ID, 'unknown'), ")")

            # Choose targets: explicit target_ids take precedence; else use first N USVs
            actual = get_beacon_ids(env)
            # Build name->id for USVs only
            name_to_id = {name: bid for bid, name in id_to_agent.items() if bid in [
                USV_BEACON_ID_1, USV_BEACON_ID_2, USV_BEACON_ID_3, USV_BEACON_ID_4
            ]}

            requested_ids = []
            if target_names:
                for n in target_names:
                    if n in name_to_id:
                        requested_ids.append(name_to_id[n])
                    else:
                        print(f"[INFO] Unknown target name '{n}' — skipping")
            if requested_ids:
                requested_ids = list(dict.fromkeys(requested_ids))
                selected_usv_ids = [bid for bid in requested_ids if (not actual or bid in actual)]
                removed = [bid for bid in requested_ids if bid not in selected_usv_ids]
                if removed:
                    print(f"[INFO] Skipping non-existent beacons {removed}; using {selected_usv_ids}")
            else:
                selected_usv_ids = [USV_BEACON_ID_1] if (not actual or USV_BEACON_ID_1 in actual) else []
            print(f"Targeting {len(selected_usv_ids)} USV beacons: {selected_usv_ids}")


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

            # Optionally perform acoustic round-robin at the chosen interval
            ranges = {}
            if use_acoustic_update_1 and (k % ACOUSTIC_UPDATE_PERIOD_TICKS) == 0:
                ranges = run_round_robin(env, id_to_agent, selected_usv_ids, AUV_ID, ticks_per_sec, verbose=verbose)

            # Print collected ranges for inspection (verbose only)
            if verbose and ranges:
                ranges_list = [ranges.get(bid) for bid in selected_usv_ids]
                print(f"\n[RESULT] Ranges list ({len(ranges_list)} beacons, order={selected_usv_ids}):")
                print(ranges_list)

            # Read last acoustic message at AUV
            if use_acoustic_update_1:
                # Use the first selected target as the active beacon
                active_id = selected_usv_ids[0] if selected_usv_ids else None
                if active_id is not None and ranges:
                    r_val = ranges.get(active_id)
                    if r_val is not None:
                        r_meas_1 = float(r_val)
                        active_name = id_to_agent.get(active_id)
                        if active_name and active_name in state:
                            if "PoseSensor" in state[active_name]:
                                usv_pose = state[active_name]["PoseSensor"]
                                beacon_1_pos = usv_pose[0:3, 3]
                            else:
                                beacon_1_pos = state[active_name].get("LocationSensor")

            # Apply currents (optional)
            apply_currents(env, state, clock)

            # Read sensors
            imu   = state[AUV_NAME]["IMUSensor"]
            pose  = state[AUV_NAME]["PoseSensor"]
            loc   = state[AUV_NAME]["LocationSensor"]
            dvl   = state[AUV_NAME]["DVLSensor"]
            depth = state[AUV_NAME]["DepthSensor"]

            accel_meas = imu[0, :]
            accel_bias = imu[2, :] if imu.shape[0] >= 3 else np.zeros(3)
            accel_body = accel_meas - accel_bias

            T_ws = pose
            R_ws = T_ws[0:3, 0:3]
            true_pos = loc

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


'''def main(loop=True, num_usvs=4, target_ids=None, target_names=None, verbose=False):

    # Create the environment once, then use functions to operate on it
    with holoocean.make(SCENARIO_NAME, show_viewport=False, frames_per_sec=False) as env:

        # Minimal startup diagnostics (verbose)
        if verbose:
            print(f"Scenario: {SCENARIO_NAME}")
            print("Beacons:", get_beacon_ids(env))
            statuses = get_beacon_statuses(env)
            if statuses:
                print("Statuses:", statuses)

        # Build maps and defaults from global configuration
        id_to_agent, USV_IDS, AUV_ID, TICKS_PER_SEC = build_mappings_from_globals(env)

        if not USV_IDS:
            print("[ERROR] No USV_IDS detected. Aborting.")
            return
        if AUV_ID is None:
            print("[ERROR] No AUV_ID detected. Aborting.")
            return

        if verbose:
            print("Using beacon ID", AUV_ID, "as AUV sender (agent=", id_to_agent.get(AUV_ID, 'unknown'), ")")

        # Choose targets: explicit target_ids take precedence; else use first N USVs
        actual = get_beacon_ids(env)
        # Build name->id for USVs only
        name_to_id = {name: bid for bid, name in id_to_agent.items() if bid in [
            USV_BEACON_ID_1, USV_BEACON_ID_2, USV_BEACON_ID_3, USV_BEACON_ID_4
        ]}

        requested_ids = []
        # Names first (preserve provided order)
        if target_names:
            for n in target_names:
                if n in name_to_id:
                    requested_ids.append(name_to_id[n])
                else:
                    print(f"[INFO] Unknown target name '{n}' — skipping")
        # Then explicit IDs
        if target_ids:
            requested_ids.extend(target_ids)

        if requested_ids:
            # Deduplicate preserving order
            requested_ids = list(dict.fromkeys(requested_ids))
            selected_usv_ids = [bid for bid in requested_ids if (not actual or bid in actual)]
            removed = [bid for bid in requested_ids if bid not in selected_usv_ids]
            if removed:
                print(f"[INFO] Skipping non-existent beacons {removed}; using {selected_usv_ids}")
        else:
            selected_usv_ids = sorted(USV_IDS)[:max(1, min(int(num_usvs), 4))]
            if actual:
                before = list(selected_usv_ids)
                selected_usv_ids = [bid for bid in selected_usv_ids if bid in actual]
                removed = [bid for bid in before if bid not in selected_usv_ids]
                if removed:
                    print(f"[INFO] Skipping non-existent beacons {removed}; using {selected_usv_ids}")
        print(f"Targeting {len(selected_usv_ids)} USV beacons: {selected_usv_ids}")

        try:
            # Run at least once; if loop=True, run repeatedly until Ctrl-C

            clock = 0
            while True:
                # Step once to get current AUV location and apply currents
                last_state = [{}]
                state = safe_tick(env, last_state)
                clock += 1
                apply_currents(env, state, clock)

                # Now run the round-robin ranging (it will advance the sim internally)
                ranges = run_round_robin(env, id_to_agent, selected_usv_ids, AUV_ID, TICKS_PER_SEC, verbose=verbose)

                # Print collected ranges for inspection
                if ranges:
                    # Build and print a dynamic list aligned to the selected USV beacons order
                    ranges_list = [ranges.get(bid) for bid in selected_usv_ids]
                    print(f"\n[RESULT] Ranges list ({len(ranges_list)} beacons, order={selected_usv_ids}):")
                    print(ranges_list)

                if not loop:
                    break

                # small pause to avoid tight loop; user can Ctrl-C to quit
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl-C). Exiting.")'''


'''if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect acoustic ranges from USV beacons in HoloOcean")
    parser.add_argument("--loop", action="store_true", help="Run round-robin continuously until Ctrl-C")
    parser.add_argument("--num-usvs", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Number of USVs to range (1..4)")
    parser.add_argument("--target-id", type=int, action="append",
                        help="Beacon ID to range from; repeat to specify multiple (overrides --num-usvs)")
    parser.add_argument("--target-name", type=str, action="append",
                        help="Target agent name (e.g., usv1); repeat to specify multiple")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    main(loop=args.loop, num_usvs=args.num_usvs, target_ids=args.target_id, target_names=args.target_name, verbose=args.verbose)

'''


def main(target_names=None, verbose=False):
    print("\nRunning EKF fusion: IMU + DVL + Depth + Acoustic_1...")
    (times,
     true_pos,
     est_pos,
     true_vel,
     est_vel,

    Ppos) = run_ekf_acoustics(target_names=target_names, verbose=verbose)

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

    # ===== Plots (similar style to EKF.py) =====
    chi2_2d_95 = 5.991
    chi2_3d_95 = 7.815

    # XY trajectory + covariance ellipses
    plt.figure()
    plt.plot(true_pos[:, 0], true_pos[:, 1], label="true", color="k", linewidth=2)
    plt.plot(est_pos[:, 0], est_pos[:, 1], "--", label="EKF", color="purple")
    num_ellipses = 4
    idxs = np.linspace(0, len(est_pos) - 1, num_ellipses, dtype=int)
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
    plt.plot(times, true_pos[:, 2], label="true z", color="k")
    plt.plot(times, est_pos[:, 2], "--", label="EKF z", color="purple")
    plt.xlabel("time [s]")
    plt.ylabel("z [m]")
    plt.title("Z position vs time")
    plt.legend()

    # Position error vs time
    pos_err = np.linalg.norm(est_pos - true_pos, axis=1)
    plt.figure()
    plt.plot(times, pos_err, label="EKF position error", color="purple")
    plt.xlabel("time [s]")
    plt.ylabel("||position error|| [m]")
    plt.title("Position error over time")
    plt.legend()

    # 3D uncertainty bubble at final step
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    idx = -1
    mean = est_pos[idx]
    P_final = Ppos[idx]
    X, Y, Z = covariance_ellipsoid_mesh(P_final, mean,
                                        chi2_val=chi2_3d_95,
                                        num_u=25, num_v=25)
    ax.plot_surface(X, Y, Z, alpha=0.3, color="purple", edgecolor="none")
    ax.scatter(true_pos[idx, 0], true_pos[idx, 1], true_pos[idx, 2], color="k", label="true pos")
    ax.scatter(mean[0], mean[1], mean[2], color="purple", label="EKF mean")
    ax.set_title("EKF 95% uncertainty bubble")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKF fusion with optional acoustic ranging")
    parser.add_argument("--target-name", type=str, action="append",
                        help="Target agent name (e.g., usv1); repeat to specify multiple")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    main(target_names=args.target_name, verbose=args.verbose)

# End of script
