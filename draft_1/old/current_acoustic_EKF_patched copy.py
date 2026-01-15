import holoocean
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from kalman_utils import EKF, compute_rmse
from uncertainty_utils import (
    covariance_ellipse_points,
    covariance_ellipsoid_mesh,
    axis_uncertainty_bounds,
)
from validation_metrics import (
    calculate_nees,
    nees_consistency_test,
    calculate_nis,
    nis_consistency_test,
)


# Sensor rates (Hz) - must match your scenario JSON
IMU_HZ   = 100   # same as ticks_per_sec
DVL_HZ   = 20
DEPTH_HZ = 50


# EKF fusion script: IMU + DVL + Depth + Acoustic range (optional range logging).

# ===== Scenario & agent configuration (global assignments) =====
SCENARIO_NAME = "usv_auv_100_imu"   # <-- set this to your scenario name
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
DEFAULT_TICKS_PER_SEC = 100
SIM_DURATION_SEC = 600.0

# Gravity in WORLD frame
GRAVITY_WORLD = np.array([0.0, 0.0, 9.81])

# EKF process noise (Q)
Q_POS_STD = 0.0
Q_VEL_STD = 0.085

# EKF initial covariance (P)
P_POS_STD_INIT = 1.0
P_VEL_STD_INIT = 1.0

# DVL measurement noise (R)
DVL_VEL_STD = 0.24

# Depth measurement noise (R_depth)
DEPTH_STD = 0.03

# Acoustic range noise (R_range)
ACOUSTIC_RANGE_STD = 0.1

# Acoustic ping interval (in ticks)
ACOUSTIC_UPDATE_PERIOD_TICKS = 100  # ~1 second if ticks_per_sec=100

# ===== Currents control =====
USE_CURRENTS = True
VEHICLES_FOR_CURRENTS = [AUV_NAME]
MAP_DIMENSIONS = [100, 100, 25]
DRAW_CURRENT_FIELD_STEP = 100

# ===== Waypoint navigation (6-DOF targets) =====
# The vehicle tracks XY waypoints at a fixed depth. Yaw is steered to face the
# incoming current (upstream) when a measurable current exists; otherwise it
# faces the active waypoint.
TARGET_Z = -15.0
TARGET_YAW = 0.0
POS_TOL = 2.0


def lawnmower_waypoints(start_xy=(-35.0, -35.0),
                        xmin=-35.0, xmax=35.0,
                        ymin=-35.0, ymax=35.0,
                        spacing=12.5):
    start_xy = np.array(start_xy, dtype=float)
    ys = np.arange(ymin, ymax + 1e-9, spacing)

    wps = [start_xy, np.array([xmin, ys[0]])]  # start + entry

    # build alternating sweeps with vertical step between lanes
    x_end = xmin
    for i, y in enumerate(ys):
        x_target = xmax if (i % 2 == 0) else xmin
        wps.append(np.array([x_target, y]))
        x_end = x_target
        if i < len(ys) - 1:
            wps.append(np.array([x_end, ys[i+1]]))  # step up at the end

    return np.vstack(wps)

WAYPOINTS_XY = lawnmower_waypoints(spacing=12.5)

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
    TICKS_PER_SEC = getattr(env, "ticks_per_sec", 100.0) if hasattr(env, "ticks_per_sec") else 100.0

    id_to_agent = {
        AUV_BEACON_ID: AUV_NAME,
        USV_BEACON_ID_1: USV_1_NAME,
        USV_BEACON_ID_2: USV_2_NAME,
        USV_BEACON_ID_3: USV_3_NAME,
        USV_BEACON_ID_4: USV_4_NAME,
    }
    USV_IDS = [USV_BEACON_ID_1, USV_BEACON_ID_2, USV_BEACON_ID_3, USV_BEACON_ID_4]
    AUV_ID = AUV_BEACON_ID

    return id_to_agent, USV_IDS, AUV_ID, TICKS_PER_SEC


class AcousticRoundRobin:
    """Non-blocking acoustic ranging scheduler for HoloOcean AcousticBeaconSensor.

    Why this exists:
      - Avoids calling env.tick() inside the ranging routine (prevents time desync).
      - Supports multiple beacons by round-robining targets across outer EKF ticks.
      - Enforces "one pending request at a time" which matches typical modem constraints.

    Usage:
      mgr = AcousticRoundRobin(auv_id, target_ids, ticks_per_sec, period_ticks, timeout_ticks)
      each outer loop tick:
        mgr.maybe_send(env, sim_tick)
        responses = mgr.poll(state, sim_tick, auv_name)
        for (from_id, r, d) in responses: ekf.update_range(...)
    """

    def __init__(self, auv_id, target_ids, ticks_per_sec, period_ticks=100, timeout_ticks=600):
        self.auv_id = int(auv_id)
        self.target_ids = list(target_ids)
        self.ticks_per_sec = float(ticks_per_sec)
        self.period_ticks = int(period_ticks)
        self.timeout_ticks = int(timeout_ticks)

        self.pending = None   # {"target_id":..., "send_tick":...}
        self.rr_idx = 0

    def _status(self, env, beacon_id):
        return get_status_by_beacon_id(env, beacon_id)

    def _next_target(self):
        if not self.target_ids:
            return None
        tid = self.target_ids[self.rr_idx]
        self.rr_idx = (self.rr_idx + 1) % len(self.target_ids)
        return tid

    def maybe_send(self, env, sim_tick, verbose=False):
        """Send MSG_REQX to next target if:
          - no request pending
          - (sim_tick % period_ticks) == 0
          - both AUV and target modems are Idle
        """
        if not self.target_ids:
            return
        if self.pending is not None:
            return
        if self.period_ticks > 0 and (sim_tick % self.period_ticks) != 0:
            return

        target_id = self._next_target()
        if target_id is None:
            return

        auv_stat = self._status(env, self.auv_id)
        tgt_stat = self._status(env, target_id)

        if auv_stat in (None, "Idle") and tgt_stat in (None, "Idle"):
            env.send_acoustic_message(self.auv_id, target_id, "MSG_REQX", "range_req")
            self.pending = {"target_id": int(target_id), "send_tick": int(sim_tick)}
            if verbose:
                print(f"[ACOUSTIC] Sent MSG_REQX auv({self.auv_id})->target({target_id}) at tick={sim_tick}")
        else:
            if verbose:
                print(f"[ACOUSTIC] Skip send (not idle): auv={auv_stat}, target={tgt_stat}")

    def poll(self, state, sim_tick, auv_name, verbose=False):
        """Poll AUV AcousticBeaconSensor once (non-blocking).

        Returns:
          list of (from_id, range_m, depth_m) for any MSG_RESPX received this tick.
        """
        results = []

        # Timeout a pending request (prevents getting stuck forever)
        if self.pending is not None and (sim_tick - self.pending["send_tick"] > self.timeout_ticks):
            if verbose:
                print(f"[ACOUSTIC] Timeout waiting for target({self.pending['target_id']}) at tick={sim_tick}")
            self.pending = None

        auv_sensors = state.get(auv_name, {})
        msg = auv_sensors.get("AcousticBeaconSensor", None)

        if msg and isinstance(msg, (list, tuple)) and len(msg) >= 7:
            msg_type = msg[0]
            from_id = int(msg[1])

            if msg_type == "MSG_RESPX":
                # ["MSG_RESPX", from_sensor, payload, phi, theta, r, d]
                dist = float(msg[5])
                depth = float(msg[6])
                results.append((from_id, dist, depth))

                if verbose:
                    print(f"[ACOUSTIC] Got MSG_RESPX from {from_id} at tick={sim_tick}: r={dist:.2f} m, d={depth:.2f} m")

                # Clear pending if this matches the expected target
                if self.pending is not None and from_id == int(self.pending["target_id"]):
                    self.pending = None

        return results



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

    strength = 1.0
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
    """Run one EKF fusion with IMU + DVL + Depth + Acoustic ranges (multi-target ready).

    Key properties of this implementation:
      - Exactly ONE env.tick() per EKF iteration (no hidden ticks inside acoustic logic).
      - Acoustic ranging handled asynchronously via AcousticRoundRobin state machine.
      - Supports fusing multiple ranges (sequential updates) as they arrive.
    """
    true_positions = []
    est_positions = []
    true_velocities = []
    est_velocities = []
    pos_covariances = []
    full_covariances = []
    nis_logs = {
        "dvl": {"t": [], "values": [], "dof": 3},
        "depth": {"t": [], "values": [], "dof": 1},
        "acoustic": {"t": [], "values": [], "dof": 1},
    }
    times = []

    use_dvl_update = True
    use_depth_update = True
    use_acoustic_updates = True  # enable acoustic fusion (multi-range ready)

    with holoocean.make(SCENARIO_NAME, show_viewport=False, frames_per_sec=False, verbose=False) as env:

        ticks_per_sec = getattr(env, "ticks_per_sec", DEFAULT_TICKS_PER_SEC)
        dt = 1.0 / float(ticks_per_sec)

        def hz_to_period_ticks(hz, ticks_per_sec, name):
            if hz <= 0:
                return None
            raw = ticks_per_sec / float(hz)
            period = int(round(raw))
            if abs(raw - period) > 1e-6:
                print(f"[WARN] {name} rate {hz} is not a factor of ticks_per_sec {ticks_per_sec}. "
                    f"Using period_ticks={period} (effective Hz={ticks_per_sec/period:.3f}).")
            return max(1, period)

        dvl_period_ticks   = hz_to_period_ticks(DVL_HZ, ticks_per_sec, "DVLSensor")
        depth_period_ticks = hz_to_period_ticks(DEPTH_HZ, ticks_per_sec, "DepthSensor")


        # Build maps and defaults from global configuration
        id_to_agent, USV_IDS, AUV_ID, _TICKS_PER_SEC = build_mappings_from_globals(env)

        actual_ids = get_beacon_ids(env)
        if verbose:
            print(f"Scenario: {SCENARIO_NAME}")
            print("Beacons:", actual_ids)
            statuses = get_beacon_statuses(env)
            if statuses:
                print("Statuses:", statuses)

        # Build name->id for USVs only
        name_to_id = {
            USV_1_NAME: USV_BEACON_ID_1,
            USV_2_NAME: USV_BEACON_ID_2,
            USV_3_NAME: USV_BEACON_ID_3,
            USV_4_NAME: USV_BEACON_ID_4,
        }

        # Choose which USVs to range to (CLI uses --target-name)
        if target_names:
            requested = []
            for n in target_names:
                if n in name_to_id:
                    requested.append(name_to_id[n])
                else:
                    print(f"[INFO] Unknown target name '{n}' — skipping")
            # dedupe preserve order
            requested = list(dict.fromkeys(requested))
            selected_usv_ids = requested
        else:
            selected_usv_ids = list(USV_IDS)

        # Filter to beacons actually present
        if actual_ids:
            selected_usv_ids = [bid for bid in selected_usv_ids if bid in actual_ids]

        if use_acoustic_updates:
            if AUV_ID is None or AUV_ID not in (actual_ids if actual_ids else [AUV_ID]):
                raise RuntimeError(f"AUV beacon id {AUV_ID} not present / invalid.")
            if not selected_usv_ids:
                print("[WARN] No valid USV beacons selected. Acoustic fusion will be disabled.")
                use_acoustic_updates = False
            else:
                if verbose:
                    print(f"AUV beacon: {AUV_ID} ({id_to_agent.get(AUV_ID, AUV_NAME)})")
                    print(f"Targeting {len(selected_usv_ids)} USV beacons: {selected_usv_ids}")

        # Acoustic scheduler (non-blocking)
        acoustic_mgr = None
        if use_acoustic_updates:
            acoustic_mgr = AcousticRoundRobin(
                auv_id=AUV_ID,
                target_ids=selected_usv_ids,
                ticks_per_sec=ticks_per_sec,
                period_ticks=ACOUSTIC_UPDATE_PERIOD_TICKS,   # total ping rate, not per-target
                timeout_ticks=int(3 * ticks_per_sec),        # 3s timeout
            )

        # EKF uses IMU acceleration as an input. With the updated kalman_utils.EKF,
        # q_vel_std is interpreted as accel uncertainty std (sigma_a, m/s^2)
        # and q_pos_std is an optional position random-walk (m/sqrt(s)).
        ekf = EKF(
            dt=dt,
            q_pos_std=Q_POS_STD,
            q_vel_std=Q_VEL_STD,
            p_pos_std_init=P_POS_STD_INIT,
            p_vel_std_init=P_VEL_STD_INIT,
        )

        n_steps = int(SIM_DURATION_SEC * ticks_per_sec)
        prev_true_pos = None
        clock = 0
        sim_tick = 0  # counts env.step() calls (true simulation time base)
        idx = 0
        yaw_cmd = TARGET_YAW

        for xy in WAYPOINTS_XY:
            env.draw_point([xy[0], xy[1], TARGET_Z], color=[0, 255, 0], thickness=20.0, lifetime=0)


        last_dvl = None
        last_depth = None


        for k in range(n_steps):
            #print (f"--- EKF step {k+1}/{n_steps} (sim_tick={sim_tick}) (duration={k/ticks_per_sec} ---")

            clock += 1

            x_wp, y_wp = WAYPOINTS_XY[idx]
            target_6d = np.array([x_wp, y_wp, TARGET_Z, 0.0, 0.0, yaw_cmd], dtype=float)

            # --- Step env (exactly once per EKF step) ---
            state = env.step(target_6d)
            sim_tick += 1
            t_current = sim_tick * dt

            # Apply currents (optional)
            apply_currents(env, state, clock)

            # --- Read sensors ---
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

            # --- IMU prediction ---
            a_world_raw = R_ws @ accel_body
            a_world = a_world_raw - GRAVITY_WORLD
            #print (f'accel_body: {accel_body}, a_world_raw: {a_world_raw}, a_world: {a_world}')

            if k == 0:
                ekf.x[0:3] = true_pos.copy()
                ekf.x[3:6] = true_vel.copy()

            ekf.predict(a_world)

            # --- DVL update (velocity) ---
            if use_dvl_update and (dvl_period_ticks is None or (sim_tick % dvl_period_ticks) == 0):
                prior_x = ekf.x.copy()
                prior_P = ekf.P.copy()
                v_body = dvl[0:3]
                #v_world_meas = R_ws.T @ v_body
                v_world_meas = v_body

                # Only update if this is actually a new measurement (not held/repeated)
                if last_dvl is None or not np.allclose(v_world_meas, last_dvl, atol=1e-3):
                    H_dvl = np.array([
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ])
                    R_dvl = np.diag([DVL_VEL_STD**2]*3)

                    y_dvl = v_world_meas.reshape(3, 1) - H_dvl @ prior_x.reshape(6, 1)
                    S_dvl = H_dvl @ prior_P @ H_dvl.T + R_dvl
                    try:
                        nis_val = float(y_dvl.T @ np.linalg.solve(S_dvl, y_dvl))
                    except np.linalg.LinAlgError:
                        nis_val = float(y_dvl.T @ np.linalg.pinv(S_dvl) @ y_dvl)
                    nis_logs["dvl"]["t"].append(t_current)
                    nis_logs["dvl"]["values"].append(nis_val)

                    ekf.update_linear(v_world_meas, H_dvl, R_dvl)
                    last_dvl = v_world_meas.copy()



            # --- Depth update (z) ---
            if use_depth_update and (depth_period_ticks is None or (sim_tick % depth_period_ticks) == 0):
                prior_x = ekf.x.copy()
                prior_P = ekf.P.copy()
                z_meas = float(depth[0])

                # Only update if new (depth often holds last value between true updates)
                if last_depth is None or abs(z_meas - last_depth) > 1e-3:
                    z_vec = np.array([z_meas])
                    H_depth = np.array([[0, 0, 1, 0, 0, 0]])
                    R_depth = np.array([[DEPTH_STD**2]])

                    y_depth = z_vec.reshape(1, 1) - H_depth @ prior_x.reshape(6, 1)
                    S_depth = H_depth @ prior_P @ H_depth.T + R_depth
                    try:
                        nis_val = float(y_depth.T @ np.linalg.solve(S_depth, y_depth))
                    except np.linalg.LinAlgError:
                        nis_val = float(y_depth.T @ np.linalg.pinv(S_depth) @ y_depth)
                    nis_logs["depth"]["t"].append(t_current)
                    nis_logs["depth"]["values"].append(nis_val)

                    ekf.update_linear(z_vec, H_depth, R_depth)
                    last_depth = z_meas



            # --- Acoustic scheduling + polling (non-blocking) ---
            if use_acoustic_updates and acoustic_mgr is not None:
                acoustic_mgr.maybe_send(env, sim_tick, verbose=verbose)

                responses = acoustic_mgr.poll(state, sim_tick, auv_name=AUV_NAME, verbose=verbose)

                # Fuse ALL ranges received on this tick (sequential EKF updates)
                for from_id, dist_m, depth_m in responses:
                    if from_id not in selected_usv_ids:
                        continue

                    usv_name = id_to_agent.get(from_id, None)
                    if usv_name is None or usv_name not in state:
                        continue

                    # Use USV pose at CURRENT tick (time aligned)
                    if "PoseSensor" in state[usv_name]:
                        usv_pose = state[usv_name]["PoseSensor"]
                        beacon_pos = usv_pose[0:3, 3]
                    else:
                        beacon_pos = state[usv_name].get("LocationSensor", None)

                    if beacon_pos is None:
                        continue

                    prior_x = ekf.x.copy()
                    prior_P = ekf.P.copy()
                    px, py, pz = prior_x[0:3]
                    bx, by, bz = beacon_pos
                    dx = px - bx
                    dy = py - by
                    dz = pz - bz
                    dist_pred = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9

                    H_range = np.zeros((1, 6))
                    H_range[0, 0] = dx / dist_pred
                    H_range[0, 1] = dy / dist_pred
                    H_range[0, 2] = dz / dist_pred
                    R_range = np.array([[ACOUSTIC_RANGE_STD**2]])
                    y_range = np.array([[dist_m - dist_pred]])
                    S_range = H_range @ prior_P @ H_range.T + R_range
                    try:
                        nis_val = float(y_range.T @ np.linalg.solve(S_range, y_range))
                    except np.linalg.LinAlgError:
                        nis_val = float(y_range.T @ np.linalg.pinv(S_range) @ y_range)
                    nis_logs["acoustic"]["t"].append(t_current)
                    nis_logs["acoustic"]["values"].append(nis_val)

                    ekf.update_range(dist_m, beacon_pos, ACOUSTIC_RANGE_STD**2)

            # --- Heading control: face currents or waypoint ---
            current_vec = vortex_field(loc)
            if np.linalg.norm(current_vec[0:2]) > 1e-4:
                yaw_cmd = np.arctan2(-current_vec[1], -current_vec[0])
            else:
                to_wp = np.array([x_wp - loc[0], y_wp - loc[1]])
                if np.linalg.norm(to_wp) > 1e-4:
                    yaw_cmd = np.arctan2(to_wp[1], to_wp[0])

            # --- Logging ---
            times.append(t_current)
            true_positions.append(true_pos.copy())
            est_positions.append(ekf.x[0:3].copy())
            true_velocities.append(true_vel.copy())
            est_velocities.append(ekf.x[3:6].copy())
            pos_covariances.append(ekf.P[0:3, 0:3].copy())
            full_covariances.append(ekf.P.copy())

            # --- Waypoint switching ---
            dist_to_wp = np.linalg.norm(loc[0:2] - np.array([x_wp, y_wp]))
            depth_err = abs(loc[2] - TARGET_Z)
            if dist_to_wp <= POS_TOL and depth_err <= POS_TOL:
                if idx >= len(WAYPOINTS_XY) - 1:
                    print("Final waypoint reached. Ending simulation.")
                    break
                idx += 1

    times = np.array(times)
    true_positions = np.array(true_positions)
    est_positions = np.array(est_positions)
    true_velocities = np.array(true_velocities)
    est_velocities = np.array(est_velocities)
    pos_covariances = np.array(pos_covariances)

    return (
        times,
        true_positions,
        est_positions,
        true_velocities,
        est_velocities,
        pos_covariances,
        full_covariances,
        nis_logs,
    )


def main(target_names=None, verbose=False):
    print("\nRunning EKF fusion: IMU + DVL + Depth + Acoustic_1...")
    (times,
     true_pos,
     est_pos,
     true_vel,
     est_vel,
        Ppos,
        Pfull,
        nis_logs) = run_ekf_acoustics(target_names=target_names, verbose=verbose)

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

    # === Consistency metrics ===
    state_err = np.hstack((est_pos - true_pos, est_vel - true_vel))
    nees_full = calculate_nees(state_err, Pfull)
    nees_pos = calculate_nees(state_err, Pfull, indices=[0, 1, 2])
    nees_full_test = nees_consistency_test(nees_full, dof=6)
    nees_pos_test = nees_consistency_test(nees_pos, dof=3)

    print("NEES (full state 6D): avg={avg:.3f}, expected=6, ci=[{lo:.3f},{hi:.3f}], inside={pct:.1f}%".format(
        avg=nees_full_test["avg_nees"], lo=nees_full_test["lower_bound"], hi=nees_full_test["upper_bound"],
        pct=nees_full_test["percent_inside_bounds"]))
    print("NEES (position 3D):  avg={avg:.3f}, expected=3, ci=[{lo:.3f},{hi:.3f}], inside={pct:.1f}%".format(
        avg=nees_pos_test["avg_nees"], lo=nees_pos_test["lower_bound"], hi=nees_pos_test["upper_bound"],
        pct=nees_pos_test["percent_inside_bounds"]))

    nis_results = {}
    for key in ("dvl", "depth", "acoustic"):
        vals = np.array(nis_logs[key]["values"])
        if vals.size == 0:
            nis_results[key] = None
            continue
        nis_results[key] = nis_consistency_test(vals, dof=nis_logs[key]["dof"])

    for key in ("dvl", "depth", "acoustic"):
        res = nis_results.get(key)
        if res is None:
            print(f"NIS ({key}): no samples")
        else:
            print(f"NIS ({key}): avg={res['avg_nis']:.3f}, expected={res['expected_nis']}, ci=[{res['lower_bound']:.3f},{res['upper_bound']:.3f}], inside={res['percent_inside_bounds']:.1f}%")

    # ===== Plots (similar style to EKF.py) =====
    chi2_1d_95 = 3.841
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

    # NEES time series with per-sample bounds
    nees_lower_full, nees_upper_full = nees_full_test["per_sample_bounds"]
    plt.figure()
    plt.plot(times, nees_full, label="NEES (6D)", color="purple")
    plt.axhline(nees_lower_full, color="gray", linestyle="--", label="95% lower")
    plt.axhline(nees_upper_full, color="gray", linestyle="--", label="95% upper")
    plt.xlabel("time [s]")
    plt.ylabel("NEES")
    plt.title("NEES consistency (full state)")
    plt.legend()

    nees_lower_pos, nees_upper_pos = nees_pos_test["per_sample_bounds"]
    plt.figure()
    plt.plot(times, nees_pos, label="NEES (pos)", color="teal")
    plt.axhline(nees_lower_pos, color="gray", linestyle="--", label="95% lower")
    plt.axhline(nees_upper_pos, color="gray", linestyle="--", label="95% upper")
    plt.xlabel("time [s]")
    plt.ylabel("NEES")
    plt.title("NEES consistency (position only)")
    plt.legend()

    # NIS per sensor
    for key, color, label in [("dvl", "red", "DVL"), ("depth", "blue", "Depth"), ("acoustic", "green", "Acoustic")]:
        vals = np.array(nis_logs[key]["values"])
        tvals = np.array(nis_logs[key]["t"])
        if vals.size == 0:
            continue
        res = nis_results[key]
        lower, upper = res["per_sample_bounds"]
        plt.figure()
        plt.plot(tvals, vals, label=f"NIS {label}", color=color)
        plt.axhline(lower, color="gray", linestyle="--", label="95% lower")
        plt.axhline(upper, color="gray", linestyle="--", label="95% upper")
        plt.xlabel("time [s]")
        plt.ylabel("NIS")
        plt.title(f"NIS consistency ({label})")
        plt.legend()

    # Per-axis positional uncertainty over time (1D 95%)
    diag_cov = np.diagonal(Ppos, axis1=1, axis2=2)
    half_widths_t = np.sqrt(diag_cov * chi2_1d_95)
    plt.figure()
    plt.plot(times, half_widths_t[:, 0], label="x 95% half-width", color="red")
    plt.plot(times, half_widths_t[:, 1], label="y 95% half-width", color="green")
    plt.plot(times, half_widths_t[:, 2], label="z 95% half-width", color="blue")
    plt.xlabel("time [s]")
    plt.ylabel("pos half-width [m]")
    plt.title("Position uncertainty over time (95% per-axis)")
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

    # Final-axis positional uncertainty (1D 95%)
    centers, half_widths, labels = axis_uncertainty_bounds(P_final, mean, chi2_val=chi2_1d_95)
    plt.figure()
    plt.errorbar(labels, centers, yerr=half_widths, fmt="o", color="purple", ecolor="purple", capsize=6, linewidth=2)
    plt.plot(labels, true_pos[idx, :3], "x", color="k", label="true pos")
    plt.ylabel("position [m]")
    plt.title("Final position uncertainty (95% per-axis)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKF fusion with optional acoustic ranging")
    parser.add_argument("--target-name", type=str, action="append",
                        help="Target agent name (e.g., usv1); repeat to specify multiple")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    main(target_names=args.target_name, verbose=args.verbose)

# End of script
