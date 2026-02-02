import holoocean
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from kalman_utils import EKF, compute_rmse
from trajectory import build_trajectory, lawnmower_waypoints, spiral_waypoints, concentric_circles_waypoints, figure_eight_waypoints
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
# Default to shorter runs to keep Monte Carlo batches fast; override via config.
SIM_DURATION_SEC = 300

# Gravity in WORLD frame
GRAVITY_WORLD = np.array([0.0, 0.0, 9.81])

# EKF process noise (Q
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
USE_CURRENTS = False
VEHICLES_FOR_CURRENTS = [AUV_NAME]
MAP_DIMENSIONS = [100, 100, 25]
DRAW_CURRENT_FIELD_STEP = 100

# ===== Waypoint navigation (6-DOF targets) =====
# The vehicle tracks XY waypoints at a fixed depth with a fixed yaw target.
TARGET_Z = -150.0
TARGET_YAW_DEG = 60.0
POS_TOL = 1.0


def path_length(points: np.ndarray) -> float:
    """Return cumulative Euclidean distance along a polyline of points."""
    if points is None or len(points) < 2:
        return 0.0
    diffs = np.diff(np.asarray(points), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _reset_acoustic_sensor_state():
    """Clear class-level state that can persist across HoloOcean env instances."""
    try:
        from holoocean.sensors import AcousticBeaconSensor

        if hasattr(AcousticBeaconSensor, "instances"):
            AcousticBeaconSensor.instances = {}
        if hasattr(AcousticBeaconSensor, "pending_responses"):
            AcousticBeaconSensor.pending_responses = {}
        if hasattr(AcousticBeaconSensor, "sending_to"):
            AcousticBeaconSensor.sending_to = {}
    except Exception:
        pass

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


def apply_currents(env, state, clock, enabled=True):
    """Apply currents to vehicles if USE_CURRENTS is True (AUV only)."""
    if not enabled:
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



def run_ekf_acoustics(
    target_names=None,
    verbose=False,
    sim_config: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    debug_steps: bool = False,
    random_streams: Optional[Any] = None,
):
    """Run one EKF fusion with IMU + DVL + Depth + Acoustic ranges (multi-target ready).

    Key properties of this implementation:
      - Exactly ONE env.tick() per EKF iteration (no hidden ticks inside acoustic logic).
      - Acoustic ranging handled asynchronously via AcousticRoundRobin state machine.
      - Supports fusing multiple ranges (sequential updates) as they arrive.
    """
    cfg = {
        "use_currents": False,
        "use_dvl_update": True,
        "use_depth_update": True,
        "use_acoustic_updates": True,
        "use_all_acoustic": True,
        "show_viewport": False,
        "acoustic_uncertainty_trace_thresh": None,
        "acoustic_period_ticks": ACOUSTIC_UPDATE_PERIOD_TICKS,
        "dvl_measurement_std": DVL_VEL_STD,
        "depth_measurement_std": DEPTH_STD,
        "acoustic_measurement_std": ACOUSTIC_RANGE_STD,
        "dvl_extra_std": 0.0,
        "depth_extra_std": 0.0,
        "range_extra_std": 0.0,
        "imu_accel_extra_std": 0.0,
        "imu_bias_rw_std": 0.0,
        "q_pos_std": Q_POS_STD,
        "q_vel_std": Q_VEL_STD,
        "p_pos_std_init": P_POS_STD_INIT,
        "p_vel_std_init": P_VEL_STD_INIT,
        "trajectory": "lawnmower",
        "waypoint_spacing": 12.5,
        # Spiral tuning (start at spawn: 200, -200, -5; grow to 50 m radius; descend to -90 m)
        "spiral_center": (200.0, -200.0),
        "spiral_min_radius": 20.0,
        "spiral_max_radius": 50.0,
        "spiral_turns": 6,
        "spiral_points_per_rev": 250,
        "spiral_z_start": -5.0,
        "spiral_z_end": TARGET_Z,
        # Concentric circles tuning
        "concentric_radii": (10.0, 20.0, 30.0),
        "concentric_points_per_circle": 400,
        # Figure-eight tuning
        "figure8_scale": 20.0,
        "figure8_turns": 3,
        "figure8_points_per_turn": 300,
        "duration_sec": SIM_DURATION_SEC,
        "modem_selector_fn": None,
    }
    if sim_config:
        cfg.update(sim_config)

    local_rng = rng
    bias_state = np.zeros(3)

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
        "acoustic_skipped": {"t": [], "values": [], "dof": 1},
    }
    times = []

    active_counts = []
    active_sets = []
    rank_xy_series = []
    gdop_xy_series = []
    rank_3d_series = []
    gdop_3d_series = []
    mode_series = []

    use_dvl_update = bool(cfg.get("enable_dvl", cfg.get("use_dvl_update", True)))
    use_depth_update = bool(cfg.get("enable_depth", cfg.get("use_depth_update", True)))
    use_acoustic_updates = bool(cfg.get("enable_acoustic", cfg.get("use_acoustic_updates", True)))

    # Clear any lingering static sensor state before creating a new env instance
    _reset_acoustic_sensor_state()

    with holoocean.make(
        SCENARIO_NAME,
        show_viewport=bool(cfg.get("show_viewport", False)),
        frames_per_sec=False,
        verbose=False,
    ) as env:

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

        # Choose which USVs to range to (CLI uses --target-name). Default: all.
        selected_usv_ids = list(USV_IDS)
        if target_names and not bool(cfg.get("use_all_acoustic", True)):
            requested = []
            for n in target_names:
                if n in name_to_id:
                    requested.append(name_to_id[n])
                else:
                    print(f"[INFO] Unknown target name '{n}' — skipping")
            # dedupe preserve order
            selected_usv_ids = list(dict.fromkeys(requested))

        # Filter to beacons actually present
        if actual_ids:
            selected_usv_ids = [bid for bid in selected_usv_ids if bid in actual_ids]

        # Modem dropout schedule (per USV name)
        def _normalize_dropout(intervals_cfg: Dict[Any, Any]) -> Dict[str, List[Tuple[float, float]]]:
            if not intervals_cfg:
                return {}
            out: Dict[str, List[Tuple[float, float]]] = {}
            for key, spans in intervals_cfg.items():
                if isinstance(key, int):
                    name = id_to_agent.get(key)
                else:
                    name = str(key)
                if name not in name_to_id:
                    continue
                norm_spans: List[Tuple[float, float]] = []
                for span in spans or []:
                    if span is None or len(span) < 2:
                        continue
                    try:
                        a, b = float(span[0]), float(span[1])
                    except Exception:
                        continue
                    if b < a:
                        a, b = b, a
                    norm_spans.append((a, b))
                if norm_spans:
                    out[name] = norm_spans
            return out

        modem_dropout_mode = cfg.get("modem_dropout_mode", None)
        modem_dropout_intervals = _normalize_dropout(cfg.get("modem_dropout_intervals", {}))

        def modem_is_enabled(usv_name: str, t_sec: float) -> bool:
            spans = modem_dropout_intervals.get(usv_name)
            if not spans:
                return True
            for a, b in spans:
                if a <= t_sec <= b:
                    return False
            return True

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
                period_ticks=int(cfg.get("acoustic_period_ticks", ACOUSTIC_UPDATE_PERIOD_TICKS)),   # total ping rate, not per-target
                timeout_ticks=int(3 * ticks_per_sec),        # 3s timeout
            )

        # EKF uses IMU acceleration as an input. With the updated kalman_utils.EKF,
        # q_vel_std is interpreted as accel uncertainty std (sigma_a, m/s^2)
        # and q_pos_std is an optional position random-walk (m/sqrt(s)).
        ekf = EKF(
            dt=dt,
            q_pos_std=cfg.get("q_pos_std", Q_POS_STD),
            q_vel_std=cfg.get("q_vel_std", Q_VEL_STD),
            p_pos_std_init=cfg.get("p_pos_std_init", P_POS_STD_INIT),
            p_vel_std_init=cfg.get("p_vel_std_init", P_VEL_STD_INIT),
        )

        duration_sec = float(cfg.get("duration_sec", SIM_DURATION_SEC))
        n_steps = int(duration_sec * ticks_per_sec)
        noise_streams = None
        x0_perturb = None
        if random_streams is not None:
            if hasattr(random_streams, "build"):
                built = random_streams.build(n_steps=n_steps, dt=dt)
            else:
                built = random_streams
            if isinstance(built, dict):
                x0_perturb = built.get("x0_perturb") or built.get("x0")
                noise_streams = built.get("noise_streams") or built.get("streams")
                if noise_streams is None:
                    noise_streams = built

        def _stream_noise(key: str, idx: int, size: int):
            if noise_streams is None:
                return None
            arr = noise_streams.get(key) if isinstance(noise_streams, dict) else None
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.ndim == 1:
                if idx >= arr.shape[0]:
                    return None
                return float(arr[idx])
            if arr.ndim == 2:
                if idx >= arr.shape[0]:
                    return None
                row = arr[idx]
                if row.shape[0] < size:
                    return None
                return row[:size]
            return None
        prev_true_pos = None
        clock = 0
        sim_tick = 0  # counts env.step() calls (true simulation time base)
        idx = 0
        yaw_cmd = TARGET_YAW_DEG

        # Build waypoints from selected trajectory
        waypoint_spacing = cfg.get("waypoint_spacing", 12.5)
        trajectory_name = cfg.get("trajectory", "lawnmower")
        waypoints = np.asarray(build_trajectory(trajectory_name, cfg), dtype=float)
        has_z_in_waypoints = waypoints.shape[1] >= 3

        def _trajectory_xyz() -> np.ndarray:
            if waypoints.size == 0:
                return np.empty((0, 3), dtype=float)
            if has_z_in_waypoints:
                return waypoints[:, :3]
            z_col = np.full((waypoints.shape[0], 1), TARGET_Z, dtype=float)
            return np.hstack((waypoints[:, :2], z_col))

        if cfg.get("print_trajectory_summary", False):
            if not getattr(run_ekf_acoustics, "_printed_traj", False):
                run_ekf_acoustics._printed_traj = True

        for wp in waypoints:
            z_draw = wp[2] if has_z_in_waypoints else TARGET_Z
            env.draw_point([wp[0], wp[1], z_draw], color=[0, 255, 0], thickness=20.0, lifetime=0)


        last_dvl = None
        last_depth = None


        for k in range(n_steps):
            if debug_steps:
                print(f"--- EKF step {k+1}/{n_steps} (sim_tick={sim_tick}) (duration={k/ticks_per_sec}) ---")

            clock += 1

            x_wp = waypoints[idx, 0]
            y_wp = waypoints[idx, 1]
            z_wp = waypoints[idx, 2] if has_z_in_waypoints else TARGET_Z
            target_6d = np.array([x_wp, y_wp, z_wp, 0.0, 0.0, yaw_cmd], dtype=float)

            # --- Step env (exactly once per EKF step) ---
            state = env.step(target_6d)
            sim_tick += 1
            t_current = sim_tick * dt

            # Apply currents (optional)
            apply_currents(env, state, clock, enabled=cfg.get("use_currents", USE_CURRENTS))

            # --- Read sensors ---
            imu   = state[AUV_NAME]["IMUSensor"]
            pose  = state[AUV_NAME]["PoseSensor"]
            loc   = state[AUV_NAME]["LocationSensor"]
            dvl   = state[AUV_NAME]["DVLSensor"]
            depth = state[AUV_NAME]["DepthSensor"]

            accel_meas = imu[0, :]
            accel_bias = imu[2, :] if imu.shape[0] >= 3 else np.zeros(3)
            accel_body = accel_meas - accel_bias

            if cfg.get("imu_accel_extra_std", 0.0) > 0.0:
                noise = _stream_noise("imu_accel", k, 3)
                if noise is not None:
                    accel_body = accel_body + np.asarray(noise, dtype=float) * cfg["imu_accel_extra_std"]
                elif local_rng is not None:
                    accel_body = accel_body + local_rng.normal(0.0, cfg["imu_accel_extra_std"], size=3)

            if cfg.get("imu_bias_rw_std", 0.0) > 0.0:
                noise = _stream_noise("imu_bias_rw", k, 3)
                if noise is not None:
                    bias_state += np.asarray(noise, dtype=float) * cfg["imu_bias_rw_std"] * np.sqrt(dt)
                elif local_rng is not None:
                    bias_state += local_rng.normal(0.0, cfg["imu_bias_rw_std"] * np.sqrt(dt), size=3)
                accel_body = accel_body + bias_state

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
            if debug_steps:
                print(f"accel_body: {accel_body}, a_world_raw: {a_world_raw}, a_world: {a_world}")

            if k == 0:
                ekf.x[0:3] = true_pos.copy()
                ekf.x[3:6] = true_vel.copy()
                if isinstance(x0_perturb, dict):
                    pos_off = np.asarray(x0_perturb.get("pos", [0.0, 0.0, 0.0]), dtype=float)
                    vel_off = np.asarray(x0_perturb.get("vel", [0.0, 0.0, 0.0]), dtype=float)
                    if pos_off.shape[0] >= 3:
                        ekf.x[0:3] = ekf.x[0:3] + pos_off[:3]
                    if vel_off.shape[0] >= 3:
                        ekf.x[3:6] = ekf.x[3:6] + vel_off[:3]

            ekf.predict(a_world)

            # --- DVL update (velocity) ---
            if use_dvl_update and (dvl_period_ticks is None or (sim_tick % dvl_period_ticks) == 0):
                prior_x = ekf.x.copy()
                prior_P = ekf.P.copy()
                v_body = dvl[0:3]
                # DVL reports body-frame velocity; transform to world-frame
                v_world_meas = R_ws @ v_body

                if cfg.get("dvl_extra_std", 0.0) > 0.0:
                    noise = _stream_noise("dvl", k, 3)
                    if noise is not None:
                        v_world_meas = v_world_meas + np.asarray(noise, dtype=float) * cfg["dvl_extra_std"]
                    elif local_rng is not None:
                        v_world_meas = v_world_meas + local_rng.normal(0.0, cfg["dvl_extra_std"], size=3)

                # Only update if this is actually a new measurement (not held/repeated)
                if last_dvl is None or not np.allclose(v_world_meas, last_dvl, atol=1e-3):
                    H_dvl = np.array([
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ])
                    R_dvl = np.diag([cfg.get("dvl_measurement_std", DVL_VEL_STD)**2]*3)

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
                if cfg.get("depth_extra_std", 0.0) > 0.0:
                    noise = _stream_noise("depth", k, 1)
                    if noise is not None:
                        z_meas = z_meas + float(np.asarray(noise).reshape(-1)[0]) * cfg["depth_extra_std"]
                    elif local_rng is not None:
                        z_meas = z_meas + float(local_rng.normal(0.0, cfg["depth_extra_std"]))

                # Only update if new (depth often holds last value between true updates)
                if last_depth is None or abs(z_meas - last_depth) > 1e-3:
                    z_vec = np.array([z_meas])
                    H_depth = np.array([[0, 0, 1, 0, 0, 0]])
                    R_depth = np.array([[cfg.get("depth_measurement_std", DEPTH_STD)**2]])

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
            selector_fn = cfg.get("modem_selector_fn", None)
            active_names = [id_to_agent.get(uid, None) for uid in selected_usv_ids]
            active_names = [n for n in active_names if n is not None]
            rank_xy_sel = np.nan
            gdop_xy_sel = np.nan
            rank_3d_sel = np.nan
            gdop_3d_sel = np.nan
            mode_sel = "none"
            uncertainty_gated = False
            target_info = []

            pos_trace = float(np.trace(ekf.P[0:3, 0:3]))
            unc_thresh = cfg.get("acoustic_uncertainty_trace_thresh", None)
            if unc_thresh is not None and pos_trace <= float(unc_thresh):
                uncertainty_gated = True
                active_names = []
                mode_sel = "uncertainty_gated"

            if selector_fn is not None:
                for usv_id in selected_usv_ids:
                    usv_nm = id_to_agent.get(usv_id, None)
                    if usv_nm is None or usv_nm not in state:
                        continue
                    pos = None
                    if "PoseSensor" in state[usv_nm]:
                        pos = state[usv_nm]["PoseSensor"][0:3, 3]
                    else:
                        pos = state[usv_nm].get("LocationSensor", None)
                    if pos is None:
                        continue
                    target_info.append((usv_nm, np.asarray(pos, dtype=float)))

                decision = None
                if target_info and not uncertainty_gated:
                    try:
                        decision = selector_fn(
                            t_current=t_current,
                            ekf_state=ekf.x.copy(),
                            depth_available=use_depth_update,
                            target_info=target_info,
                            covariance=ekf.P.copy(),
                        )
                    except TypeError:
                        decision = selector_fn(
                            t_current=t_current,
                            ekf_state=ekf.x.copy(),
                            depth_available=use_depth_update,
                            target_info=target_info,
                        )

                if decision:
                    active_indices = decision.get("active_indices", None)
                    if active_indices is not None:
                        active_names = [target_info[i][0] for i in active_indices if 0 <= i < len(target_info)]
                    else:
                        active_names = decision.get("active_names", active_names)
                    rank_xy_sel = float(decision.get("rank_xy", np.nan))
                    gdop_xy_sel = float(decision.get("gdop_xy", np.nan))
                    rank_3d_sel = float(decision.get("rank_3d", np.nan))
                    gdop_3d_sel = float(decision.get("gdop_3d", np.nan))
                    mode_sel = decision.get("mode", "none")

            active_set = set(active_names)
            if target_info:
                available_names = {n for n, _ in target_info}
                active_set = active_set.intersection(available_names)

            if modem_dropout_mode == "ignore_updates" and active_set:
                active_set = {n for n in active_set if modem_is_enabled(n, t_current)}

            if uncertainty_gated:
                if verbose:
                    print(f"[acoustic] gated by uncertainty: trace(Ppos)={pos_trace:.4f} <= {unc_thresh}")

            if use_acoustic_updates and acoustic_mgr is not None and not uncertainty_gated:
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
                    dist_m_noisy = dist_m
                    if cfg.get("range_extra_std", 0.0) > 0.0:
                        noise = _stream_noise("range", k, 1)
                        if noise is not None:
                            dist_m_noisy = dist_m_noisy + float(np.asarray(noise).reshape(-1)[0]) * cfg["range_extra_std"]
                        elif local_rng is not None:
                            dist_m_noisy = dist_m_noisy + float(local_rng.normal(0.0, cfg["range_extra_std"]))

                    H_range = np.zeros((1, 6))
                    H_range[0, 0] = dx / dist_pred
                    H_range[0, 1] = dy / dist_pred
                    H_range[0, 2] = dz / dist_pred
                    R_range = np.array([[cfg.get("acoustic_measurement_std", ACOUSTIC_RANGE_STD)**2]])
                    y_range = np.array([[dist_m_noisy - dist_pred]])
                    S_range = H_range @ prior_P @ H_range.T + R_range
                    try:
                        nis_val = float(y_range.T @ np.linalg.solve(S_range, y_range))
                    except np.linalg.LinAlgError:
                        nis_val = float(y_range.T @ np.linalg.pinv(S_range) @ y_range)

                    disabled = False
                    if usv_name not in active_set:
                        disabled = True
                    if modem_dropout_mode == "ignore_updates":
                        if usv_name is not None and not modem_is_enabled(usv_name, t_current):
                            disabled = True

                    if disabled:
                        nis_logs.setdefault("acoustic_skipped", {"t": [], "values": [], "dof": 1})
                        nis_logs["acoustic_skipped"]["t"].append(t_current)
                        nis_logs["acoustic_skipped"]["values"].append(nis_val)
                        continue

                    nis_logs["acoustic"]["t"].append(t_current)
                    nis_logs["acoustic"]["values"].append(nis_val)

                    ekf.update_range(dist_m_noisy, beacon_pos, cfg.get("acoustic_measurement_std", ACOUSTIC_RANGE_STD)**2)

            active_counts.append(len(active_set))
            active_sets.append("|".join(sorted(active_set)) if active_set else "")
            rank_xy_series.append(rank_xy_sel)
            gdop_xy_series.append(gdop_xy_sel)
            rank_3d_series.append(rank_3d_sel)
            gdop_3d_series.append(gdop_3d_sel)
            mode_series.append(mode_sel)

            # --- Heading control: face currents or waypoint ---
            # Hold yaw fixed at 60 degrees (TARGET_YAW_DEG)

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
            depth_err = abs(loc[2] - z_wp)
            if dist_to_wp <= POS_TOL and depth_err <= POS_TOL:
                if idx >= len(waypoints) - 1:
                    if verbose:
                        print("Final waypoint reached. Ending simulation.")
                    break
                idx += 1

    times = np.array(times)
    true_positions = np.array(true_positions)
    est_positions = np.array(est_positions)
    true_velocities = np.array(true_velocities)
    est_velocities = np.array(est_velocities)
    pos_covariances = np.array(pos_covariances)

    active_counts_arr = np.asarray(active_counts)
    active_sets_arr = np.asarray(active_sets)
    rank_xy_arr = np.asarray(rank_xy_series)
    gdop_xy_arr = np.asarray(gdop_xy_series)
    rank_3d_arr = np.asarray(rank_3d_series)
    gdop_3d_arr = np.asarray(gdop_3d_series)
    mode_arr = np.asarray(mode_series)

    return (
        times,
        true_positions,
        est_positions,
        true_velocities,
        est_velocities,
        pos_covariances,
        full_covariances,
        nis_logs,
        active_counts_arr,
        active_sets_arr,
        rank_xy_arr,
        gdop_xy_arr,
        rank_3d_arr,
        gdop_3d_arr,
        mode_arr,
    )


def plot_ekf_outputs(
    times,
    true_pos,
    est_pos,
    true_vel,
    est_vel,
    Ppos,
    Pfull,
    nees_full,
    nees_pos,
    nees_full_test,
    nees_pos_test,
    nis_logs,
    nis_results,
):
    """Render diagnostic plots shared by main() and run_single_trial()."""

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
    X, Y, Z = covariance_ellipsoid_mesh(
        P_final,
        mean,
        chi2_val=chi2_3d_95,
        num_u=25,
        num_v=25,
    )
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


def run_single_trial(
    seed: int,
    config_overrides: Optional[Dict[str, Any]] = None,
    return_timeseries: bool = False,
    target_names=None,
    make_plots: bool = False,
    random_streams: Optional[Any] = None,
):
    """Run one EKF simulation with reproducible seed and collect metrics."""

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    base_config: Dict[str, Any] = {
        "use_currents": False,
        "use_dvl_update": True,
        "use_depth_update": True,
        "use_acoustic_updates": True,
        "enable_dvl": True,
        "enable_depth": True,
        "enable_acoustic": True,
        "use_all_acoustic": True,
        "acoustic_uncertainty_trace_thresh": None,
        "acoustic_period_ticks": ACOUSTIC_UPDATE_PERIOD_TICKS,
        "dvl_measurement_std": DVL_VEL_STD,
        "depth_measurement_std": DEPTH_STD,
        "acoustic_measurement_std": ACOUSTIC_RANGE_STD,
        "dvl_extra_std": 0.0,
        "depth_extra_std": 0.0,
        "range_extra_std": 0.0,
        "imu_accel_extra_std": 0.0,
        "imu_bias_rw_std": 0.0,
        "q_pos_std": Q_POS_STD,
        "q_vel_std": Q_VEL_STD,
        "p_pos_std_init": P_POS_STD_INIT,
        "p_vel_std_init": P_VEL_STD_INIT,
        "waypoint_spacing": 12.5,
        "trajectory": "lawnmower",
        "spiral_center": (200.0, -200.0),
        "spiral_min_radius": 20.0,
        "spiral_max_radius": 50.0,
        "spiral_turns": 6,
        "spiral_points_per_rev": 250,
        "spiral_z_start": -5.0,
        "spiral_z_end": TARGET_Z,
        "nees_ds_stride": 0,
        "nees_ds_alpha": 0.05,
        "duration_sec": SIM_DURATION_SEC,
        "print_trajectory_summary": False,
    }

    if config_overrides:
        base_config.update(config_overrides)

    t0 = time.perf_counter()
    (
        times,
        true_pos,
        est_pos,
        true_vel,
        est_vel,
        Ppos,
        Pfull,
        nis_logs,
        active_counts,
        active_sets,
        rank_xy_series,
        gdop_xy_series,
        rank_3d_series,
        gdop_3d_series,
        mode_series,
    ) = run_ekf_acoustics(
        target_names=target_names,
        verbose=False,
        sim_config=base_config,
        rng=rng,
        random_streams=random_streams,
    )
    runtime = time.perf_counter() - t0

    distance_true_m = path_length(true_pos)
    distance_est_m = path_length(est_pos)

    dt_est = float(times[1] - times[0]) if len(times) > 1 else 1.0 / float(DEFAULT_TICKS_PER_SEC)
    base_config["dt"] = dt_est
    base_config["ticks_per_sec"] = float(1.0 / dt_est) if dt_est > 0 else DEFAULT_TICKS_PER_SEC

    pos_rmse, pos_axis = compute_rmse(true_pos, est_pos)
    vel_rmse, vel_axis = compute_rmse(true_vel, est_vel)
    final_err = float(np.linalg.norm(est_pos[-1] - true_pos[-1]))

    state_err = np.hstack((est_pos - true_pos, est_vel - true_vel))
    nees_full = calculate_nees(state_err, Pfull)
    nees_pos = calculate_nees(state_err, Pfull, indices=[0, 1, 2])
    nees_full_test = nees_consistency_test(nees_full, dof=6)
    nees_pos_test = nees_consistency_test(nees_pos, dof=3)
    ds_stride = int(base_config.get("nees_ds_stride", 0) or 0)
    nees_full_ds = None
    if ds_stride > 0:
        from validation_metrics import downsampled_mean_nees_test

        nees_full_ds = downsampled_mean_nees_test(
            nees_full,
            dof=6,
            stride=ds_stride,
            alpha=float(base_config.get("nees_ds_alpha", 0.05)),
        )

    nis_results: Dict[str, Optional[Dict[str, Any]]] = {}
    for key in ("dvl", "depth", "acoustic"):
        vals = np.array(nis_logs[key]["values"])
        if vals.size == 0:
            nis_results[key] = None
            continue
        nis_results[key] = nis_consistency_test(vals, dof=nis_logs[key]["dof"])

    if make_plots:
        plot_ekf_outputs(
            times,
            true_pos,
            est_pos,
            true_vel,
            est_vel,
            Ppos,
            Pfull,
            nees_full,
            nees_pos,
            nees_full_test,
            nees_pos_test,
            nis_logs,
            nis_results,
        )

    # Drop non-serializable fields (e.g., callable modem_selector_fn) from config for return
    config_for_log = {k: v for k, v in base_config.items() if not callable(v)}

    trial: Dict[str, Any] = {
        "seed": int(seed),
        "runtime_seconds": float(runtime),
        "pos_rmse_total": float(pos_rmse),
        "pos_rmse_xyz": pos_axis.tolist(),
        "vel_rmse_total": float(vel_rmse),
        "vel_rmse_xyz": vel_axis.tolist(),
        "final_position_error": final_err,
        "distance_true_m": distance_true_m,
        "distance_est_m": distance_est_m,
        "nees_full": nees_full_test,
        "nees_full_ds_mean": nees_full_ds,
        "nees_pos": nees_pos_test,
        "nis": nis_results,
        "config": config_for_log,
    }

    if isinstance(random_streams, dict) and ("x0_perturb" in random_streams or "x0" in random_streams):
        trial["x0_perturb"] = random_streams.get("x0_perturb") or random_streams.get("x0")
    elif hasattr(random_streams, "x0_perturb"):
        trial["x0_perturb"] = getattr(random_streams, "x0_perturb")

    if return_timeseries:
        trial["timeseries"] = {
            "t": np.asarray(times),
            "true_pos": np.asarray(true_pos),
            "est_pos": np.asarray(est_pos),
            "true_vel": np.asarray(true_vel),
            "est_vel": np.asarray(est_vel),
            "Ppos": np.asarray(Ppos),
            "Pfull": np.asarray(Pfull),
            "err_norm": np.linalg.norm(state_err[:, 0:3], axis=1),
            "pos_err": np.linalg.norm(state_err[:, 0:3], axis=1),
            "nees_full": np.asarray(nees_full),
            "nees_pos": np.asarray(nees_pos),
            "nis_logs": nis_logs,
            "active_count": np.asarray(active_counts),
            "active_set": np.asarray(active_sets),
            "rank_xy": np.asarray(rank_xy_series),
            "gdop_xy": np.asarray(gdop_xy_series),
            "rank_3d": np.asarray(rank_3d_series),
            "gdop_3d": np.asarray(gdop_3d_series),
            "mode": np.asarray(mode_series),
        }

    return trial


def main(target_names=None, verbose=False, trajectory="lawnmower", show_viewport=False):
    print("\nRunning EKF fusion: IMU + DVL + Depth + Acoustic_1...")
    (
        times,
        true_pos,
        est_pos,
        true_vel,
        est_vel,
        Ppos,
        Pfull,
        nis_logs,
        active_counts,
        active_sets,
        rank_xy_series,
        gdop_xy_series,
        rank_3d_series,
        gdop_3d_series,
        mode_series,
    ) = run_ekf_acoustics(
        target_names=target_names,
        verbose=verbose,
        sim_config={"trajectory": trajectory, "show_viewport": show_viewport},
    )

    pos_rmse, pos_axis = compute_rmse(true_pos, est_pos)
    vel_rmse, vel_axis = compute_rmse(true_vel, est_vel)
    final_err = np.linalg.norm(est_pos[-1] - true_pos[-1])
    distance_true_m = path_length(true_pos)
    distance_est_m = path_length(est_pos)

    print("\n========== EKF FUSION RESULTS ==========")
    print(f"  Position RMSE total  : {pos_rmse:.3f} m")
    print(f"    axes (x,y,z)       : {pos_axis}")
    print(f"  Velocity RMSE total  : {vel_rmse:.3f} m/s")
    print(f"    axes (x,y,z)       : {vel_axis}")
    print(f"  Final position error : {final_err:.3f} m")
    print(f"  Distance traveled (true) : {distance_true_m:.1f} m")
    print(f"  Distance traveled (est)  : {distance_est_m:.1f} m")
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

    plot_ekf_outputs(
        times,
        true_pos,
        est_pos,
        true_vel,
        est_vel,
        Ppos,
        Pfull,
        nees_full,
        nees_pos,
        nees_full_test,
        nees_pos_test,
        nis_logs,
        nis_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKF fusion with optional acoustic ranging")
    parser.add_argument("--target-name", type=str, action="append",
                        help="Target agent name (e.g., usv1); repeat to specify multiple")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--trajectory", choices=["lawnmower", "spiral", "concentric", "figure8"], default="spiral",
                        help="Select trajectory for XY waypoints")
    parser.add_argument("--show-viewport", action="store_true", default=False,
                        help="Show HoloOcean viewport (off by default for headless runs)")
    args = parser.parse_args()
    main(target_names=args.target_name, verbose=args.verbose, trajectory=args.trajectory, show_viewport=args.show_viewport)

# End of script
