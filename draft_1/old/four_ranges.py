import numpy as np
import time


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


def build_mappings(env, static_agent_positions=None):
    """Deprecated mapping helper (kept for backwards compatibility)."""
    TICKS_PER_SEC = getattr(env, "ticks_per_sec", 30.0) if hasattr(env, "ticks_per_sec") else 30.0
    return {}, {}, [], None, TICKS_PER_SEC


def collect_acoustic_ranges(env,
                            auv_name,
                            auv_beacon_id,
                            target_usv_ids,
                            beacon_id_to_agent,
                            num_targets=None,
                            max_wait_seconds=1.0,
                            ticks_per_sec=None,
                            verbose=False):
    """
    Send MSG_REQX from AUV beacon to each target beacon id and wait for MSG_RESPX.

    Inputs:
    - env: active HoloOcean environment (already created by caller)
    - auv_name: agent name for the AUV (to read its AcousticBeaconSensor)
    - auv_beacon_id: beacon id used by AUV to send requests
    - target_usv_ids: list of beacon ids to range to
    - beacon_id_to_agent: dict mapping beacon_id -> agent name (for position lookup)
    - num_targets: if set, limit to first N target ids from target_usv_ids
    - max_wait_seconds: timeout per target
    - ticks_per_sec: optional; inferred from env if None
    - verbose: if True, print steps

    Returns:
    - results: list of dicts (ordered as processed) with keys:
        { 'beacon_id': int, 'range': float or None, 'beacon_pos': np.ndarray or None }
    """
    last_state = [{}]
    T = ticks_per_sec if ticks_per_sec is not None else (
        getattr(env, "ticks_per_sec", 30.0) if hasattr(env, "ticks_per_sec") else 30.0
    )
    max_wait_ticks = int(max_wait_seconds * T)

    targets = list(target_usv_ids)
    if num_targets is not None:
        targets = targets[:max(0, int(num_targets))]

    results = []

    for target_id in targets:
        if verbose:
            print(f"[ranges] Pinging beacon {target_id} from {auv_beacon_id}")

        # tick once for freshness
        _ = safe_tick(env, last_state)

        env.send_acoustic_message(auv_beacon_id, target_id, "MSG_REQX", "range_req")

        dist_val = None
        beacon_pos = None

        # wait for response
        for _ in range(max_wait_ticks):
            state = safe_tick(env, last_state)
            auv_sensors = state.get(auv_name, {}) if isinstance(state, dict) else {}
            if isinstance(auv_sensors, dict) and "AcousticBeaconSensor" in auv_sensors:
                msg = auv_sensors["AcousticBeaconSensor"]
                if msg is None or not isinstance(msg, (list, tuple)) or len(msg) < 3:
                    continue
                msg_type = msg[0]
                from_id = msg[1]
                if msg_type == "MSG_RESPX" and from_id == target_id:
                    if len(msg) >= 7:
                        _, _, payload, phi, theta, dist, depth = msg
                        if dist is not None:
                            dist_val = float(dist)
                    break

        # try to read beacon (USV) position from the same (or latest) state
        state = last_state[0] if last_state[0] else state
        agent_name = beacon_id_to_agent.get(target_id) if isinstance(beacon_id_to_agent, dict) else None
        if isinstance(state, dict) and agent_name in state:
            agent_state = state[agent_name]
            if isinstance(agent_state, dict):
                if "PoseSensor" in agent_state:
                    pose = agent_state["PoseSensor"]
                    beacon_pos = np.array(pose[0:3, 3], dtype=float)
                elif "LocationSensor" in agent_state:
                    beacon_pos = np.array(agent_state["LocationSensor"], dtype=float)

        results.append({
            'beacon_id': target_id,
            'range': dist_val,
            'beacon_pos': beacon_pos,
        })

        if verbose:
            print(f"[ranges] beacon {target_id}: r={dist_val} pos={beacon_pos}")

        # no extra cooldown ticks; proceed immediately to next target

    return results


# Backwards-compatible alias
def ranges(*args, **kwargs):
    return collect_acoustic_ranges(*args, **kwargs)


__all__ = [
    'safe_tick',
    'collect_acoustic_ranges',
    'ranges',
]
