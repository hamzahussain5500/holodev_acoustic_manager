import holoocean
import numpy as np
import time
import argparse


# Trilateration removed: this script now only collects and prints ranges from beacons.

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
USV_BEACON_ID_3  = 3
USV_BEACON_ID_4      = 4

# ===== Currents control (ported from EKF.py) =====
USE_CURRENTS = False
MAP_DIMENSIONS = [100, 100, 35]
DRAW_CURRENT_FIELD_STEP = 100
VEHICLES_FOR_CURRENTS = ["auv"]  # apply currents to AUV only


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

    USV_IDS = [USV_BEACON_ID_1, USV_BEACON_ID_2, USV_BEACON_ID_3, USV_BEACON_ID_4]
    AUV_ID = AUV_BEACON_ID

    return id_to_agent, USV_IDS, AUV_ID, TICKS_PER_SEC


def run_round_robin(env, id_to_agent, USV_IDS, AUV_ID, TICKS_PER_SEC, max_wait_seconds=3, verbose=False):
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
        print(f"\nRanging to {target_name} (beacon {target_id})")

        # Show current modem statuses (verbose)
        if verbose:
            auv_stat = get_status_by_beacon_id(env, AUV_ID)
            tgt_stat = get_status_by_beacon_id(env, target_id)
            print(f"    Modem statuses before send: AUV={auv_stat}, Target={tgt_stat}")

        # Tick once before sending
        state = local_safe_tick()
        tick_count += 1

        # Wait for Idle statuses before sending (best-effort)
        waited_ticks = 0
        for _ in range(max_wait_ticks):
            auv_stat = get_status_by_beacon_id(env, AUV_ID)
            tgt_stat = get_status_by_beacon_id(env, target_id)
            if auv_stat in (None, "Idle") and tgt_stat in (None, "Idle"):
                break
            state = local_safe_tick()
            tick_count += 1
            waited_ticks += 1
        if verbose and waited_ticks:
            print(f"    Waited {waited_ticks} ticks for Idle")

        # Send request from AUV to target
        send_tick = tick_count
        env.send_acoustic_message(AUV_ID, target_id, "MSG_REQX", "range_req")

        # Wait for response
        got_response = False
        for _ in range(max_wait_ticks):
            state = local_safe_tick()
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
                        break

        if not got_response:
            print(f"  [WARN] No MSG_RESPX received from {target_name} within timeout")

    return ranges


def vortex_field(location):
    """
    Example vortex-like current field in XY plane (from EKF.py).
    Rotates around origin in XY, zero Z.
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


def main(loop=True, num_usvs=4, target_ids=None, target_names=None, verbose=False):

    # Create the environment once, then use functions to operate on it
    with holoocean.make(SCENARIO_NAME, show_viewport=True, frames_per_sec=False) as env:

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
            print("\nInterrupted by user (Ctrl-C). Exiting.")


if __name__ == "__main__":
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

# End of script
