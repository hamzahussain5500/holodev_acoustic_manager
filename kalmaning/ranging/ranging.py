import holoocean
import numpy as np
import time
import argparse


# Trilateration removed: this script now only collects and prints ranges from beacons.


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
    """Build id_to_agent and beacon_positions mappings where possible.

    static_agent_positions: optional dict agent_name -> np.array pos used as fallback for beacon positions.
    Returns (id_to_agent, beacon_positions, usv_ids, auv_id, ticks_per_sec)
    """
    # Basic config constants (can be tuned)
    TICKS_PER_SEC = getattr(env, "ticks_per_sec", 30.0) if hasattr(env, "ticks_per_sec") else 30.0

    id_to_agent = {}
    try:
        for b in env.beacons:
            if not isinstance(b, int) and hasattr(b, "id"):
                id_to_agent[b.id] = getattr(b, "agent_name", None)
    except Exception:
        # env.beacons can be simple list of ints or other structures
        pass

    # If env.beacons_id exists, use it; otherwise try to infer
    beacons_id = getattr(env, "beacons_id", None)
    if beacons_id is None and hasattr(env, "beacons"):
        # try to extract ints from env.beacons
        try:
            beacons_id = [b.id if not isinstance(b, int) else b for b in env.beacons]
        except Exception:
            beacons_id = None

    # Build beacon_positions from static_agent_positions using a default USV ordering if provided
    beacon_positions = {}
    USV_IDS = []
    AUV_ID = None

    if static_agent_positions and id_to_agent:
        for bid, agent in id_to_agent.items():
            if agent in static_agent_positions:
                beacon_positions[bid] = static_agent_positions[agent]

    # Best-effort USV / AUV id detection
    if beacons_id is not None:
        # assume the AUV is the largest id (heuristic) or id not in USV list
        all_ids = list(beacons_id)
        # Use 4 USVs heuristic if present
        if len(all_ids) >= 4:
            USV_IDS = sorted(all_ids)[:4]
            # take AUV as last id not in USV_IDS if present
            rest = [i for i in all_ids if i not in USV_IDS]
            AUV_ID = rest[0] if rest else (max(all_ids) if all_ids else None)
        else:
            USV_IDS = all_ids
            AUV_ID = max(all_ids) if all_ids else None

    # Fallback: if we couldn't construct beacon_positions from id_to_agent,
    # try to populate it from provided static_agent_positions by assigning
    # USV IDs to names 'usv1', 'usv2', ... in order. Also populate id_to_agent
    # accordingly so later prints show names.
    if static_agent_positions and not beacon_positions and USV_IDS:
        for idx, bid in enumerate(sorted(USV_IDS)):
            name = f"usv{idx+1}"
            if name in static_agent_positions:
                beacon_positions[bid] = static_agent_positions[name]
                # fill id_to_agent if missing
                if bid not in id_to_agent:
                    id_to_agent[bid] = name
        # map AUV_ID to 'auv' if possible
        if AUV_ID is not None and AUV_ID not in beacon_positions and 'auv' in static_agent_positions:
            beacon_positions[AUV_ID] = static_agent_positions['auv']
            if AUV_ID not in id_to_agent:
                id_to_agent[AUV_ID] = 'auv'

    return id_to_agent, beacon_positions, USV_IDS, AUV_ID, TICKS_PER_SEC


def run_round_robin(env, id_to_agent, beacon_positions, USV_IDS, AUV_ID, TICKS_PER_SEC, max_wait_seconds=3):
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
        target_name = id_to_agent.get(target_id, f"id={target_id}")
        print(f"\n--- Ranging to {target_name} (beacon {target_id}) ---")

        # Tick once before sending
        state = local_safe_tick()
        tick_count += 1

        # Send request from AUV to target
        send_tick = tick_count
        send_time = send_tick / TICKS_PER_SEC
        env.send_acoustic_message(AUV_ID, target_id, "MSG_REQX", "range_req")

        # Wait for response
        got_response = False
        for _ in range(max_wait_ticks):
            state = local_safe_tick()
            tick_count += 1

            auv_sensors = state.get("auv", {})
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
                            print(f"    r_modem (from msg) = {dist:.2f} m")
                            ranges[target_id] = float(dist)
                        else:
                            print("    [WARN] response arrived but distance not present in message")

                        got_response = True
                        break

        if not got_response:
            print(f"  [WARN] No MSG_RESPX received from {target_name} within timeout")

    return ranges


def vortex_field(cqenter, point, strength=5.0):
    """Return a current vector at `point` induced by a vortex centered at `center`.

    - center: np.array([x,y,z]) center of vortex (AUV location)
    - point: sequence-like [x,y,z] where we want the current
    - strength: scalar to scale the flow magnitude

    The field has strong horizontal swirl and a small vertical component.
    """
    px, py, pz = point
    cx, cy, cz = 0.0, 0.0
    dx = px - cx
    dy = py - cy
    dz = pz - cz

    # ignore points above surface (z>0)
    if pz > 0:
        return [0.0, 0.0, 0.0]

    # horizontal distance squared (avoid div by zero)
    r2 = dx * dx + dy * dy + 1e-6

    # simple vortex: tangential velocity ~ strength / r
    vx = -dy / r2 * strength
    vy = dx / r2 * strength

    # vertical component small, decays with radius
    vz = 0.2 * np.cos(0.1 * r2) * (1.0 / (1.0 + 0.1 * abs(dz)))

    # clamp or scale down so values are reasonable for env.set_ocean_currents
    return [vx, vy, vz]


def main(loop=True):
    # Static positions taken from your usv_auv JSON (world coordinates)
    static_agent_positions = {
        "usv1": np.array([10.0,  10.0,   -1]),
        "usv2": np.array([0.0, 0.0,   -1]),
       # "usv3": np.array([5.0,  -651.0,   0.0]),
        #"usv4": np.array([5.0,  -657.0,  -2.0]),
        "auv":  np.array([-5,  -5,  -15.0]),
    }

    # Create the environment once, then use functions to operate on it
    with holoocean.make("blue_rov", show_viewport=False, frames_per_sec=False) as env:
        print("=== env.info() ===")
        print(env.info())
        print("==================")

        # Build maps and defaults
        id_to_agent, beacon_positions, USV_IDS, AUV_ID, TICKS_PER_SEC = build_mappings(env, static_agent_positions)

        if not USV_IDS:
            print("[ERROR] No USV_IDS detected. Aborting.")
            return
        if AUV_ID is None:
            print("[ERROR] No AUV_ID detected. Aborting.")
            return

        print("Beacon IDs list from env.beacons_id:", getattr(env, "beacons_id", None))
        print("Using beacon ID", AUV_ID, "as AUV sender (agent=", id_to_agent.get(AUV_ID, 'unknown'), ")")

        try:
            # Run at least once; if loop=True, run repeatedly until Ctrl-C
            # prepare vehicles list (agent names) for applying currents
            vehicles = [name for name in set(id_to_agent.values()) if name]
            if 'auv' not in vehicles and AUV_ID in id_to_agent:
                vehicles.append(id_to_agent[AUV_ID])

            debug_draw_counter = 0
            while True:
                # Step once to get current AUV location and apply currents
                last_state = [{}]
                state = safe_tick(env, last_state)

                # Get AUV location (if available) to center the vortex
                auv_loc = None
                if isinstance(state, dict) and 'auv' in state and isinstance(state['auv'], dict):
                    auv_loc = state['auv'].get('LocationSensor')

                if auv_loc is not None:
                    center = np.array(auv_loc, dtype=float)

                    # draw debug vector field around AUV periodically
                    debug_draw_counter += 1
                    if debug_draw_counter % 50 == 0:
                        # vector field dimensions: x,y = 100 m, z = 10 m
                        env.draw_debug_vector_field(lambda loc: vortex_field(center, loc), location=center.tolist(), vector_field_dimensions=[100, 100, 10], arrow_thickness=4, arrow_size=0.25, spacing=5)

                    # apply currents to all known vehicles using their LocationSensor
                    for vehicle in vehicles:
                        try:
                            veh_loc = state.get(vehicle, {}).get('LocationSensor')
                            if veh_loc is None:
                                continue
                            current_velocity = vortex_field(center, veh_loc)
                            env.set_ocean_currents(vehicle, current_velocity)
                        except Exception:
                            # ignore vehicles without LocationSensor or other issues
                            continue

                # Now run the round-robin ranging (it will advance the sim internally)
                ranges = run_round_robin(env, id_to_agent, beacon_positions, USV_IDS, AUV_ID, TICKS_PER_SEC)

                # Print collected ranges for inspection
                if ranges:
                    print("\n[RESULT] Collected ranges:")
                    for bid in sorted(ranges.keys()):
                        agent = id_to_agent.get(bid, f"id={bid}")
                        print(f"  Beacon {bid} ({agent}): {ranges[bid]:.2f} m")

                if not loop:
                    break

                # small pause to avoid tight loop; user can Ctrl-C to quit
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl-C). Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-range trilateration using HoloOcean acoustic messages")
    parser.add_argument("--loop", action="store_true", help="Run round-robin continuously until Ctrl-C")
    args = parser.parse_args()
    main(loop=args.loop)

# End of script
