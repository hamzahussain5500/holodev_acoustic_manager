import holoocean
import numpy as np
from pynput import keyboard

# set to track currently pressed keys
pressed_keys = set()


def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        pressed_keys.add(str(key))


def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:
        pressed_keys.discard(str(key))


# start the listener in a background thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()


with holoocean.make("usv_auv") as env:

    # --- Print environment info (optional, but very useful) ---
    print("=== env.info() ===")
    print(env.info())
    print("==================")

    # --- Correct way to read and print beacon IDs and their agents ---
    print("=== Acoustic beacons in environment ===")
    # env.beacons_id is a list of ints
    print("Beacon IDs list from env.beacons_id:", env.beacons_id)

    # env.beacons may be a list of AcousticBeaconSensor objects or simple beacon IDs (ints).
    for beacon in env.beacons:
        # If the entry is an int, just print the ID. If it's an object, print its attributes.
        if isinstance(beacon, int):
            print(f"Beacon ID {beacon} (raw int entry)")
        else:
            try:
                print(
                    f"Beacon ID {beacon.id} belongs to agent '{beacon.agent_name}', "
                    f"status={beacon.status}"
                )
            except Exception:
                # Fallback for unexpected structures
                print(f"Beacon (unknown structure): {beacon}")
    print("========================================")

    # Configuration: set to True to print non-dict sensor diagnostics
    SHOW_SENSOR_DIAGNOSTICS = False

    # Build ID -> agent name mapping when possible.
    id_to_agent = {}
    # First try: env.beacons may contain objects with .id and .agent_name
    try:
        for b in env.beacons:
            if not isinstance(b, int):
                try:
                    id_to_agent[b.id] = b.agent_name
                except Exception:
                    # ignore entries we can't introspect
                    pass
    except Exception:
        pass

    # Fallback: try to parse agent names from env.info() and map by order
    if not id_to_agent:
        try:
            info = env.info()
            # collect agent names in order of appearance
            names = []
            for line in info.splitlines():
                line = line.strip()
                if line.startswith("Name:"):
                    # format: "Name: usv1"
                    parts = line.split("Name:")
                    if len(parts) > 1:
                        names.append(parts[1].strip())
            # if the count matches, map env.beacons_id in order
            if hasattr(env, "beacons_id") and len(names) == len(env.beacons_id):
                for bid, name in zip(env.beacons_id, names):
                    id_to_agent[bid] = name
        except Exception:
            pass

    # pick one beacon ID to act as the sender (auv has id=0 in your JSON)
    SENDER_ID = 0
    
    print(f"Using beacon ID {SENDER_ID} as sender\n")

    try:
        while True:

            if "q" in pressed_keys:
                break

            command = np.array([0, 0, 0, 0, 1, 1, 0, 0])
            #command_1 = np.array([150, 150])

            # press "1" to trigger a broadcast ping from AUV (id=0) to all others
            if "1" in pressed_keys:
                # tick once to update sim before sending
                state = env.tick()

                # broadcast: id_to = -1 means "all other beacons"
                env.send_acoustic_message(SENDER_ID, -1, "MSG_REQX", "ping_all")

                # give the sim a few ticks to deliver the message
                for t in range(10):
                    state = env.tick()

                    # check all agents for received messages
                    for agent_name, sensors in state.items():
                        # sensors is expected to be a dict mapping sensor names -> values,
                        # but in some configurations it may be a float/None/etc. Guard against that.
                        if isinstance(sensors, dict):
                            if "AcousticBeaconSensor" in sensors:
                                msg = sensors["AcousticBeaconSensor"]
                                if msg is not None:
                                    # msg is typically like ['OWAY', sender_id, 'ping_all']
                                    sender_id = None
                                    try:
                                        # try to extract sender id from typical message format
                                        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                                            sender_id = msg[1]
                                    except Exception:
                                        sender_id = None

                                    # resolve sender name if we built a map
                                    sender_name = id_to_agent.get(sender_id) if sender_id is not None else None
                                    if sender_name:
                                        print(f"[t={t}] Received at {agent_name} from ID {sender_id} ({sender_name}): {msg}")
                                    else:
                                        print(f"[t={t}] Received at {agent_name} from ID {sender_id}: {msg}")
                        else:
                            # unexpected sensor structure — either suppress or print diagnostic
                            if SHOW_SENSOR_DIAGNOSTICS and sensors is not None:
                                print(
                                    f"[t={t}] {agent_name} sensors unexpected type "
                                    f"{type(sensors).__name__}: {sensors}"
                                )

            # normal sim tick + actions
            state = env.tick()
            env.act("auv", command)
            # env.act("usv1", command_1)

    finally:
        listener.stop()
