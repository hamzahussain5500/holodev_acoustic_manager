import holoocean
import numpy as np

# Scenario and agent names
SCENARIO_NAME = "blue_rov"
AGENT_NAME    = "auv"   # change to your AUV agent name if different

# Desired constant depth (z) and yaw for the AUV.
# Note: If your world frame uses positive-down z, set a positive depth.
TARGET_Z = -10.0
TARGET_YAW = 0.0  # radians

# Waypoints in XY (we will lift them to 6-DOF targets)
waypoints_xy = np.array([
    [ 15.0,  15.0],
    [-15.0,  15.0],
    [-15.0, -15.0],
    [ 15.0, -15.0],
], dtype=float)

# Proximity threshold to switch to next waypoint (meters)
POS_TOL = 1.0

with holoocean.make(SCENARIO_NAME) as env:
    # Visualize target points at the chosen depth
    for xy in waypoints_xy:
        env.draw_point([xy[0], xy[1], TARGET_Z], lifetime=0)

    idx = 0
    print("Going to waypoint", idx)

    while True:
        # Build 6-DOF target: [x, y, z, roll, pitch, yaw]
        x, y = waypoints_xy[idx]
        target_6d = np.array([x, y, TARGET_Z, 0.0, 0.0, TARGET_YAW], dtype=float)

        # Send 6-length target to AUV PID controller
        state = env.step(target_6d)

        # Read current AUV position (x, y, z)
        pos = state[AGENT_NAME]["LocationSensor"][0:3]

        # Check proximity (3D distance to target position)
        if np.linalg.norm(pos - np.array([x, y, TARGET_Z])) <= POS_TOL:
            idx = (idx + 1) % len(waypoints_xy)
            print("Going to waypoint", idx)