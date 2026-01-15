import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import csv


class EKF:
    def __init__(self, dt):
        self.dt = dt

        # state: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1.0

        # Process noise (tune later)
        self.Q = np.diag([0.01, 0.01, 0.01,
                          0.1,  0.1,  0.1])

    def predict(self, acc_world):
        """
        acc_world: np.array([ax, ay, az]) in world frame, gravity removed
        """
        dt = self.dt

        # State transition matrix
        F = np.eye(6)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Control matrix (acceleration)
        B = np.zeros((6, 3))
        B[3, 0] = dt   # vx += ax * dt
        B[4, 1] = dt
        B[5, 2] = dt

        acc_world = np.asarray(acc_world).reshape(3,)

        # Predict state and covariance
        self.x = F @ self.x + B @ acc_world
        self.P = F @ self.P @ F.T + self.Q


pressed_keys = list()
force = 10

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4,7]] += .25 * val
        command[[5,6]] -= .25 * val
    if 'l' in keys:
        command[[4,7]] -= .25 * val
        command[[5,6]] += .25 * val

    if 'w' in keys:
        command[4:8] += val
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val

    return command


env = holoocean.make("blue_rov")  # or your scenario name
# storage for plotting
accel_history = []  # list of [ax, ay, az]
time_history = []
step = 0
for k in range(1000):
    state = env.tick()
    imu = state["auv"]["IMUSensor"]
    accel_body = imu[0, :]
    gyro_body  = imu[1, :]

    command = parse_keys(pressed_keys, force)
    env.act("auv", command)

    if k % 10 == 0:
        print(f"Step {k}")
        print("Accel:", accel_body)
        print("Gyro :", gyro_body)
    # record every step
    accel_history.append([float(accel_body[0]), float(accel_body[1]), float(accel_body[2])])
    time_history.append(step)
    step += 1
# I want to plot the value from accel_body. the output is like: [ 0.05736593 -0.07610494  9.826778  ].
# The x and y values ot get from accel_body are small values around zero, while the z value is around 9.8 (gravity).
# I want to see how these values change over time as the vehicle moves.

# Save CSV of accel history
csv_path = "accel_history.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "ax", "ay", "az"])
    for t, a in zip(time_history, accel_history):
        writer.writerow([t, a[0], a[1], a[2]])

# Plot the accelerations
accel_arr = np.array(accel_history)
plt.figure(figsize=(10, 6))
plt.plot(time_history, accel_arr[:, 0], label="ax")
plt.plot(time_history, accel_arr[:, 1], label="ay")
plt.plot(time_history, accel_arr[:, 2], label="az")
plt.xlabel("Step")
plt.ylabel("Acceleration (m/s^2)")
plt.title("IMU accel_body over time")
plt.legend()
plt.grid(True)
png_path = "accel_plot.png"
plt.tight_layout()
plt.savefig(png_path)
try:
    plt.show()
except Exception:
    # In headless environments, plt.show() may fail; saved PNG is available instead
    pass
