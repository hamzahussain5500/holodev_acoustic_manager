import holoocean
import numpy as np
from pynput import keyboard


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

def vortex_field(location):
    x, y, z = location
    if z > 0:
        return [0, 0, 0]
    strength = 5.0
    r_squared = x**2 + y**2 + 1e-5  # avoid divide by zero
    dx = -y / r_squared * strength
    dy = x / r_squared * strength
    dz = 0.2 * np.cos(0.1 * r_squared)
    return [3*dx, 3*dy, 3*dz]

vehicles = ['auv', 'usv1']
map_dimensions = [100, 100, 35]

with holoocean.make("blue_rov") as env:
    clock = 0
    while True:
        clock += 1
        if clock == 20:
            env.draw_debug_vector_field(vortex_field, location=[0, 0, 0], vector_field_dimensions=map_dimensions, arrow_thickness=7, arrow_size=.25, spacing=3)
        tick_info = env.tick()
        command = parse_keys(pressed_keys, force)
        env.act("auv", command)

        # Apply currents
        for vehicle in vehicles:
            location = tick_info[vehicle]['LocationSensor']
            current_velocity = vortex_field(location)
            env.set_ocean_currents(vehicle, current_velocity)