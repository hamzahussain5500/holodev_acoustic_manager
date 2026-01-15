import holoocean
import numpy as np
from std_msgs.msg import ByteMultiArray
from pynput import keyboard


# set to track currently pressed keys
pressed_keys = set()


def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        # non-char keys (e.g., Key.space) - store their name if needed
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


with holoocean.make("blue_rov") as env:
   
   try:
      while True:
         
         if "q" in pressed_keys:
             break
         command = np.array([0,0,0,0,5,5,0,0])
         command_1=np.array([150,150])

         if "1" in pressed_keys:
            state= env.tick()
            env.send_acoustic_message( 1, 0,  "MSG_REQX", "1")
            state= env.tick()

            


      
         state= env.tick() 
         env.act("auv", command)
         #env.act("usv1", command_1)
         
         
         if "AcousticBeaconSensor" in state["usv1"]:
            print("Received message at usv1:", state["usv1"]["AcousticBeaconSensor"])
         
   finally:
      # ensure listener is stopped when exiting
      listener.stop()
