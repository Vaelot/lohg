from concurrent.futures import ProcessPoolExecutor as PPE
from concurrent.futures import ThreadPoolExecutor as TPE

import pprint
import time
import cv2
import keyboard
import mss
import mouse
import numpy as np
from PIL import Image

from parser import extract_metadata

THRESHOLD = 0.8

MASK_REGIONS = [
    (1248, 138, 1763, 563),
]


def init_import():
    import config  # noqa: F401


ex = PPE(max_workers=16, initializer=init_import)


def capture_screen():
    global ex
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        time.sleep(0.01)
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        data = extract_metadata(ex, img_bgr)
        pprint.pprint(data)

ctrl_pressed = False


def on_key_event(e):
    global ctrl_pressed
    if e.name == "ctrl" and e.event_type == "down":
        ctrl_pressed = True
    elif e.name == "ctrl" and e.event_type == "up":
        ctrl_pressed = False


def on_click(event):
    global ctrl_pressed
    if (
        isinstance(event, mouse.ButtonEvent)
        and event.event_type == "down"
        and event.button == mouse.LEFT
    ):
        if ctrl_pressed:
            capture_screen()


if __name__ == "__main__":
    mouse.hook(on_click)
    keyboard.hook(on_key_event)

    while True:
        if keyboard.is_pressed("esc"):
            break
        time.sleep(0.1)

    ex.shutdown()
    mouse.unhook_all()
    keyboard.unhook_all()
