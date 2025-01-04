from tkinter import *
from PIL import ImageTk, Image

import dxcam
import mss
import time

TIMER_INTERVAL = 0.0
TIMER = 0.0
DELTA_TIME = 0.0
LAST_TIME = None
DISPLAYED_IMAGE = None
LAST_CONFIGURE_EVENT = None
WINDOW = None
IMG = None
CAMERA = dxcam.create(output_color='BGR')
MSS = mss.mss()


def configure_handler(event):
    global DISPLAYED_IMAGE
    global LAST_CONFIGURE_EVENT
    global WINDOW
    if event.widget == WINDOW:
        if DISPLAYED_IMAGE is not None:
            DISPLAYED_IMAGE.destroy()
            DISPLAYED_IMAGE = None
    LAST_CONFIGURE_EVENT = event


def destroy_handler(event):
    global CAMERA
    if event.widget == WINDOW:
        del CAMERA


def create_window(interval=10.0):
    global TIMER_INTERVAL
    global TIMER
    global WINDOW
    TIMER_INTERVAL = interval
    TIMER = interval
    root = Tk()
    root.geometry('800x600')
    root.attributes('-transparentcolor', '#f0f0f0')
    root.attributes('-topmost', True)
    root.resizable(True, True)
    root.bind('<Configure>', configure_handler)
    root.bind('<Destroy>', destroy_handler)
    WINDOW = root
    return root


def update_window(window: Tk, new_image=None):
    global TIMER
    global DELTA_TIME
    global LAST_TIME
    global IMG
    TIMER -= DELTA_TIME
    if new_image is not None:
        global TIMER_INTERVAL
        global DISPLAYED_IMAGE
        if DISPLAYED_IMAGE is not None:
            DISPLAYED_IMAGE.destroy()
            DISPLAYED_IMAGE = None
        TIMER = TIMER_INTERVAL
        IMG = ImageTk.PhotoImage(Image.fromarray(new_image))
        DISPLAYED_IMAGE = Label(window, image=IMG, padx=0, pady=0, borderwidth=0)
        DISPLAYED_IMAGE.place(x=0, y=0)

    window.update()
    new_time = time.time()
    if LAST_TIME is not None:
        DELTA_TIME = new_time - LAST_TIME
    LAST_TIME = new_time


def get_timer():
    global TIMER
    return TIMER


def take_screenshot(window: Tk):
    global DISPLAYED_IMAGE
    global CAMERA
    global MSS
    x = window.winfo_rootx()
    y = window.winfo_rooty()
    w = window.winfo_width()
    h = window.winfo_height()
    top = MSS.monitors[0]['top']
    left = MSS.monitors[0]['left']

    if DISPLAYED_IMAGE is not None:
        DISPLAYED_IMAGE.place_forget()
        window.update()
        screenshot_time = time.time()
        screenshot = MSS.grab({'top': y, 'left': x, 'width': w, 'height': h})
        screenshot_time = time.time() - screenshot_time
        print('time_to_screenshot', screenshot_time)
        DISPLAYED_IMAGE.place(x=0, y=0)
        window.update()
    else:
        screenshot_time = time.time()
        screenshot = MSS.grab({'top': y, 'left': x, 'width': w, 'height': h})
        screenshot_time = time.time() - screenshot_time
        print('time_to_screenshot', screenshot_time)
    return screenshot
