"""Hand-gesture mouse controller backed by pynput."""

import sys
from unittest.mock import MagicMock
# Mock tkinter to prevent pyautogui/pymsgbox import crashes on headless Linux systems
mock_tk = MagicMock()
mock_tk.TkVersion = 8.6
sys.modules['tkinter'] = mock_tk
sys.modules['_tkinter'] = MagicMock()

import time

import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import sounddevice as sd

import hand_tracker as htm

pyautogui.FAILSAFE = False

# Sound helpers
def play_sound_enable():
    fs = 44100
    duration = 0.15
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = np.linspace(600, 900, len(t))
    wave = np.sin(2 * np.pi * freqs * t)
    fade = np.ones_like(wave)
    fade[-1000:] = np.linspace(1, 0, 1000)
    sd.play(wave * 0.3 * fade, fs)

def play_sound_disable():
    fs = 44100
    duration = 0.15
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = np.linspace(400, 250, len(t))
    wave = np.sin(2 * np.pi * freqs * t)
    fade = np.ones_like(wave)
    fade[-1000:] = np.linspace(1, 0, 1000)
    sd.play(wave * 0.3 * fade, fs)


# Constants

W_CAM, H_CAM = 640, 480
FRAME_R = 100        # Active gesture area margin (pixels)
SMOOTHENING = 7      # Cursor smoothing factor

FINGER_CONFIG = {
    "all_up":          [1, 1, 1, 1, 1],
    "all_down":        [0, 0, 0, 0, 0],
    "move":            [0, 1, 0, 0, 0],
    "right_click":     [0, 1, 0, 0, 1], # Index + Pinky up
    "left_click":      [1, 1, 0, 0, 0], # Thumb + Index up
    "drag":            [0, 1, 1, 1, 0],
    "scroll_up":       [1, 1, 1, 0, 0], # Index + Middle + Ring up
    "scroll_down":     [0, 1, 1, 0, 0], # 4 fingers up (except thumb)
    "enable_control":  [1, 1, 1, 0, 1], # Thumb + Index + Middle + Pinky up (4 fingers)
    "disable_control": [1, 1, 1, 1, 1], # 5 fingers up
}


# Drag helpers
_mouse = Controller()


def start_drag():
    _mouse.press(Button.left)


def stop_drag():
    _mouse.release(Button.left)



# Main loop
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, W_CAM)
    cap.set(4, H_CAM)

    detector = htm.HandDetector(max_hands=1)
    w_scr, h_scr = pyautogui.size()

    p_time = 0
    p_loc_x, p_loc_y = 0, 0
    c_loc_x, c_loc_y = 0, 0
    dragging = False

    left_clicked = False
    right_clicked = False
    scroll_counter = 0
    control_enabled = False
    last_non_click_gesture = None

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lm_list, _ = detector.find_position(img)

        x1 = y1 = x2 = y2 = 0
        if lm_list:
            x1, y1, _ = lm_list[8][1:]
            x2, y2, _ = lm_list[4][1:]

        fingers = detector.fingers_up()

        if fingers:
            # 1. Check for enable/disable gestures first
            if fingers == FINGER_CONFIG["enable_control"]:
                if not control_enabled:
                    control_enabled = True
                    play_sound_enable()
            elif fingers == FINGER_CONFIG["disable_control"]:
                if control_enabled:
                    control_enabled = False
                    play_sound_disable()
                    if dragging:
                        stop_drag()
                        dragging = False

            angle = detector.get_hand_angle()
            is_valid_orientation = (55 <= angle <= 125)

            # Determine color and screen message
            if not control_enabled:
                rect_color = (128, 128, 128)  # Gray
            elif is_valid_orientation:
                rect_color = (255, 0, 255)  # Magenta
            else:
                rect_color = (0, 0, 255)  # Red

            cv2.rectangle(
                img,
                (FRAME_R, FRAME_R),
                (W_CAM - FRAME_R, H_CAM - FRAME_R),
                rect_color,
                2,
            )

            if not control_enabled:
                cv2.putText(img, "CONTROL OFF", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (128, 128, 128), 2)
            elif not is_valid_orientation:
                cv2.putText(img, "Tilted Hand", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            if control_enabled and is_valid_orientation:
                # Update last non-click gesture state to ensure clicks only trigger after moving
                if fingers != FINGER_CONFIG["left_click"] and fingers != FINGER_CONFIG["right_click"]:
                    if fingers == FINGER_CONFIG["move"]:
                        last_non_click_gesture = "move"
                    elif fingers in [FINGER_CONFIG["scroll_up"], FINGER_CONFIG["scroll_down"], FINGER_CONFIG["drag"]]:
                        last_non_click_gesture = None

                # Move — only index finger up, hand not fully open
                if fingers != FINGER_CONFIG["all_up"] and fingers == FINGER_CONFIG["move"]:
                    x3 = np.interp(x1, (FRAME_R, W_CAM - FRAME_R), (0, w_scr))
                    y3 = np.interp(y1, (FRAME_R, H_CAM - FRAME_R), (0, h_scr))
                    c_loc_x = p_loc_x + (x3 - p_loc_x) / SMOOTHENING
                    c_loc_y = p_loc_y + (y3 - p_loc_y) / SMOOTHENING
                    _mouse.position = (c_loc_x, c_loc_y)
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    p_loc_x, p_loc_y = c_loc_x, c_loc_y

                # Left click (one-time trigger, only allowed if transition from "move")
                if fingers == FINGER_CONFIG["left_click"]:
                    if last_non_click_gesture == "move":
                        if not left_clicked:
                            _mouse.click(Button.left)
                            left_clicked = True
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                else:
                    left_clicked = False

                # Right click (one-time trigger, only allowed if transition from "move")
                if fingers == FINGER_CONFIG["right_click"]:
                    if last_non_click_gesture == "move":
                        if not right_clicked:
                            _mouse.click(Button.right)
                            right_clicked = True
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                else:
                    right_clicked = False

                # Drag — thumb only
                if fingers == FINGER_CONFIG["drag"]:
                    if not dragging:
                        start_drag()
                        dragging = True
                    x4 = np.interp(x2, (FRAME_R, W_CAM - FRAME_R), (0, w_scr))
                    y4 = np.interp(y2, (FRAME_R, H_CAM - FRAME_R), (0, h_scr))
                    c_loc_x = p_loc_x + (x4 - p_loc_x) / SMOOTHENING
                    c_loc_y = p_loc_y + (y4 - p_loc_y) / SMOOTHENING
                    _mouse.position = (c_loc_x, c_loc_y)
                    cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
                    p_loc_x, p_loc_y = c_loc_x, c_loc_y
                else:
                    if dragging:
                        stop_drag()
                        dragging = False

                # Scroll Up — index + middle + ring fingers up
                if fingers == FINGER_CONFIG["scroll_up"]:
                    scroll_counter += 1
                    if scroll_counter % 3 == 0:
                        _mouse.scroll(0, 1)

                # Scroll Down — 4 fingers up (except thumb)
                elif fingers == FINGER_CONFIG["scroll_down"]:
                    scroll_counter += 1
                    if scroll_counter % 3 == 0:
                        _mouse.scroll(0, -1)
                else:
                    scroll_counter = 0
            else:
                left_clicked = False
                right_clicked = False
                scroll_counter = 0

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Mouse Controller (pynput)", img)

        key = cv2.waitKey(1)
        if key & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
