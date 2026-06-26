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

import hand_tracker as htm

pyautogui.FAILSAFE = False


# Constants

W_CAM, H_CAM = 640, 480
FRAME_R = 100        # Active gesture area margin (pixels)
SMOOTHENING = 7      # Cursor smoothing factor

FINGER_CONFIG = {
    "all_up":     [1, 1, 1, 1, 1],
    "all_down":   [0, 0, 0, 0, 0],
    "move":       [0, 1, 0, 0, 0],
    "right_click":[0, 1, 1, 0, 0], # Index + Middle up
    "left_click": [1, 1, 0, 0, 0], # Thumb + Index up
    "drag":       [1, 0, 0, 0, 0],
    "scroll_up":  [0, 1, 1, 1, 0], # Index + Middle + Ring up
    "scroll_down":[0, 1, 1, 1, 1], # 4 fingers up (except thumb)
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
            angle = detector.get_hand_angle()
            is_valid_orientation = (55 <= angle <= 125)

            rect_color = (255, 0, 255) if is_valid_orientation else (0, 0, 255)
            cv2.rectangle(
                img,
                (FRAME_R, FRAME_R),
                (W_CAM - FRAME_R, H_CAM - FRAME_R),
                rect_color,
                2,
            )

            if not is_valid_orientation:
                cv2.putText(img, "Tilted Hand", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            if is_valid_orientation:
                # Move — only index finger up, hand not fully open
                if fingers != FINGER_CONFIG["all_up"] and fingers == FINGER_CONFIG["move"]:
                    x3 = np.interp(x1, (FRAME_R, W_CAM - FRAME_R), (0, w_scr))
                    y3 = np.interp(y1, (FRAME_R, H_CAM - FRAME_R), (0, h_scr))
                    c_loc_x = p_loc_x + (x3 - p_loc_x) / SMOOTHENING
                    c_loc_y = p_loc_y + (y3 - p_loc_y) / SMOOTHENING
                    _mouse.position = (c_loc_x, c_loc_y)
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    p_loc_x, p_loc_y = c_loc_x, c_loc_y

                # Left click (one-time trigger)
                if fingers == FINGER_CONFIG["left_click"]:
                    if not left_clicked:
                        _mouse.click(Button.left)
                        left_clicked = True
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                else:
                    left_clicked = False

                # Right click (one-time trigger)
                if fingers == FINGER_CONFIG["right_click"]:
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
