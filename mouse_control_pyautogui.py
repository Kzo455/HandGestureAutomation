"""Hand-gesture mouse controller backed by pyautogui."""

import sys
from unittest.mock import MagicMock
# Mock tkinter to prevent pyautogui/pymsgbox import crashes on headless Linux systems
mock_tk = MagicMock()
mock_tk.TkVersion = 8.6
sys.modules['tkinter'] = mock_tk
sys.modules['_tkinter'] = MagicMock()

import ctypes
import time

import cv2
import numpy as np
import pyautogui

import hand_tracker as htm

pyautogui.FAILSAFE = False


# Constants

FRAME_R = 100    # Active gesture area margin (pixels)
SMOOTHENING = 6  # Cursor smoothing factor

FINGER_CONFIG = {
    "move":       [0, 1, 0, 0, 0],
    "left_click": [1, 1, 0, 0, 0], # Thumb + Index up
    "right_click":[0, 1, 1, 0, 0], # Index + Middle up
    "drag":       [0, 1, 0, 0, 1],
    "scroll_up":  [0, 1, 1, 1, 0], # Index + Middle + Ring up
    "scroll_down":[0, 1, 1, 1, 1], # 4 fingers up (except thumb)
}



# Mouse controller class
class MouseController:
    """Controls the mouse using hand-gesture coordinates."""

    def __init__(self, smoothening: int = SMOOTHENING):
        self.smoothening = smoothening
        self.p_loc_x, self.p_loc_y = 0, 0
        self.c_loc_x, self.c_loc_y = 0, 0
        self.dragging = False

    
    # Drag helpers
    def start_drag(self):
        pyautogui.mouseDown()

    def stop_drag(self):
        pyautogui.mouseUp()

    
    # Actions
    def move(self, img, x1, y1, frame_r, w_cam, h_cam, w_scr, h_scr):
        """Smooth-move the cursor to the position mapped from the camera frame."""
        x3 = np.interp(x1, (frame_r, w_cam - frame_r), (0, w_scr))
        y3 = np.interp(y1, (frame_r, h_cam - frame_r), (0, h_scr))
        self.c_loc_x = self.p_loc_x + (x3 - self.p_loc_x) / self.smoothening
        self.c_loc_y = self.p_loc_y + (y3 - self.p_loc_y) / self.smoothening
        pyautogui.moveTo(int(self.c_loc_x), int(self.c_loc_y))
        self.p_loc_x, self.p_loc_y = self.c_loc_x, self.c_loc_y
        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

    def left_click(self, img, x, y):
        pyautogui.click()
        cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)

    def right_click(self, img, x, y):
        pyautogui.rightClick()
        cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)

    def drag(self, img, detector, x1, y1, frame_r, w_cam, h_cam, w_scr, h_scr,
             finger1=4, finger2=12, finger3=16):
        length1, img, line_info1 = detector.find_distance(finger1, finger2, img)
        length2, img, line_info2 = detector.find_distance(finger1, finger3, img)

        if length1 < 30 and length2 < 30:
            if not self.dragging:
                self.start_drag()
                self.dragging = True

            x3 = np.interp(x1, (frame_r, w_cam - frame_r), (0, w_scr))
            y3 = np.interp(y1, (frame_r, h_cam - frame_r), (0, h_scr))
            self.c_loc_x = self.p_loc_x + (x3 - self.p_loc_x) / self.smoothening
            self.c_loc_y = self.p_loc_y + (y3 - self.p_loc_y) / self.smoothening
            pyautogui.moveTo(int(self.c_loc_x), int(self.c_loc_y))
            self.p_loc_x, self.p_loc_y = self.c_loc_x, self.c_loc_y
            cv2.circle(img, (line_info1[4], line_info1[5]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (line_info2[4], line_info2[5]), 15, (0, 255, 0), cv2.FILLED)
        else:
            if self.dragging:
                self.stop_drag()
            self.dragging = False

    def scroll(self, img, detector, y1, frame_r, h_cam, h_scr,
               finger1=8, finger2=12, finger3=16):
        length1, img, line_info1 = detector.find_distance(finger1, finger2, img)
        length2, img, line_info2 = detector.find_distance(finger2, finger3, img)

        if length1 < 30 and length2 < 30:
            y3 = np.interp(y1, (frame_r, h_cam - frame_r), (0, h_scr))
            scroll_y = int(np.clip((y3 - self.p_loc_y) / 5, -5, 5))
            if scroll_y != 0:
                pyautogui.scroll(scroll_y)
            cv2.circle(img, (line_info1[4], line_info1[5]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (line_info2[4], line_info2[5]), 15, (0, 255, 0), cv2.FILLED)



# Utility
def get_screen_resolution():
    """Return the physical screen resolution, DPI-aware on Windows, with fallback on Linux/Mac."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except AttributeError:
        # Fallback using pyautogui for non-Windows platforms
        return pyautogui.size()



# Main loop
def main():
    p_time = 0

    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector(max_hands=1)
    w_scr, h_scr = get_screen_resolution()
    mouse_ctrl = MouseController(SMOOTHENING)
    scroll_counter = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h_cam, w_cam, _ = img.shape

        img = detector.find_hands(img)
        lm_list, _ = detector.find_position(img)

        if lm_list:
            x1, y1, z1 = lm_list[8][1:]
            fingers = detector.fingers_up()
            folded = detector.fingers_half_closed()

            if fingers and -z1 > 0.03:
                angle = detector.get_hand_angle()
                is_valid_orientation = (55 <= angle <= 125)

                rect_color = (255, 0, 255) if is_valid_orientation else (0, 0, 255)
                cv2.rectangle(
                    img,
                    (FRAME_R, FRAME_R),
                    (w_cam - FRAME_R, h_cam - FRAME_R),
                    rect_color,
                    2,
                )

                if not is_valid_orientation:
                    cv2.putText(img, "Tilted Hand", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                if is_valid_orientation:
                    if fingers == FINGER_CONFIG["move"] and not folded[1]:
                        mouse_ctrl.move(img, x1, y1, FRAME_R, w_cam, h_cam, w_scr, h_scr)
                        scroll_counter = 0

                    elif fingers == FINGER_CONFIG["left_click"]:
                        mouse_ctrl.left_click(img, x1, y1)
                        scroll_counter = 0

                    elif fingers == FINGER_CONFIG["right_click"]:
                        mouse_ctrl.right_click(img, x1, y1)
                        scroll_counter = 0

                    elif fingers == FINGER_CONFIG["drag"]:
                        thumb_tip = lm_list[4][1:]
                        mouse_ctrl.drag(
                            img, detector,
                            thumb_tip[0], thumb_tip[1],
                            FRAME_R, w_cam, h_cam, w_scr, h_scr,
                        )
                        scroll_counter = 0

                    elif fingers == FINGER_CONFIG["scroll_up"]:
                        scroll_counter += 1
                        if scroll_counter % 3 == 0:
                            pyautogui.scroll(1)

                    elif fingers == FINGER_CONFIG["scroll_down"]:
                        scroll_counter += 1
                        if scroll_counter % 3 == 0:
                            pyautogui.scroll(-1)
                    else:
                        scroll_counter = 0

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Mouse Controller (pyautogui)", img)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
