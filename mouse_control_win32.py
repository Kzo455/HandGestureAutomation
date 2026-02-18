"""Hand-gesture mouse controller backed by the Win32 API (pywin32)."""

import ctypes
import time

import cv2
import numpy as np
import win32api
import win32con

import hand_tracker as htm


# Constants

FRAME_R = 100    # Active gesture area margin (pixels)
SMOOTHENING = 6  # Cursor smoothing factor

FINGER_CONFIG = {
    "all_up":     [1, 1, 1, 1, 1],
    "all_down":   [0, 0, 0, 0, 0],
    "move":       [0, 1, 0, 0, 0],
    "left_click": [0, 1, 0, 0, 0],   # detected via half-closed fingers
    "right_click":[0, 1, 1, 0, 0],
    "drag":       [0, 1, 0, 0, 1],
    "scroll":     [0, 1, 1, 1, 0],
}



# Mouse controller class
class MouseController:
    """Controls the mouse using hand-gesture coordinates via the Win32 API."""

    def __init__(self, smoothening: int = SMOOTHENING):
        self.smoothening = smoothening
        self.p_loc_x, self.p_loc_y = 0, 0  # previous cursor position
        self.c_loc_x, self.c_loc_y = 0, 0  # current cursor position
        self.dragging = False

    
    # Drag helpers
    def start_drag(self):
        """Press and hold the left mouse button."""
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)

    def stop_drag(self):
        """Release the left mouse button."""
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    
    # Actions
    def move(self, img, x1, y1, frame_r, w_cam, h_cam, w_scr, h_scr):
        """Smooth-move the cursor to the position mapped from the camera frame."""
        x3 = np.interp(x1, (frame_r, w_cam - frame_r), (0, w_scr))
        y3 = np.interp(y1, (frame_r, h_cam - frame_r), (0, h_scr))
        self.c_loc_x = self.p_loc_x + (x3 - self.p_loc_x) / self.smoothening
        self.c_loc_y = self.p_loc_y + (y3 - self.p_loc_y) / self.smoothening

        try:
            win32api.SetCursorPos((int(self.c_loc_x), int(self.c_loc_y)))
        except Exception as exc:
            print(f"SetCursorPos error ({self.c_loc_x:.0f}, {self.c_loc_y:.0f}): {exc}")

        self.p_loc_x, self.p_loc_y = self.c_loc_x, self.c_loc_y
        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

    def left_click(self, img, x, y):
        """Simulate a left mouse click at the current cursor position."""
        cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.08)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    def right_click(self, img, detector, finger1=8, finger2=12):
        """Simulate a right mouse click when two fingertips are close together."""
        length, img, line_info = detector.find_distance(finger1, finger2, img)
        if length < 30:
            cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)

    def drag(self, img, detector, x1, y1, frame_r, w_cam, h_cam, w_scr, h_scr,
             finger1=4, finger2=12, finger3=16):
        """Hold the left button and move the cursor to simulate a drag."""
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
            win32api.SetCursorPos((int(self.c_loc_x), int(self.c_loc_y)))
            self.p_loc_x, self.p_loc_y = self.c_loc_x, self.c_loc_y
            cv2.circle(img, (line_info1[4], line_info1[5]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (line_info2[4], line_info2[5]), 15, (0, 255, 0), cv2.FILLED)
        else:
            if self.dragging:
                self.stop_drag()
            self.dragging = False

    def scroll(self, img, detector, y1, frame_r, h_cam, h_scr,
               finger1=8, finger2=12, finger3=16):
        """Simulate vertical scrolling based on finger position."""
        length1, img, line_info1 = detector.find_distance(finger1, finger2, img)
        length2, img, line_info2 = detector.find_distance(finger2, finger3, img)

        if length1 < 30 and length2 < 30:
            y3 = np.interp(y1, (frame_r, h_cam - frame_r), (0, h_scr))
            scroll_y = int(np.clip((y3 - self.p_loc_y) / 5, -3, 3))
            if scroll_y != 0:
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, scroll_y * 120)
            cv2.circle(img, (line_info1[4], line_info1[5]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (line_info2[4], line_info2[5]), 15, (0, 255, 0), cv2.FILLED)



# Utility
def get_screen_resolution():
    """Return the physical screen resolution, DPI-aware."""
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)



# Main loop
def main():
    p_time = 0

    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector(max_hands=1)
    w_scr, h_scr = get_screen_resolution()
    mouse_ctrl = MouseController(SMOOTHENING)

    while True:
        success, img = cap.read()
        if not success:
            break

        h_cam, w_cam, _ = img.shape
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lm_list, _ = detector.find_position(img)

        hand_in_correct_orientation = False
        x1 = y1 = z1 = 0

        if lm_list:
            x1, y1, z1 = lm_list[8][1:]
            hand_in_correct_orientation = not (
                detector.is_hand_turned() or detector.is_hand_flipped()
            )

        fingers = detector.fingers_up()
        folded = detector.fingers_half_closed()

        if fingers and -z1 > 0.03 and hand_in_correct_orientation:
            cv2.rectangle(
                img,
                (FRAME_R, FRAME_R),
                (w_cam - FRAME_R, h_cam - FRAME_R),
                (255, 0, 255),
                2,
            )

            if fingers == FINGER_CONFIG["move"] and not folded[1]:
                mouse_ctrl.move(img, x1, y1, FRAME_R, w_cam, h_cam, w_scr, h_scr)

            elif folded == FINGER_CONFIG["left_click"]:
                mouse_ctrl.left_click(img, x1, y1)

            elif fingers == FINGER_CONFIG["right_click"]:
                mouse_ctrl.right_click(img, detector)

            elif fingers == FINGER_CONFIG["drag"]:
                thumb_tip = lm_list[4][1:]
                mouse_ctrl.drag(
                    img, detector,
                    thumb_tip[0], thumb_tip[1],
                    FRAME_R, w_cam, h_cam, w_scr, h_scr,
                )

            elif fingers == FINGER_CONFIG["scroll"]:
                middle_finger_tip_y = lm_list[12][2]
                mouse_ctrl.scroll(img, detector, middle_finger_tip_y, FRAME_R, h_cam, h_scr)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Mouse Controller (Win32)", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
