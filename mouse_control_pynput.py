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
from pynput.keyboard import Key, Controller as KeyboardController
_keyboard = KeyboardController()
import sounddevice as sd

import hand_tracker as htm

pyautogui.FAILSAFE = False

# Sound helpers
def play_sound_enable():
    fs = 44100
    d1, d2 = 0.07, 0.12
    t1 = np.linspace(0, d1, int(fs * d1), endpoint=False)
    t2 = np.linspace(0, d2, int(fs * d2), endpoint=False)
    wave1 = np.sin(2 * np.pi * 400 * t1)
    wave2 = np.sin(2 * np.pi * 700 * t2)
    fade1 = np.ones_like(wave1)
    fade1[-500:] = np.linspace(1, 0, 500)
    fade2 = np.ones_like(wave2)
    fade2[-1000:] = np.linspace(1, 0, 1000)
    wave = np.concatenate([wave1 * fade1, wave2 * fade2])
    sd.play(wave * 0.35, fs)

def play_sound_disable():
    fs = 44100
    d1, d2 = 0.07, 0.12
    t1 = np.linspace(0, d1, int(fs * d1), endpoint=False)
    t2 = np.linspace(0, d2, int(fs * d2), endpoint=False)
    wave1 = np.sin(2 * np.pi * 700 * t1)
    wave2 = np.sin(2 * np.pi * 400 * t2)
    fade1 = np.ones_like(wave1)
    fade1[-500:] = np.linspace(1, 0, 500)
    fade2 = np.ones_like(wave2)
    fade2[-1000:] = np.linspace(1, 0, 1000)
    wave = np.concatenate([wave1 * fade1, wave2 * fade2])
    sd.play(wave * 0.35, fs)


# Constants

W_CAM, H_CAM = 640, 480
FRAME_R = 100        # Active gesture area margin (pixels)
SMOOTHENING = 7      # Cursor smoothing factor

FINGER_CONFIG = {
    "all_up":          [1, 1, 1, 1, 1],
    "all_down":        [0, 0, 0, 0, 0],
    "move":            [0, 1, 0, 0, 0],
    "right_click":     [0, 1, 1, 0, 0], # Index + Pinky up
    "left_click":      [1, 1, 0, 0, 0], # Thumb + Index up
    "scroll_up":       [1, 1, 0, 0, 1], # Index + Middle + Ring up
    "scroll_down":     [0, 1, 0, 0, 1], # Index + Middle up
    "volume_up":       [0, 1, 1, 1, 0], # Index + Middle + Ring up (3 fingers)
    "volume_down":     [0, 1, 1, 1, 1], # 4 fingers up (except thumb)
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
    import subprocess
    import platform
    if platform.system() == "Linux":
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", "/dev/video0", "-c", "exposure_dynamic_framerate=0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    cap = cv2.VideoCapture(0)
    cap.set(3, W_CAM)
    cap.set(4, H_CAM)

    detector = htm.HandDetector(max_hands=1)
    w_scr, h_scr = pyautogui.size()

    p_time = 0
    p_loc_x, p_loc_y = 0, 0
    c_loc_x, c_loc_y = 0, 0
    dragging = False
    left_click_start_time = None
    right_click_start_time = None
    volume_start_time = None
    last_volume_change_time = 0
    enable_start_time = None
    disable_start_time = None

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
            # 1. Check for enable/disable gestures first with 150ms verification delay
            if fingers == FINGER_CONFIG["enable_control"]:
                disable_start_time = None
                if not control_enabled:
                    if enable_start_time is None:
                        enable_start_time = time.time()
                    if time.time() - enable_start_time >= 0.15:
                        control_enabled = True
                        play_sound_enable()
                        enable_start_time = None
            elif fingers == FINGER_CONFIG["disable_control"]:
                enable_start_time = None
                if control_enabled:
                    if disable_start_time is None:
                        disable_start_time = time.time()
                    if time.time() - disable_start_time >= 0.15:
                        control_enabled = False
                        play_sound_disable()
                        disable_start_time = None
                        if dragging:
                            stop_drag()
                            dragging = False
            else:
                enable_start_time = None
                disable_start_time = None

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
                    elif fingers in [FINGER_CONFIG["scroll_up"], FINGER_CONFIG["scroll_down"]]:
                        last_non_click_gesture = None

                # Move — only index finger up, hand not fully open
                if fingers != FINGER_CONFIG["all_up"] and fingers == FINGER_CONFIG["move"]:
                    x3 = np.interp(x1, (FRAME_R, W_CAM - FRAME_R), (0, w_scr))
                    y3 = np.interp(y1, (FRAME_R, H_CAM - FRAME_R), (0, h_scr))
                    c_loc_x = p_loc_x + (x3 - p_loc_x) / SMOOTHENING
                    c_loc_y = p_loc_y + (y3 - p_loc_y) / SMOOTHENING
                    _mouse.position = (int(c_loc_x), int(c_loc_y))
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                    p_loc_x, p_loc_y = c_loc_x, c_loc_y

                # Left click and Drag/Hold (merged)
                if fingers == FINGER_CONFIG["left_click"]:
                    thumb_index_angle = detector.get_thumb_index_angle()
                    if len(detector.lm_list) >= 9:
                        x4, y4 = detector.lm_list[4][1:3]
                        x5, y5 = detector.lm_list[5][1:3]
                        x8, y8 = detector.lm_list[8][1:3]
                        cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 2)
                        cv2.line(img, (x8, y8), (x5, y5), (0, 0, 255), 2)
                        cv2.putText(img, f"Angle: {int(thumb_index_angle)} deg", (x5 + 10, y5 + 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

                    if 60 <= thumb_index_angle <= 120:
                        if left_click_start_time is None:
                            left_click_start_time = time.time()
                        
                        # Only start drag if held for > 300ms
                        if time.time() - left_click_start_time > 0.3:
                            if not dragging and last_non_click_gesture == "move":
                                _mouse.press(Button.left)
                                dragging = True
                    else:
                        if left_click_start_time is not None:
                            if dragging:
                                _mouse.release(Button.left)
                                dragging = False
                            else:
                                # Only click if gesture was held for at least 100ms (filters out noise)
                                if time.time() - left_click_start_time >= 0.1:
                                    if last_non_click_gesture == "move":
                                        _mouse.click(Button.left)
                            left_click_start_time = None

                    if dragging:
                        x3 = np.interp(x1, (FRAME_R, W_CAM - FRAME_R), (0, w_scr))
                        y3 = np.interp(y1, (FRAME_R, H_CAM - FRAME_R), (0, h_scr))
                        c_loc_x = p_loc_x + (x3 - p_loc_x) / SMOOTHENING
                        c_loc_y = p_loc_y + (y3 - p_loc_y) / SMOOTHENING
                        _mouse.position = (int(c_loc_x), int(c_loc_y))
                        p_loc_x, p_loc_y = c_loc_x, c_loc_y

                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                else:
                    if left_click_start_time is not None:
                        if dragging:
                            _mouse.release(Button.left)
                            dragging = False
                        else:
                            # Only click if gesture was held for at least 100ms (filters out noise)
                            if time.time() - left_click_start_time >= 0.1:
                                if last_non_click_gesture == "move":
                                    _mouse.click(Button.left)
                        left_click_start_time = None

                # Right click (with 100ms noise filter, triggers on release)
                if fingers == FINGER_CONFIG["right_click"]:
                    if right_click_start_time is None:
                        right_click_start_time = time.time()
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                else:
                    if right_click_start_time is not None:
                        duration = time.time() - right_click_start_time
                        if duration >= 0.1:
                            if last_non_click_gesture == "move":
                                _mouse.click(Button.right)
                        right_click_start_time = None

                # Scroll Up — index + middle + ring fingers up
                if fingers == FINGER_CONFIG["scroll_up"]:
                    scroll_counter += 1
                    if scroll_counter % 3 == 0:
                        _mouse.scroll(0, 1)

                # Scroll Down — Index + Middle up
                elif fingers == FINGER_CONFIG["scroll_down"]:
                    scroll_counter += 1
                    if scroll_counter % 3 == 0:
                        _mouse.scroll(0, -1)
                else:
                    scroll_counter = 0

                # Volume Control (with 100ms noise filter and 150ms throttling)
                if fingers in [FINGER_CONFIG["volume_up"], FINGER_CONFIG["volume_down"]]:
                    if volume_start_time is None:
                        volume_start_time = time.time()
                    
                    if time.time() - volume_start_time >= 0.1:  # 100ms noise filter
                        if time.time() - last_volume_change_time >= 0.15:  # Throttle volume adjustments
                            if fingers == FINGER_CONFIG["volume_up"]:
                                _keyboard.press(Key.media_volume_up)
                                _keyboard.release(Key.media_volume_up)
                            else:
                                _keyboard.press(Key.media_volume_down)
                                _keyboard.release(Key.media_volume_down)
                            last_volume_change_time = time.time()
                else:
                    volume_start_time = None
            else:
                left_clicked = False
                right_clicked = False
                scroll_counter = 0
                volume_start_time = None
        else:
            left_click_start_time = None
            right_click_start_time = None
            volume_start_time = None
            enable_start_time = None
            disable_start_time = None

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
