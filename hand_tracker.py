import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    """Detects and tracks hand landmarks using MediaPipe."""

    TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.lm_list = []
        self.results = None

    
    # Detection helpers
    

    def find_hands(self, img, draw=True):
        """Process a BGR frame and optionally draw landmarks. Returns the frame."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, hand_no=0, draw=True):
        """Return a list of [id, cx, cy, cz] for each landmark and a bounding box."""
        x_list, y_list = [], []
        bbox = []
        self.lm_list = []

        if self.results and self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            h, w, _ = img.shape

            for lm_id, lm in enumerate(my_hand.landmark):
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([lm_id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bbox = (x_min, y_min, x_max, y_max)

            if draw:
                cv2.rectangle(
                    img,
                    (x_min - 20, y_min - 20),
                    (x_max + 20, y_max + 20),
                    (0, 255, 0),
                    2,
                )

        return self.lm_list, bbox

    
    # Finger-state helpers
    def fingers_up(self):
        """Return a list of 5 booleans (1/0) indicating which fingers are extended."""
        if not self.lm_list:
            return []

        fingers = []
        hand_type = self.results.multi_handedness[0].classification[0].label

        # Thumb — direction depends on hand type
        thumb_tip_x = self.lm_list[self.TIP_IDS[0]][1]
        thumb_ip_x = self.lm_list[self.TIP_IDS[0] - 1][1]
        if (thumb_tip_x > thumb_ip_x and hand_type == "Left") or (
            thumb_tip_x < thumb_ip_x and hand_type == "Right"
        ):
            fingers.append(1)
        else:
            fingers.append(0)

        # Four fingers — compare tip Y to PIP Y
        for idx in range(1, 5):
            tip_y = self.lm_list[self.TIP_IDS[idx]][2]
            pip_y = self.lm_list[self.TIP_IDS[idx] - 2][2]
            fingers.append(1 if tip_y < pip_y else 0)

        return fingers

    def fingers_half_closed(self):
        """Return a list of 5 booleans indicating which fingers are half-closed."""
        if not self.lm_list:
            return []

        fingers = []
        hand_type = self.results.multi_handedness[0].classification[0].label

        # Thumb
        tip_x = self.lm_list[self.TIP_IDS[0]][1]
        ip_x = self.lm_list[self.TIP_IDS[0] - 1][1]
        mcp_x = self.lm_list[self.TIP_IDS[0] - 2][1]
        if (tip_x < ip_x and tip_x > mcp_x and hand_type == "Left") or (
            tip_x > ip_x and tip_x < mcp_x and hand_type == "Right"
        ):
            fingers.append(1)
        else:
            fingers.append(0)

        # Four fingers — tip between DIP and PIP
        for idx in range(1, 5):
            tip_y = self.lm_list[self.TIP_IDS[idx]][2]
            dip_y = self.lm_list[self.TIP_IDS[idx] - 3][2]
            pip_y = self.lm_list[self.TIP_IDS[idx] - 1][2]
            fingers.append(1 if (dip_y <= tip_y <= pip_y) or (pip_y <= tip_y <= dip_y) else 0)

        return fingers

    
    # Orientation helpers
    def is_hand_flipped(self):
        """Return True if the palm is facing up (middle finger tip below wrist)."""
        landmarks = self.results.multi_hand_landmarks[0].landmark
        return landmarks[12].y > landmarks[0].y

    def is_hand_turned(self):
        """Return True if the hand is turned away from the camera."""
        landmarks = self.results.multi_hand_landmarks[0].landmark
        hand_type = self.results.multi_handedness[0].classification[0].label

        index_bottom = landmarks[5]
        pinky_bottom = landmarks[17]

        if index_bottom.x > pinky_bottom.x and hand_type == "Right":
            return True
        if index_bottom.x < pinky_bottom.x and hand_type == "Left":
            return True
        return False

    
    # Distance helper
    def find_distance(self, p1, p2, img, draw=True, r=10, t=3):
        """Return the pixel distance between two landmarks, the annotated frame, and line info."""
        x1, y1, _ = self.lm_list[p1][1:]
        x2, y2, _ = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]



# Backwards-compatible alias so existing imports keep working
handDetector = HandDetector


# Quick demo / standalone test
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    p_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lm_list, _ = detector.find_position(img)
        detector.fingers_up()

        if lm_list:
            print(lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hand Tracker", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
