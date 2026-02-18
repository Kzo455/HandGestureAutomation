"""Standalone test script for palm orientation detection (front vs. back)."""

import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils



# Orientation helpers
def is_hand_flipped(landmarks):
    """Return True if the palm is facing up (middle finger tip above wrist)."""
    wrist = landmarks[0]
    middle_finger_tip = landmarks[12]
    return middle_finger_tip.y < wrist.y


def is_hand_turned(results, landmarks):
    """Return True if the hand is turned so the back faces the camera."""
    index_bottom = landmarks[5]
    pinky_bottom = landmarks[17]
    flipped = is_hand_flipped(landmarks)
    hand_type = results.multi_handedness[0].classification[0].label

    if index_bottom.x > pinky_bottom.x and hand_type == "Right":
        turned = True
    elif index_bottom.x < pinky_bottom.x and hand_type == "Left":
        turned = True
    else:
        turned = False

    return turned ^ flipped



# Main loop
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 400)
    cap.set(4, 400)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Empty camera frame — skipping.")
            continue

        # Flip horizontally for selfie-view, then convert to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                turned = is_hand_turned(results, hand_landmarks.landmark)
                label = "Palm Front" if turned else "Palm Back"
                color = (0, 255, 0) if turned else (0, 0, 255)
                cv2.putText(image, label, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow("Palm Orientation Test", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
