"""Standalone test script for palm orientation detection (front vs. back)."""

import cv2
from hand_tracker import HandDetector


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

    detector = HandDetector(max_hands=1)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Empty camera frame — skipping.")
            continue

        # Flip horizontally for selfie-view, then convert to BGR (HandDetector expects BGR, processes RGB internally)
        image = cv2.flip(image, 1)
        image = detector.find_hands(image, draw=True)

        if detector.results and detector.results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(detector.results.multi_hand_landmarks):
                turned = is_hand_turned(detector.results, hand_landmarks.landmark)
                hand_type = detector.results.multi_handedness[i].classification[0].label
                label = f"{hand_type} - {'Palm Front' if turned else 'Palm Back'}"
                color = (0, 255, 0) if turned else (0, 0, 255)
                cv2.putText(image, label, (40, 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow("Palm Orientation Test", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
