import cv2
import mediapipe as mp

"""
Actual working version of Rock Paper Scissors
Paper is just a plain hand
Scissors are like a peace sign
Rock is supposed to be shown phalanges forward
"""


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Extract the landmarks of the hand
            landmarks = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            # Determine which fingers are bent
            fingers = []
            for finger in range(5):
                if finger == 0:
                    if landmarks[4][1] < landmarks[3][1] and landmarks[3][1] < landmarks[2][1] and landmarks[2][1] < landmarks[1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    finger_tip = 4 * finger - 1
                    finger_base = finger_tip - 2
                    if landmarks[finger_tip][1] < landmarks[finger_base][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

            # Determine the gesture based on the bent fingers
            if fingers == [1, 1, 1, 1, 0]:
                gesture = "scissors"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "paper"
            elif fingers == [0, 0, 0, 0, 0]:
                gesture = "rock"
            else:
                gesture = "unknown"

            # Draw the gesture label on the image
            cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw the hand landmarks and connections on the image
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    # Show the image
    cv2.imshow("Output", image)
    cv2.waitKey(1)
