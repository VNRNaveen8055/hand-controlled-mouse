import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# screen size
screen_w, screen_h = pyautogui.size()

# camera size
cam_w, cam_h = 640, 480

pyautogui.FAILSAFE = False

# mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

# to smooth cursor movement
prev_x, prev_y = 0, 0

import time
last_click_time = 0


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # get all landmark positions
            lm = []
            for point in hand.landmark:
                lm.append((int(point.x * cam_w), int(point.y * cam_h)))

            # index finger tip = lm[8], thumb tip = lm[4]
            ix, iy = lm[8]   # index tip - controls cursor
            tx, ty = lm[4]   # thumb tip

            # move cursor (map camera coords to screen coords)
            screen_x = np.interp(ix, [50, cam_w - 50], [0, screen_w])
            screen_y = np.interp(iy, [50, cam_h - 50], [0, screen_h])

            # smooth the movement a little
            smooth_x = int(prev_x + (screen_x - prev_x) / 4)
            smooth_y = int(prev_y + (screen_y - prev_y) / 4)

            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

            # left click - thumb + index pinch
            if get_distance(lm[4], lm[8]) < 35:
                if time.time() - last_click_time > 0.5:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.putText(frame, "Left Click", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # right click - thumb + middle finger pinch
            elif get_distance(lm[4], lm[12]) < 35:
                if time.time() - last_click_time > 0.5:
                    pyautogui.rightClick()
                    last_click_time = time.time()
                    cv2.putText(frame, "Right Click", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # scroll up - index + middle fingers up, move hand up
            elif lm[8][1] < lm[6][1] and lm[12][1] < lm[10][1]:
                if lm[8][1] < lm[7][1] - 10:
                    pyautogui.scroll(3)
                    cv2.putText(frame, "Scroll Up", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                elif lm[8][1] > lm[7][1] + 10:
                    pyautogui.scroll(-3)
                    cv2.putText(frame, "Scroll Down", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

    cv2.imshow("Hand Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()