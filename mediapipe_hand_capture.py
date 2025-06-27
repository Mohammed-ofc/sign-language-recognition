import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

label = 'E'  # Change this for each gesture
data = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("[INFO] Showing webcam. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img = cv2.flip(frame, 1)

    # 🔽 Add this line to show current label collecting
    cv2.putText(img, f"Collecting: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)


    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            # Only append if no NaNs
            if not any(pd.isnull(features)):
                features.append(label)
                data.append(features)

    cv2.imshow("Hand Gesture Capture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(data)
df = pd.DataFrame(data)

# If file exists, append without header
if os.path.exists("data/sign_data.csv"):
    df.to_csv("data/sign_data.csv", mode='a', index=False, header=False)
else:
    df.to_csv("data/sign_data.csv", index=False, header=False)

print("[INFO] Data saved to data/sign_data.csv")

