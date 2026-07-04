import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter

# ---- LOAD TRAINED MODEL ----
model = joblib.load("sign_model.pkl")

# ---- SETUP MEDIAPIPE ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ---- STABILITY BUFFER ----
# Keeps last N predictions so we only show a sign once it's consistent,
# avoiding flickery/jumpy output
BUFFER_SIZE = 10
CONFIDENCE_THRESHOLD = 0.7   # how many of the last N predictions must agree
prediction_buffer = deque(maxlen=BUFFER_SIZE)

stable_prediction = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_prediction = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Flatten landmarks into feature vector (same format as training data)
        row = []
        for lm in hand_landmarks.landmark:
            row += [lm.x, lm.y, lm.z]

        # Predict
        row_np = np.array(row).reshape(1, -1)
        current_prediction = model.predict(row_np)[0]

        prediction_buffer.append(current_prediction)
    else:
        prediction_buffer.append(None)  # no hand detected

    # ---- CHECK STABILITY ----
    if len(prediction_buffer) == BUFFER_SIZE:
        counts = Counter(prediction_buffer)
        most_common, freq = counts.most_common(1)[0]
        if most_common is not None and freq / BUFFER_SIZE >= CONFIDENCE_THRESHOLD:
            stable_prediction = most_common
        else:
            stable_prediction = ""

    # ---- DISPLAY ----
    display_text = stable_prediction if stable_prediction else "..."
    cv2.putText(frame, f"Sign: {display_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()