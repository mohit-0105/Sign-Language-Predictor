import cv2
import mediapipe as mp
import csv
import os
import time

# ---- CONFIG ----
SIGNS = ["hello", "thank_you", "yes", "no", "please"]  # edit this list as needed
SAMPLES_PER_SIGN = 150
CSV_FILE = "hand_landmarks.csv"

# ---- SETUP MEDIAPIPE ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # single hand for now, keeps data collection simpler
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ---- CREATE CSV WITH HEADER IF IT DOESN'T EXIST ----
if not os.path.exists(CSV_FILE):
    header = ["label"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Instructions:")
print(" - For each sign, get your hand ready, then press SPACE to start recording.")
print(" - Press 'q' at any time to quit early.")
print(" - Press 's' to skip a sign.\n")

with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    for sign in SIGNS:
        print(f"\n=== Sign: '{sign}' ===")
        print("Position your hand, then press SPACE to begin recording this sign.")

        # Wait for user to press SPACE before starting this sign
        ready = False
        while not ready:
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            cv2.putText(frame, f"Sign: {sign} | Press SPACE to record", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Data Collection - Press Q to Quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                ready = True
            elif key == ord('s'):
                print(f"Skipped '{sign}'")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if not ready:
            continue  # sign was skipped

        # ---- RECORD SAMPLES FOR THIS SIGN ----
        count = 0
        while count < SAMPLES_PER_SIGN:
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # only first detected hand

                # Draw landmarks for visual feedback
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Flatten landmark coordinates into one row
                row = [sign]
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                writer.writerow(row)
                count += 1

            cv2.putText(frame, f"Recording '{sign}': {count}/{SAMPLES_PER_SIGN}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Data Collection - Press Q to Quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"Finished recording '{sign}' ({count} samples).")
        time.sleep(1)  # brief pause before moving to next sign

print("\nData collection complete! Saved to", CSV_FILE)
cap.release()
cv2.destroyAllWindows()