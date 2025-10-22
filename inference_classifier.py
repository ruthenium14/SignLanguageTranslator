import pickle
import cv2
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Safe protobuf handling
try:
    import google.protobuf
    if hasattr(google.protobuf, 'internal'):
        try:
            import google.protobuf.internal.builder as builder
            if hasattr(builder, 'GetPrototype'):
                builder.GetPrototype = lambda descriptor: None
        except (AttributeError, ImportError):
            pass
except ImportError:
    pass

import mediapipe as mp

# Load trained model
model_path = './model.p'
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file 'model.p' not found! Run train_classifier.py first.")

model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Dynamic labels dictionary
labels_dict = {
    0: 'Hello', 1: 'Iloveyou', 2: 'Yes', 3: 'No', 4: 'Please',
    5: 'Thanks', 6: 'A', 7: 'B', 8: 'C', 9: 'D',
    10: 'E', 11: 'F', 12: 'G', 13: 'H', 14: 'I',
    15: 'K', 16: 'L', 17: 'M', 18: 'N', 19: 'O',
    20: 'P', 21: 'Q', 22: 'R', 23: 'S', 24: 'T',
    25: 'U', 26: 'V', 27: 'W', 28: 'X', 29:'Y'
    }

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("Starting hand sign detection...")
print("Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract normalized coordinates
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            data_aux = []
            min_x, min_y = min(x_coords), min(y_coords)
            for x, y in zip(x_coords, y_coords):
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)

            # Bounding box
            x1 = max(0, int(min(x_coords) * W) - 10)
            y1 = max(0, int(min(y_coords) * H) - 10)
            x2 = min(W, int(max(x_coords) * W) + 10)
            y2 = min(H, int(max(y_coords) * H) + 10)

            # Prediction
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), '?')
                
                # Display
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, "Error", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # No hand detected
        cv2.putText(frame, "Show hand to camera", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Hand Sign Detection', frame)

    # Exit on pressing Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")