import os
import pickle
import cv2
import numpy as np

# Suppress protobuf warnings and handle compatibility
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Safe import of mediapipe with protobuf handling
try:
    import google.protobuf
    # Only apply the fix if the old structure exists
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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    print(f"ERROR: {DATA_DIR} directory not found!")
    print("Please run 'collect_imgs.py' first to collect images.")
    exit()

print("Processing images...")
processed_count = 0
skipped_count = 0

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip files in DATA_DIR

    print(f"Processing class: {dir_}")
    class_processed = 0

    for img_path in os.listdir(dir_path):
        img_file = os.path.join(dir_path, img_path)
        
        try:
            img = cv2.imread(img_file)
            if img is None:
                skipped_count += 1
                continue  # Skip unreadable images

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if not results.multi_hand_landmarks:
                skipped_count += 1
                continue  # Skip images without detected hands

            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                data_aux = []
                min_x, min_y = min(x_coords), min(y_coords)
                for x, y in zip(x_coords, y_coords):
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)

                data.append(data_aux)
                labels.append(dir_)
                processed_count += 1
                class_processed += 1
                break  # Process only the first detected hand per image
                
        except Exception as e:
            print(f"  Error processing {img_file}: {e}")
            skipped_count += 1
            continue
    
    print(f"  Processed {class_processed} images")

print("\n" + "="*50)
print(f"Total processed: {processed_count}")
print(f"Total skipped: {skipped_count}")

if processed_count == 0:
    print("\nERROR: No hand landmarks detected!")
    print("Make sure you have collected images using collect_imgs.py")
    exit()

# Save data
print(f"\nSaving to 'data.pickle'...")
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✓ Dataset created successfully!")
print(f"  Samples: {len(data)}")
print(f"  Classes: {len(set(labels))}")