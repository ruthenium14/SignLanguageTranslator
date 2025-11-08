import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ============= UPDATE YOUR LABELS =============
class_labels = {
    0: 'Hello', 1: 'Iloveyou', 2: 'Yes', 3: 'No', 4: 'Please',
    5: 'Thanks', 6: 'A', 7: 'B', 8: 'C', 9: 'D',
    10: 'E', 11: 'F', 12: 'G', 13: 'H', 14: 'I',
    15: 'K', 16: 'L', 17: 'M', 18: 'N', 19: 'O',
    20: 'P', 21: 'Q', 22: 'R', 23: 'S', 24: 'T',
    25: 'U', 26: 'V', 27: 'W', 28: 'X', 29: 'Y',
    30: 'how are you', 31: 'im fine', 32: 'im good',
    33: 'sorry', 34: 'see you later', 35: 'wait',
    36: 'go', 37: 'come', 38: 'help', 39: 'call me',
    40: 'hungry', 41: 'whats your name', 42:'good job', 43:'good morning', 44:'good night', 45:'good afternoon', 46:'you', 47:'nice to meet'
}

dataset_size = 300

# ======= CONFIGURE THIS TO CONTINUE FROM YOUR NEW SIGN =======
# If you already collected till class 29 ('Y'), start from 30
start_from = 30

# Automatically calculate
number_of_classes = len(class_labels)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("="*70)
print(f"DATA COLLECTION: {number_of_classes - start_from} NEW Classes (continuing from {start_from})")
print("="*70)

# Just print the new ones
print("New signs to collect:")
for i, label in list(class_labels.items())[start_from:]:
    print(f"  Class {i}: {label}")
print(f"\nImages per class: {dataset_size}")
print("="*70)

# ====== MAIN COLLECTION LOOP ======
for j in range(start_from, number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    label = class_labels.get(j, f"Class_{j}")
    
    print(f'\n{"="*70}')
    print(f'Collecting data for Class {j}: "{label}"')
    print(f'{"="*70}')

    # Ready phase
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        
        cv2.putText(frame, f'Class {j}: {label}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Ready? Press "Q"!', (50, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cv2.putText(frame, f'Starting in {countdown}...', (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)
        cv2.waitKey(1000)

    counter = 0
    print(f"Capturing {dataset_size} images for '{label}'... Keep hand steady!")
    
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        progress = int((counter / dataset_size) * 100)
        bar_width = 500
        cv2.rectangle(frame, (50, h - 60), (50 + int(progress * bar_width / 100), h - 40), 
                     (0, 255, 0), -1)
        cv2.rectangle(frame, (50, h - 60), (50 + bar_width, h - 40), (255, 255, 255), 2)
        cv2.putText(frame, label, (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f'{counter}/{dataset_size} ({progress}%)', (50, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)
        cv2.waitKey(25)
        
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1
    
    print(f'✓ Completed Class {j} "{label}": {dataset_size} images')

cap.release()
cv2.destroyAllWindows()

print('\n' + '='*70)
print('✓ DATA COLLECTION COMPLETE (continued)!')
print('='*70)
