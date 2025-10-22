import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 30
dataset_size = 300

# ============= DEFINE YOUR 30 LABELS HERE =============
# You can use A-Z + 3 extras, or completely custom labels
class_labels = {
    0: 'Hello', 1: 'Iloveyou', 2: 'Yes', 3: 'No', 4: 'Please',
    5: 'Thanks', 6: 'A', 7: 'B', 8: 'C', 9: 'D',
    10: 'E', 11: 'F', 12: 'G', 13: 'H', 14: 'I',
    15: 'K', 16: 'L', 17: 'M', 18: 'N', 19: 'O',
    20: 'P', 21: 'Q', 22: 'R', 23: 'S', 24: 'T',
    25: 'U', 26: 'V', 27: 'W', 28: 'X', 29:'Y'
}


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("="*70)
print(f"DATA COLLECTION: {number_of_classes} Classes")
print("="*70)
print("Your signs:")
for i, label in class_labels.items():
    print(f"  Class {i}: {label}")
print(f"\nImages per class: {dataset_size}")
print("="*70)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    label = class_labels.get(j, f"Class_{j}")
    
    print(f'\n{"="*70}')
    print(f'Collecting data for Class {j}: "{label}"')
    print(f'{"="*70}')

    # Ready prompt with better visuals
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Mirror flip for intuitive use
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Draw guideline box
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        
        # Display info
        cv2.putText(frame, f'Class {j}: {label}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Ready? Press "Q"!', (50, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f'Progress: {j}/{number_of_classes}', (w - 350, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Countdown before starting
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

    # Capture images
    counter = 0
    print(f"Capturing {dataset_size} images for '{label}'... Keep hand steady!")
    
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Mirror flip
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Progress bar
        progress = int((counter / dataset_size) * 100)
        bar_width = 500
        cv2.rectangle(frame, (50, h - 60), (50 + int(progress * bar_width / 100), h - 40), 
                     (0, 255, 0), -1)
        cv2.rectangle(frame, (50, h - 60), (50 + bar_width, h - 40), (255, 255, 255), 2)
        
        # Display info
        cv2.putText(frame, label, (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f'{counter}/{dataset_size} ({progress}%)', (50, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Class {j+1}/{number_of_classes}', (w - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Data Collection', frame)
        cv2.waitKey(25)
        
        # Save image
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1
    
    print(f'✓ Completed Class {j} "{label}": {dataset_size} images')

cap.release()
cv2.destroyAllWindows()

print('\n' + '='*70)
print('✓ DATA COLLECTION COMPLETE!')
print('='*70)
print(f'Total classes: {number_of_classes}')
print(f'Total images: {number_of_classes * dataset_size}')
print('\nNext steps:')
print('  1. python create_dataset.py')
print('  2. python train_classifier.py')
print('  3. python inference_classifier.py (update labels_dict there too!)')