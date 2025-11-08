from flask import Flask, render_template, Response, jsonify, request
import cv2
import pickle
import numpy as np
import mediapipe as mp
import base64
import warnings

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

app = Flask(__name__)

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Labels
labels_dict = {
    0: 'Hello', 1: 'Iloveyou', 2: 'Yes', 3: 'No', 4: 'Please',
    5: 'Thanks', 6: 'A', 7: 'B', 8: 'C', 9: 'D',
    10: 'E', 11: 'F', 12: 'G', 13: 'H', 14: 'I',
    15: 'K', 16: 'L', 17: 'M', 18: 'N', 19: 'O',
    20: 'P', 21: 'Q', 22: 'R', 23: 'S', 24: 'T',
    25: 'U', 26: 'V', 27: 'W', 28: 'X', 29: 'Y',
    30: 'how are you', 31: 'im fine', 32: 'im good',
    33: 'sorry', 34: 'see you later', 35: 'wait',
    36: 'go', 37: 'come', 38: 'help', 39: 'call me',
    40: 'hungry', 41: 'whats your name', 42:'good job', 
    43:'good morning', 44:'good night', 45:'good afternoon', 
    46:'you', 47:'nice to meet'
}

# Routes for HTML pages
@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/translator')
def translator():
    return render_template('translator.html')

@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract features
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            data_aux = []
            min_x, min_y = min(x_coords), min(y_coords)
            for x, y in zip(x_coords, y_coords):
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)
            
            # Predict
            prediction = model.predict([np.asarray(data_aux)])
            predicted_class = int(prediction[0])
            predicted_label = labels_dict.get(predicted_class, '?')
            
            # Get confidence
            try:
                probabilities = model.predict_proba([np.asarray(data_aux)])[0]
                confidence = float(probabilities[predicted_class] * 100)
            except:
                confidence = 0.0
            
            # Get bounding box
            x1 = int(min(x_coords) * W)
            y1 = int(min(y_coords) * H)
            x2 = int(max(x_coords) * W)
            y2 = int(max(y_coords) * H)
            
            # Draw landmarks on frame
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw bounding box and label
            cv2.rectangle(annotated_frame, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), 2)
            cv2.putText(annotated_frame, predicted_label, (x1, y1 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Encode annotated frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_image = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'prediction': predicted_label,
                'confidence': round(confidence, 2),
                'class_id': predicted_class,
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'annotated_image': f'data:image/jpeg;base64,{annotated_image}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No hand detected'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

if __name__ == '__main__':
    print("="*70)
    print("Sign Language Recognition Web App")
    print("="*70)
    print("Server starting at http://localhost:10000")
    print("Press Ctrl+C to stop")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=10000)
