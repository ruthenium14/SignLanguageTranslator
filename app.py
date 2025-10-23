from flask import Flask, render_template, Response, jsonify, request
import cv2
import pickle
import numpy as np
import base64
import warnings
import gc

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Lazy loading - don't load at startup
_model = None
_hands = None
_mp_hands = None
_mp_drawing = None
_mp_drawing_styles = None

# Labels
labels_dict = {
    0: 'Hello', 1: 'Iloveyou', 2: 'Yes', 3: 'No', 4: 'Please',
    5: 'Thanks', 6: 'A', 7: 'B', 8: 'C', 9: 'D',
    10: 'E', 11: 'F', 12: 'G', 13: 'H', 14: 'I',
    15: 'K', 16: 'L', 17: 'M', 18: 'N', 19: 'O',
    20: 'P', 21: 'Q', 22: 'R', 23: 'S', 24: 'T',
    25: 'U', 26: 'V', 27: 'W', 28: 'X', 29: 'Y'
}

def get_model():
    """Lazy load model only when needed"""
    global _model
    if _model is None:
        model_dict = pickle.load(open('./model.p', 'rb'))
        _model = model_dict['model']
    return _model

def get_mediapipe():
    """Lazy load mediapipe only when needed"""
    global _hands, _mp_hands, _mp_drawing, _mp_drawing_styles
    
    if _hands is None:
        import mediapipe as mp
        _mp_hands = mp.solutions.hands
        _mp_drawing = mp.solutions.drawing_utils
        _mp_drawing_styles = mp.solutions.drawing_styles
        
        # Use minimal configuration to save memory
        _hands = _mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,  # Only 1 hand
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # Lightest model (0 instead of 1)
        )
    
    return _hands, _mp_hands, _mp_drawing, _mp_drawing_styles

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lazy load resources
        model = get_model()
        hands, mp_hands, mp_drawing, mp_drawing_styles = get_mediapipe()
        
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]
        
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
            
            # Clean up to free memory
            del frame, frame_rgb, annotated_frame, buffer
            gc.collect()
            
            return jsonify({
                'success': True,
                'prediction': predicted_label,
                'confidence': round(confidence, 2),
                'class_id': predicted_class,
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'annotated_image': f'data:image/jpeg;base64,{annotated_image}'
            })
        else:
            # Clean up
            del frame, frame_rgb
            gc.collect()
            
            return jsonify({
                'success': False,
                'message': 'No hand detected'
            })
            
    except Exception as e:
        # Clean up on error
        gc.collect()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("="*70)
    print("Sign Language Recognition Web App")
    print("="*70)
  
