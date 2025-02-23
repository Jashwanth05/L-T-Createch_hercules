import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations for compatibility

import joblib
import numpy as np
import cv2
import tensorflow as tf
import pickle
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ================= Machine Failure Model ===================

# Load the machine failure model
try:
    machine_model = joblib.load("machine_failure_models.pkl")

    if isinstance(machine_model, dict):
        model_key = list(machine_model.keys())[0]
        machine_model = machine_model.get(model_key)

        if machine_model is None:
            raise ValueError("No valid machine failure model found in the dictionary.")

    print("✅ Machine failure model loaded successfully.")

except Exception as e:
    print(f"❌ Error loading machine failure model: {e}")
    machine_model = None

# Define feature names
features = [
    'Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_rpm',
    'Torque_Nm', 'Tool_Wear_min', 'Delta_T', 'Torque_per_rpm',
    'Power_W', 'Normalized_tool_wear', 'Type_Encoded'
]

# Define failure types
failure_types = [
    'No Failure', 'Tool Wear Failure', 'Heat Dissipation Failure', 
    'Power Failure', 'Overstrain Failure', 'Random Failure'
]

# Generate random IoT data
def generate_random_input():
    values = np.random.uniform(
        low=[290, 295, 1000, 10, 0, 0, 0, 0, 0, 0]
        , 
        high=[320, 350, 3000, 80, 300, 50, 0.1, 1e5, 1, 3]
    ).tolist()
    return values

@app.route("/predict/machine", methods=["GET"])
def predict_machine():
    if machine_model is None:
        return jsonify({"error": "Machine failure model not loaded"}), 500

    input_data = np.array(generate_random_input()).reshape(1, -1)

    try:
        prediction = machine_model.predict(input_data)[0]
        result = failure_types[prediction] if 0 <= prediction < len(failure_types) else "Unknown Failure Type"
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

    return jsonify({
        "prediction": result, 
        "input_data": dict(zip(features, input_data.flatten().tolist()))
    })

# ================= Crack Detection Model ===================

# Load the crack detection model
try:
    crack_model = tf.keras.models.load_model("crack_detection_model (1).h5")
    crack_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("✅ Crack detection model loaded and compiled successfully.")

except Exception as e:
    print(f"❌ Error loading crack detection model: {e}")
    crack_model = None

# Path to test images
TEST_IMAGE_FOLDER = 'C:\\Users\\DELL\\OneDrive\\Desktop\\Createch\\my-react-app\\public\\test'

# Image Preprocessing Function
def preprocess_image(filepath, img_size=120):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale
    if img is None:
        return None  # Skip invalid images
    
    img = cv2.resize(img, (img_size, img_size))  # Resize to model input shape
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (H, W, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension (1, H, W, 1)
    return img

@app.route('/predict/crack', methods=['GET'])
def predict_crack():
    if crack_model is None:
        return jsonify({"error": "Crack detection model not loaded"}), 500

    results = []
    for filename in os.listdir(TEST_IMAGE_FOLDER):
        img_path = os.path.join(TEST_IMAGE_FOLDER, filename)

        # Check if file exists and is a valid image
        if not os.path.isfile(img_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip if not a valid image file

        try:
            # Preprocess image
            img = preprocess_image(img_path)
            if img is None:
                continue  # Skip if image is invalid

            # Get prediction
            prediction = crack_model.predict(img)
            label = "No Crack" if prediction[0][0] > 0.4 else "Crack"

            if label == "Crack":  # Only include cracked images
                results.append({
                    "filename": filename,
                    "prediction": label,
                    "image_url": f"http://127.0.0.1:5000/image/{filename}"
                })

        except Exception as e:
            return jsonify({"error": f"Crack prediction error: {e}"}), 500

    return jsonify(results)

@app.route('/image/<filename>')
def get_image(filename):
    img_path = os.path.join(TEST_IMAGE_FOLDER, filename)
    
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return jsonify({"error": "Image not found"}), 404
    
    
# ================= Fire Detection Model using YOLO ===================
try:
    fire_model = YOLO("best.pt")
    print("✅ Fire detection model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading fire detection model: {e}")
    fire_model = None

TEST_FIRE_IMAGE_FOLDER = os.path.join(os.getcwd(), 'C:\\Users\\DELL\\OneDrive\\Desktop\\Createch\\my-react-app\\public\\test2')

@app.route('/predict/fire', methods=['GET'])
def predict_fire():
    if fire_model is None:
        return jsonify({"error": "Fire detection model not loaded"}), 500
    results = []
    for filename in os.listdir(TEST_FIRE_IMAGE_FOLDER):
        img_path = os.path.join(TEST_FIRE_IMAGE_FOLDER, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        prediction = fire_model(img_path)  # Run YOLO model on the image
        if any(pred.boxes for pred in prediction):  # If boxes detected
            results.append({"filename": filename, "prediction": "🔥 Fire Detected", "image_url": f"/fireimage/{filename}"})
    return jsonify(results)

@app.route('/fireimage/<filename>')
def get_fire_image(filename):
    img_path = os.path.join(TEST_FIRE_IMAGE_FOLDER, filename)
    return send_file(img_path) if os.path.exists(img_path) else jsonify({"error": "Image not found"}), 404

if __name__ == "__main__":
    app.run()