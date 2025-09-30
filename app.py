import flask
import pickle
import numpy as np
import pandas as pd
import json
import os
from flask import Flask, request, jsonify, render_template

# --- Configuration ---
MODEL_FILENAME = 'rf_model.pkl'
FEATURE_FILENAME = 'model_features.pkl'
app = Flask(__name__)

# Global variables for model and features
model = None
feature_names = None

def load_assets():
    """Loads the trained model and feature list into memory."""
    global model
    global feature_names
    
    print(f"Attempting to load model from {MODEL_FILENAME}...")
    try:
        # Load the trained model
        with open(MODEL_FILENAME, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {MODEL_FILENAME} not found. Please run malware_trainer.py first to train the model.")
        # Exit or raise error, cannot continue without model
        exit(1)
    
    print(f"Attempting to load features from {FEATURE_FILENAME}...")
    try:
        # Load the feature names (PE Imports)
        with open(FEATURE_FILENAME, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"Loaded {len(feature_names)} features.")
    except FileNotFoundError:
        print(f"Error: {FEATURE_FILENAME} not found. Please run malware_trainer.py first.")
        exit(1)

# Load the model upon application start
load_assets()


@app.route('/')
def home():
    """
    Renders the single-page HTML interface (templates/index.html).
    The 'feature_names' list is passed to the Jinja template so JavaScript
    can dynamically generate the required input toggles.
    """
    return render_template('index.html', feature_names=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles POST requests from the frontend to make predictions."""
    if model is None or feature_names is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        
        # The frontend sends a dictionary where keys are feature names and values are 0 or 1
        input_data = data.get('features', {})
        
        # 1. Input Validation: Ensure all expected features are present
        if len(input_data) != len(feature_names):
             return jsonify({'error': f"Expected {len(feature_names)} features, received {len(input_data)}. Check input structure."}), 400

        # 2. Create a standardized feature vector (NumPy array)
        # Features MUST be in the exact order the model was trained on
        feature_vector = [input_data.get(feature, 0) for feature in feature_names]
        
        # Convert to NumPy array and reshape for prediction (1 sample, N features)
        final_features = np.array(feature_vector).reshape(1, -1)
        
        # 3. Make the prediction
        prediction = model.predict(final_features)[0]
        
        # Get prediction probabilities for confidence score
        proba = model.predict_proba(final_features)[0]
        confidence = proba[prediction] * 100
        
        # 4. Format the result
        result = {
            'prediction': 'Malware Detected' if prediction == 1 else 'Benign File',
            'label': int(prediction),
            'confidence': f"{confidence:.2f}%"
        }
        
        print(f"Prediction made: {result['prediction']} with confidence {result['confidence']}")
        return jsonify(result)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# This runs the application on the local PC (accessible via http://127.0.0.1:5000)
if __name__ == '__main__':
    print("Starting Flask application...")
    # Flask automatically looks for HTML files in the 'templates' folder
    app.run(debug=True)