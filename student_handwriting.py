from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array  # Fixed import
from werkzeug.utils import secure_filename
import ollama

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "dyslexia_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

image_size = (64, 64)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_user_image(image_path):
    try:
        img = load_img(image_path, target_size=image_size, color_mode="grayscale")
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = img_array.reshape(1, 64, 64, 1)  # Reshape for model input
        return img_array
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def generate_dyslexia_report(confidence, age, learning_style, challenges, goals):
    try:
        response = ollama.chat(model='gemma:2b', messages=[
            {"role": "system", "content": "Generate a dyslexia improvement report."},
            {"role": "user", "content": f"Dyslexia Confidence: {confidence}%, Age: {age}, Learning Style: {learning_style}, Challenges: {challenges}, Goals: {goals}"}
        ])
        return response['message']['content']
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return "Error generating report."

@app.route('/')
def index():
    return render_template("student_dyslexia.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    img_array = load_user_image(file_path)
    if img_array is None:
        return jsonify({"error": "Invalid image"}), 400
    
    prediction = model.predict(img_array)
    dyslexia_confidence = int(np.max(prediction) * 100)  # Get confidence percentage
    
    # Get additional parameters from the form
    age = request.form.get('age', 'Unknown')
    learning_style = request.form.get('learning_style', 'Unknown')
    challenges = request.form.get('challenges', 'Unknown')
    goals = request.form.get('goals', 'Unknown')
    
    # Generate a report using gemma:2b
    report = generate_dyslexia_report(dyslexia_confidence, age, learning_style, challenges, goals)
    
    return render_template("student_dyslexia.html", 
                           confidence=dyslexia_confidence, 
                           report=report)

if __name__ == '__main__':
    app.run(debug=True)
