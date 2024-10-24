from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load the pre-trained model using Keras
model = load_model(r'C:\Users\nilay\OneDrive - Cal State Fullerton (1)\Desktop\NILAY-TO-JOB-DATA\SPRING 2024\Job Preperation Learning\FOR LEARNING\TensorFlow and Keras\app\model.h5')

app = Flask(__name__, template_folder=r'C:\Users\nilay\OneDrive - Cal State Fullerton (1)\Desktop\NILAY-TO-JOB-DATA\SPRING 2024\Job Preperation Learning\FOR LEARNING\TensorFlow and Keras\app\template')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        img = Image.open(file).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))           # Resize to 28x28 pixels
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize and reshape
        
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction, axis=1)[0]  # Get the predicted class
        
        return render_template('index.html', prediction=predicted_digit)

if __name__ == '__main__':
    app.run(debug=True)
