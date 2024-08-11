from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

# Paths to your models
MODEL_PATHS = {
    'tomato': 'models/tomato_model.h5',
    'potato': 'models/potato_model.h5',
    'pepper_bell': 'models/pepper_bell_model.h5'
}

# Class labels for each model
CLASS_LABELS = {
    'tomato': [
        'bacterial spot',
        'early blight',
        'late blight',
        'leaf mold',
        'Septoria leaf spot',
        'spider mites two spotted spider mites',
        'target spot',
        'yellow leaf virus',
        'Mosaic virus',
        'Healthy'
    ],
    'potato': [
        'Early Blight',
        'Late Blight',
        'Healthy'
    ],
    'pepper_bell': [
        'bacterial spot',
        'healthy'
    ]
}

# Function to load and preprocess image for potato


def load_and_preprocess_image_potato(img_url, target_size=(150, 150)):
    response = requests.get(img_url)
    img = image.load_img(BytesIO(response.content), target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of size 1
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to load and preprocess image for tomato and pepper bell


def load_and_preprocess_image_tomato_pepper(img_url, target_size=(150, 150)):
    response = requests.get(img_url)
    img = image.load_img(BytesIO(response.content), target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of size 1
    img_array = tf.keras.applications.resnet50.preprocess_input(
        img_array)  # Preprocess for ResNet50
    return img_array

# Function to load the appropriate model based on vegetable type


def load_model_for_vegetable(vegetable):
    if vegetable not in MODEL_PATHS:
        raise ValueError(f"Model for '{vegetable}' not found.")
    return tf.keras.models.load_model(MODEL_PATHS[vegetable])

# Prediction route


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Check for required inputs
    if 'vegetable' not in data or 'image_url' not in data:
        return jsonify({'error': 'Please provide both "vegetable" and "image_url".'}), 400

    vegetable = data['vegetable'].lower()

    # Validate vegetable type
    if vegetable not in MODEL_PATHS:
        return jsonify({'error': f"Vegetable '{vegetable}' not supported."}), 400

    # Load the corresponding model
    model = load_model_for_vegetable(vegetable)

    # Preprocess the image based on the vegetable type
    if vegetable == 'potato':
        img_array = load_and_preprocess_image_potato(data['image_url'])
    else:  # For tomato and pepper bell
        img_array = load_and_preprocess_image_tomato_pepper(data['image_url'])

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASS_LABELS[vegetable][predicted_class_index]
    confidence = np.max(predictions)

    return jsonify({
        'vegetable': vegetable,
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    })


if __name__ == '__main__':
    app.run(debug=True)
