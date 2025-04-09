import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = tf.keras.models.load_model("model_n.keras")

# Define class names
class_names = [
    'Bush Clock Vine', 'Common Lanthana', 'Datura', 'Hibiscus', 'Jatropha', 'Marigold',
    'Nityakalyani', 'Rose', 'Yellow_Daisy', 'adathoda', 'banana', 'champaka', 'chitrak',
    'crown flower', "four o'clock flower", 'honeysuckle', 'indian mallow', 'malabar melastome',
    'nagapoovu', 'pinwheel flower', 'shankupushpam', 'spider lily', 'sunflower', 'thechi',
    'thumba', 'touch me not', 'tridax procumbens', 'wild_potato_vine'
]

# Streamlit UI
st.title("Real-Time Flower Recognition")
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

# OpenCV webcam logic
cap = None
if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to access camera.")
        break

    # Convert BGR to RGB (OpenCV uses BGR)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize and convert to tensor (model already handles resize & normalization)
    img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
    img_array = tf.expand_dims(tf.cast(img_array, tf.float32), 0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    flower_name = class_names[predicted_class]

    # Display prediction on frame
    cv2.putText(frame, f"{flower_name} ({confidence}%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame in Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Release webcam after loop
if cap:
    cap.release()
