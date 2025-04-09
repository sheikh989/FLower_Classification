import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import keras


# Load trained model
my_model  = tf.keras.models.load_model('model_n.keras')  

# Class names (must match training order)
class_names = [
    'Bush Clock Vine', 'Common Lanthana', 'Datura', 'Hibiscus', 'Jatropha', 'Marigold',
    'Nityakalyani', 'Rose', 'Yellow_Daisy', 'adathoda', 'banana', 'champaka', 'chitrak',
    'crown flower', "four o'clock flower", 'honeysuckle', 'indian mallow', 'malabar melastome',
    'nagapoovu', 'pinwheel flower', 'shankupushpam', 'spider lily', 'sunflower', 'thechi',
    'thumba', 'touch me not', 'tridax procumbens', 'wild_potato_vine'
]

# Streamlit UI
st.markdown("## Identify The Flower in the Image")
img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img is not None:
    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load and preprocess the image (do NOT resize or normalize manually if your model handles it)
    image = Image.open(img).convert("RGB")
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)  # Add batch dimension

    if st.button("Identify Flower"):
        prediction = my_model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = round(100 * np.max(prediction[0]), 2)

        flower_name = class_names[predicted_class]
        st.success(f"Predicted Flower: **{flower_name}**")
        st.info(f"Confidence: **{confidence}%**")
