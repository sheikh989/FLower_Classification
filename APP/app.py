import os
import streamlit as st
import requests
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

my_model = load_model('flower_model_v.2.h5') 
class_names=['Bush Clock Vine',
             'Common Lanthana',
             'Datura',
             'Hibiscus',
             'Jatropha',
             'Marigold',
             'Nityakalyani',
             'Rose',
             'Yellow_Daisy',
             'adathoda',
             'banana',
             'champaka',
             'chitrak',
             'crown flower',
             "four o'clock flower",
             'honeysuckle',
             'indian mallow',
             'malabar melastome',
             'nagapoovu',
             'pinwheel flower',
             'shankupushpam',
             'spider lily',
             'sunflower',
             'thechi',
             'thumba',
             'touch me not',
             'tridax procumbens',
             'wild_potato_vine']


st.markdown("## Identify The Flower in the Image")
img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img is not None:
    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load and preprocess the image
    image = Image.open(img).convert("RGB")
    image = image.resize((224, 224))

    if st.button("Identify Flower"):

        image_array = np.array(image) / 255.0  # Normalize if your model was trained this way
        image_batch = np.expand_dims(image_array, axis=0)  # Add batch dimension

        prediction = my_model.predict(image_batch)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        flower_name = class_names[predicted_class]
        st.write(f"Predicted Flower: {flower_name}")
        st.write(f"Confidence: {confidence:.2f}")