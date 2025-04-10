# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import cv2

# # Load the model
# model = tf.keras.models.load_model("model_n.keras")

# # Define class names
# class_names = [
#     'Bush Clock Vine', 'Common Lanthana', 'Datura', 'Hibiscus', 'Jatropha', 'Marigold',
#     'Nityakalyani', 'Rose', 'Yellow_Daisy', 'adathoda', 'banana', 'champaka', 'chitrak',
#     'crown flower', "four o'clock flower", 'honeysuckle', 'indian mallow', 'malabar melastome',
#     'nagapoovu', 'pinwheel flower', 'shankupushpam', 'spider lily', 'sunflower', 'thechi',
#     'thumba', 'touch me not', 'tridax procumbens', 'wild_potato_vine'
# ]

# # Title
# st.title("Flower Identifier")

# # Choose mode
# mode = st.radio("Choose input method:", ["Upload Image", "Real-Time Camera"])

# if mode == "Upload Image":
#     st.markdown("### Upload an image of a flower")
#     img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
#     if img is not None:
#         st.image(img, caption="Uploaded Image", use_column_width=True)

#         image = Image.open(img).convert("RGB")
#         image = tf.keras.preprocessing.image.img_to_array(image)
#         image = tf.cast(image, tf.float32)
#         image = tf.expand_dims(image, 0)

#         if st.button("Identify Flower"):
#             prediction = model.predict(image)
#             predicted_class = np.argmax(prediction[0])
#             confidence = round(100 * np.max(prediction[0]), 2)
#             flower_name = class_names[predicted_class]

#             st.success(f"Predicted Flower: **{flower_name}**")
#             st.info(f"Confidence: **{confidence}%**")

# elif mode == "Real-Time Camera":
#     st.markdown("### Real-Time Flower Recognition")
#     run = st.checkbox('Start Camera')
#     FRAME_WINDOW = st.image([])

#     cap = None
#     if run:
#         cap = cv2.VideoCapture(0)
#         while run:
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning("Failed to access camera.")
#                 break

#             img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img_array = tf.keras.preprocessing.image.img_to_array(img_rgb)
#             img_array = tf.expand_dims(tf.cast(img_array, tf.float32), 0)

#             predictions = model.predict(img_array)
#             predicted_class = np.argmax(predictions[0])
#             confidence = round(100 * np.max(predictions[0]), 2)
#             flower_name = class_names[predicted_class]

#             # Annotate frame
#             cv2.putText(frame, f"{flower_name} ({confidence}%)", (10, 30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#             FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         if cap:
#             cap.release()



import gradio as gr
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

# Prediction function
def predict_flower(img):
    image = img.convert("RGB")
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction[0])
    confidence = round(100 * np.max(prediction[0]), 2)
    flower_name = class_names[predicted_class]

    return f"🌼 Predicted Flower: {flower_name} ({confidence}%)"

# Gradio interface
title = "🌸 Flower Identifier using Deep Learning"
description = "Upload an image or use your camera to identify a flower from 28 known classes."

iface = gr.Interface(
    fn=predict_flower,
    inputs=[
        gr.Image(type="pil", label="Upload or Capture Flower Image", source="upload", tool="editor")
    ],
    outputs="text",
    title=title,
    description=description,
    live=False,
    examples=None,
)

if __name__ == "__main__":
    iface.launch()
