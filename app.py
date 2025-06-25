import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model("cat_dog_model.h5")

# App title
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    if prediction >= 0.7:result = "Dog 🐶"
    elif prediction <= 0.3:result = "Cat 🐱"
    else: result = "Undefined ❓ (not confident)"

    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {prediction:.2f}")
