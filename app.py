import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model
model = load_model("mobilenet_model.keras")

# Title
st.title("🐶🐱 Cats vs Dogs Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    # Show result
    if prediction[0][0] > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")