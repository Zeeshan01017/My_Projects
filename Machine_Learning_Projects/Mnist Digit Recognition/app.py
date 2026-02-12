import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
from PIL import Image, ImageOps
model = keras.models.load_model("Machine_Learning_Projects/Mnist Digit Recognition/mnist_model.keras")

st.title("MNIST Digit Classifier")

uploaded = st.file_uploader("Upload a digit image (28x28)", type=["png","jpg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    img = ImageOps.invert(img)          
    img = img.resize((28, 28))

    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    st.write("Predicted Digit:", np.argmax(prediction))
