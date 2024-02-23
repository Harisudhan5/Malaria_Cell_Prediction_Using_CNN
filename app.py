import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
import cv2


st.title("Malarial Cell Prediction Through Microscopic Image")
uploaded_file = st.file_uploader("Upload an Image of RBC Cell...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    if st.button("Predict"):
        img = cv2.imread(uploaded_file)
        resized_img = cv2.resize(img,(256,256,3))
        loaded_model = load_model('Model//best_model.h5')
        result = loaded_model.predict(np.expand_dims(resized_img/255, 0))
        if result > 0.5:
            st.text("The cell is Non Parasitized")
        else:
            st.text("The cell is Parasitized")
        


