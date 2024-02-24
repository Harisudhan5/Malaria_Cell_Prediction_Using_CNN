import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

def extract_text(image):
    loaded_model = load_model('Model//best_model.h5')
    resized_image = cv2.resize(loaded_model,(256,256,3))
    result = loaded_model.predict(np.expand_dims(resized_image/255, 0))
    if result > 0.5:
        return "Non Parasitized"
    else:
        return "Parasitized"

def main():
    
    st.title("Malarial Cell Prediction Using Microscopic Image")

    uploaded_file = st.file_uploader("Upload an image - File format supported - [jpeg, png, jpg]", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Mine the Image"):
            # Convert PIL Image to OpenCV format
            image_cv = np.array(image)

            # Output 1: Extracted Text
            text_result = extract_text(image_cv)
            st.subheader("Result")
            st.text(text_result)

if __name__ == "__main__":
    main()
