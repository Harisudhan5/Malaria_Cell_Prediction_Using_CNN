import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

def extract_text(image):
    loaded_model = load_model('Model//best_model.h5')
    resize = tf.image.resize(image, (256,256))
    result = loaded_model.predict(np.expand_dims(resize/255, 0))
    if result > 0.5:
        return "Non Parasitized"
    else:
        return "Parasitized"

def main():
    st.title("Malarial Cell Prediction Using Microscopic Image of RBC")
    uploaded_file = st.file_uploader("Upload an image - File format supported - [jpeg, png, jpg]", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            # Convert PIL Image to OpenCV format
            image_cv = np.array(image)
            image_cv = image_cv[:, :, :3]
            print(image_cv.shape)
            # Output 1: Extracted Text
            text_result = extract_text(image_cv)
            st.subheader("Result")
            st.text(text_result)

if __name__ == "__main__":
    main()
