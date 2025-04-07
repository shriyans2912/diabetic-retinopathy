import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Loading the trained model
model = load_model('diabetic_retinopathy_model.h5')

# Define a function for prediction
def predict_image(image):
    # Preprocess the image for prediction
    resized_image = image.resize((150, 150))  
    image_array = img_to_array(resized_image) / 255.0  
    image_array = np.expand_dims(image_array, axis=0)  

    # Predict
    prediction = np.round(model.predict(image_array))
    return prediction

# Streamlit app
st.title("Optify")
st.subheader("Diabetic Retinopathy Predictor")
st.write("Upload a retinal image to predict diabetic retinopathy.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and resize the uploaded image for display
    image = Image.open(uploaded_file)
    display_image = image.resize((300, 300)) 

    # Show the resized image
    st.image(display_image, caption="Uploaded Retinal Image")

    # Predict button
    if st.button("Predict"):
        prediction = predict_image(image)  
        if prediction[0][0] == 0:
            st.info("**Result:** Positive for Diabetic Retinopathy")
            st.caption("**Disclaimer:** Optify - The Diabetic Retinopathy Predictor App doesn't intend to replace any professional medical advice, diagnosis, or treatment.")
        else:
            st.info("**Result:** Negative for Diabetic Retinopathy")
            st.caption("**Disclaimer:** Optify - The Diabetic Retinopathy Predictor App doesn't intend to replace any professional medical advice, diagnosis, or treatment.")


import streamlit as st

st.markdown(
    """
    <div style="text-align: center; font-size: 16px;">
        &copy; Optify-ShriyansRout-2025
    </div>
    """,
    unsafe_allow_html=True
)
