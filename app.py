import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the model (you'll need to upload it to your GitHub repo)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('digit_recognition_model.h5')
    return model

def preprocess_image(image):
    # Convert to grayscale and resize
    image = image.convert('L').resize((28, 28))
    # Convert to numpy array and normalize
    image_array = np.array(image).astype('float32') / 255.0
    # Reshape for model input
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def main():
    st.title("🔢 Handwritten Digit Recognition")
    st.write("Upload an image of a handwritten digit (0-9) for prediction")
    
    # Load model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Make prediction
        if st.button("Predict Digit"):
            processed_image = preprocess_image(image)
            
            # Get prediction
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Display results
            st.success(f"**Predicted Digit: {predicted_digit}**")
            st.info(f"Confidence: {confidence:.2%}")
            
            # Show probability distribution
            st.subheader("Probability Distribution")
            prob_df = {
                'Digit': list(range(10)),
                'Probability': prediction[0]
            }
            st.bar_chart(prob_df)

if __name__ == "__main__":
    main()

