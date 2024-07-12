import streamlit as st
import mediapipe as mp
import numpy as np
import cv2

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
ImageClassifier = mp.tasks.vision.ImageClassifier
ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Load the model
@st.cache_resource
def load_model():
    model_path = 'efficientnet_lite0.tflite'
    options = ImageClassifierOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=5,
        running_mode=VisionRunningMode.IMAGE
    )
    return ImageClassifier.create_from_options(options)

classifier = load_model()

def classify_image(image):
    # Convert the image to RGB (MediaPipe requires RGB input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Perform classification
    classification_result = classifier.classify(mp_image)
    
    # Process and return results
    results = []
    for category in classification_result.classifications[0].categories:
        results.append((category.category_name, category.score))
    
    return sorted(results, key=lambda x: x[1], reverse=True)

# Streamlit app
st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Display the image in the left column
    with col1:
        st.image(image, channels="BGR", use_column_width=True)
    
    # Perform classification and display results in the right column
    with col2:
        results = classify_image(image)
        st.subheader("Classification Results:")
        for label, score in results:
            st.write(f"{label}: {score*100:.2f}%")