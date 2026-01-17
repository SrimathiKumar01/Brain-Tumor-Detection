import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import imutils
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .st-emotion-cache-16idsys p {
            font-size: 20px;
        }
        .custom-metric {
            border: 1px solid #e6e6e6;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load the model
@st.cache_resource
def load_tumor_model():
    try:
        return load_model('brain_tumor_detector.h5')
    except:
        st.error("Error: Model file not found. Please ensure 'brain_tumor_detector.h5' is in the same directory.")
        return None

model = load_tumor_model()

def process_tumor_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    # Find contours
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return None
        
    c = max(cnts, key=cv.contourArea)

    # Find extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop image
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    
    # Resize for model
    processed_image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
    processed_image = processed_image / 255.
    return processed_image

def detect_tumor(image):
    processed_image = process_tumor_image(image)
    if processed_image is None:
        return None
    
    # Reshape for model
    model_input = processed_image.reshape((1, 240, 240, 3))
    prediction = model.predict(model_input)
    return prediction[0][0]

def visualize_tumor_region(image):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Threshold
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Noise removal
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    
    return image

def save_analysis_history(filename, prediction, has_tumor):
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': filename,
        'confidence': float(prediction) if prediction is not None else None,
        'has_tumor': has_tumor
    }
    st.session_state.history.append(history_entry)

def plot_confidence_history():
    if not st.session_state.history:
        return None
    
    df = pd.DataFrame(st.session_state.history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['confidence'],
        mode='lines+markers',
        name='Confidence Score'
    ))
    fig.update_layout(
        title='Confidence Score History',
        xaxis_title='Analysis Number',
        yaxis_title='Confidence Score',
        yaxis_range=[0, 1]
    )
    return fig

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v2.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Analysis History", "About"])

if page == "Home":
    # Main UI
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("### Upload your MRI scan for analysis")

    # File uploader with additional information
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image...", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Original Image", "Analysis", "Detailed View"])
        
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, 1)
        
        with tab1:
            st.image(uploaded_file, width=400, caption="Original MRI Scan")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Detect Tumor", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        prediction = detect_tumor(image)
                        if prediction is not None:
                            has_tumor = prediction > 0.5
                            save_analysis_history(uploaded_file.name, prediction, has_tumor)
                            
                            st.subheader("Detection Result")
                            if has_tumor:
                                st.error("⚠️ Tumor Detected")
                                st.metric("Confidence", f"{prediction:.2%}")
                            else:
                                st.success("✅ No Tumor Detected")
                                st.metric("Confidence", f"{(1-prediction):.2%}")
                        else:
                            st.warning("Could not process the image. Please try another image.")
            
            with col2:
                if st.button("🎯 Visualize Region", use_container_width=True):
                    with st.spinner("Generating visualization..."):
                        tumor_region = visualize_tumor_region(image.copy())
                        st.image(tumor_region, channels="BGR", width=400, caption="Tumor Region Visualization")
        
        with tab3:
            st.subheader("Image Properties")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Image Size", f"{image.shape[1]}x{image.shape[0]}px")
            with col2:
                st.metric("File Size", f"{uploaded_file.size/1024:.1f}KB")
            with col3:
                st.metric("Channels", image.shape[2])

elif page == "Analysis History":
    st.title("📊 Analysis History")
    
    if st.session_state.history:
        # Display confidence trend
        st.plotly_chart(plot_confidence_history(), use_container_width=True)
        
        # Display history table
        st.subheader("Previous Analyses")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        # Download history
        if st.button("Download History CSV"):
            df.to_csv("analysis_history.csv", index=False)
            st.success("History saved to analysis_history.csv")
    else:
        st.info("No analysis history available yet. Start by analyzing some images!")

else:  # About page
    st.title("ℹ️ About Brain Tumor Detection System")
    
    st.markdown("""
    ### Overview
    This advanced brain tumor detection system utilizes deep learning technology to analyze MRI scans 
    and identify potential tumors. The system provides both detection and visualization capabilities 
    to assist in the analysis of brain MRI scans.
    
    ### Features
    - 🔍 **Tumor Detection**: Analyzes MRI scans using a deep learning model
    - 🎯 **Region Visualization**: Highlights potential tumor regions
    - 📊 **Analysis History**: Tracks and visualizes detection history
    - 📈 **Detailed Metrics**: Provides confidence scores and image properties
    
    ### How It Works
    1. Upload an MRI scan image
    2. The system processes the image using advanced computer vision techniques
    3. A deep learning model analyzes the processed image
    4. Results are displayed with confidence scores and visualizations
    
    ### Best Practices
    - Use clear, high-quality MRI scans
    - Ensure images are properly oriented
    - Use standard medical imaging formats
    - Regular system calibration and validation
    
    ### Technical Details
    - Model: Custom CNN architecture
    - Image Processing: OpenCV
    - Visualization: Plotly & Streamlit
    - Data Analysis: Pandas & NumPy
    """)
    
    # Display system metrics
    st.subheader("System Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Python Version", "3.8+")
    with col2:
        st.metric("TensorFlow Version", "2.15.0")
    with col3:
        st.metric("OpenCV Version", "4.9.0")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Developed by The Care Crew for medical imaging analysis </p>
    </div>
    """,
    unsafe_allow_html=True
) 