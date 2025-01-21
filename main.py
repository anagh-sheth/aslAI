import cv2
import streamlit as st
import mediapipe as mp
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize Roboflow with your API key
rf = Roboflow(api_key="3uLxZ8MmVP8UrF7XDKXs")

# Load the project and version
project = rf.workspace("anagh").project("american-sign-language-letters-1hqg4")  # Roboflow 100 Sign Language Project
model = project.version(1).model


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="iFjPUVPeTAGflBUC7OvK"
)
 
st.set_page_config(page_title="LearnASL.ai", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    
    .main-title {
        background: linear-gradient(120deg, #1a237e, #0d47a1, #2196f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5rem !important;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
        animation: fadeInDown 1.2s ease-out;
        letter-spacing: -1px;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.8rem;
        color: #000000;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 600;
        animation: fadeIn 1.5s ease-out;
        letter-spacing: -0.5px;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.05);
    }
    
    .feature-card {
        background: white;
        padding: 2.5rem;
        border-radius: 24px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .feature-card h3 {
        color: #000000;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.05);
    }
    
    .feature-card p {
        color: #000000;
        font-size: 1.2rem;
        line-height: 1.6;
        font-weight: 500;
    }
    
    .start-button {
        background: linear-gradient(45deg, #1a237e, #0d47a1);
        color: white;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin: 2rem auto;
        display: block;
        width: 80%;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .start-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #0d47a1, #2196f3);
    }
    
    /* Add this to fade in the feature cards sequentially */
    .feature-card-1 { animation: slideUp 0.6s ease-out; }
    .feature-card-2 { animation: slideUp 0.8s ease-out; }
    .feature-card-3 { animation: slideUp 1s ease-out; }
    
    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        animation: bounce 2s infinite;
    }
    
    .camera-controls {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1rem 0;
    }
    
    .button-primary {
        background: linear-gradient(45deg, #1a237e, #0d47a1);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .button-secondary {
        background: linear-gradient(45deg, #d32f2f, #c62828);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .button-primary:hover, .button-secondary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .letter-card {
        background: white;
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 2rem 0;
        animation: slideUp 0.8s ease-out;
    }
    
    .letter-title {
        color: #1a237e;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        animation: pulse 2s infinite;
    }
    
    .camera-feed {
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        animation: fadeIn 1s ease-out;
        aspect-ratio: 1;
        width: 100%;
        max-width: 500px;
        margin: 0 auto;
    }
    
    .camera-feed img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .learn-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 30px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .letter-select-container {
        background: linear-gradient(145deg, #ffffff, #f5f7fa);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    .letter-display {
        color: #1a237e;
        font-size: 8rem;
        font-weight: 800;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    
    .instruction-text {
        color: #2c3e50;
        font-size: 1.2rem;
        line-height: 1.6;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .section-title {
        color: #1a237e;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        background: linear-gradient(120deg, #1a237e, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .camera-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .control-button {
        background: linear-gradient(45deg, #1a237e, #0d47a1);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stop-button {
        background: linear-gradient(45deg, #c62828, #d32f2f);
    }
    
    .letter-card {
        background: #89CFF0;
        padding: 2.5rem;
        border-radius: 24px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin: 2rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .letter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .sign-image {
        aspect-ratio: 1;
        width: 100%;
        max-width: 500px;
        margin: 0 auto;
        display: block;
    }
    
    .sign-image img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .content-wrapper {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .image-container {
        width: 100%;
        aspect-ratio: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1rem 0;
    }
    
    .image-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    
    .camera-section {
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .learn-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .sign-container {
        width: 200px;
        height: 200px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: center;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        padding: 0.5rem;
    }
    
    .camera-container {
        width: 300px;
        height: 300px;
        margin: 0 auto;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    
    .sign-image {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    
    .camera-feed {
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .camera-feed img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .learn-layout {
        max-width: 1000px;
        margin: 0 auto;
        padding: 0;
    }
    
    .letter-display {
        color: #1a237e;
        font-size: 6rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        padding: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    
    .practice-grid {
        display: grid;
        grid-template-columns: auto auto;
        gap: 1rem;
        justify-content: center;
        align-items: start;
        margin: 0 auto;
        padding: 0;
    }
    
    .sign-container, .camera-container {
        margin-top: 0.5rem;
    }
    
    .practice-grid {
        display: grid;
        grid-template-columns: auto auto;
        gap: 2rem;
        justify-content: center;
        align-items: center;
        margin: 1rem auto;
    }
    </style>
""", unsafe_allow_html=True)

# Set up page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Camera"])

# Page 1: Home
if page == "Home":
    st.markdown('<h1 class="main-title">Learn American Sign Language with AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description" style="color: #00008B;">Master ASL through interactive lessons and real time feedback</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Face Detection</h3>
                <p>Real-time face detection powered by OpenCV.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Hand Detection</h3>
                <p>Track hands with MediaPipe for seamless recognition.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)



# Page 2: Camera
elif page == "Camera":
    st.markdown('<h1 class="main-title">ASL Letter Detection</h1>', unsafe_allow_html=True)
    FRAME_WINDOW = st.image([])

    # Placeholder for the video frame
    frame_placeholder = st.empty()

    # Add a button to stop the video
    stop_button = st.button("Stop")

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        st.error("Could not open the camera.")
    else:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame. Exiting.")
                break

            cv2.imwrite("frame.jpg", frame)
    
            # Make prediction
            prediction = model.predict("frame.jpg", confidence=40).json()
            predictions = prediction.get("predictions", [])
            
            # Draw bounding boxes
            for pred in predictions:
                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                label = pred['class']
                confidence = pred['confidence']
                
                # Draw Rectangle and Label
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x - w//2, y - h//2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Convert BGR to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

    cap.release()
    cv2.destroyAllWindows()
    
