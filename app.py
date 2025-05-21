import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import time
from utils.model_handler import ASLModelHandler

# Set page config
st.set_page_config(
    page_title="ASL Recognition",
    page_icon="👋",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'model_handler' not in st.session_state:
    # Initialize model handler with API key from environment variable
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if api_key:
        st.session_state.model_handler = ASLModelHandler(api_key)
    else:
        st.warning("Please set the ROBOFLOW_API_KEY environment variable to enable ASL recognition.")
        st.session_state.model_handler = None
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

def home_page():
    st.title("Welcome to ASL Recognition! 👋")
    st.markdown("""
    ### Learn American Sign Language with AI!
    
    This application helps you learn and practice American Sign Language (ASL) using artificial intelligence. 
    Our system uses advanced computer vision to recognize hand signs in real-time.
    
    #### Features:
    - Real-time ASL letter detection
    - Interactive learning experience
    - High accuracy recognition (96.1% mAP)
    - Practice mode with instant feedback
    
    #### How to Use:
    1. Click on "Camera Practice" in the sidebar
    2. Select a letter you want to practice
    3. Show your ASL sign to the camera
    4. Get instant feedback on your sign!
    
    Get started by selecting "Camera Practice" from the sidebar menu!
    """)
    
    # Add some example images or GIFs here
    st.image("https://www.nidcd.nih.gov/sites/default/files/Content%20Images/NIDCD-ASL-hands-2.jpg", 
             caption="American Sign Language Example")

def toggle_camera():
    st.session_state.camera_running = not st.session_state.camera_running

def camera_page():
    st.title("ASL Camera Practice 📸")
    
    # Letter selection
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    selected_letter = st.selectbox("Select a letter to practice:", letters)
    
    st.write(f"Make the ASL sign for letter '{selected_letter}'")
    
    # Create placeholders
    camera_placeholder = st.empty()
    feedback_placeholder = st.empty()
    
    # Camera control button
    col1, col2 = st.columns([1, 3])
    with col1:
        if not st.session_state.camera_running:
            st.button("Start Camera", key="start_camera", on_click=toggle_camera)
        else:
            st.button("Stop Camera", key="stop_camera", on_click=toggle_camera)
    
    if st.session_state.camera_running:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # MediaPipe hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_draw = mp.solutions.drawing_utils
        
        try:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                    
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame and detect hands
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_draw.draw_landmarks(
                            rgb_frame,  # Draw on RGB frame instead of BGR
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                        
                        if st.session_state.model_handler:
                            # Get prediction
                            predicted_letter, confidence = st.session_state.model_handler.predict_letter(frame)
                            
                            if predicted_letter:
                                # Display feedback
                                if predicted_letter == selected_letter:
                                    feedback_placeholder.success(f"Correct! This is the letter {predicted_letter} (Confidence: {confidence:.2f}%)")
                                else:
                                    feedback_placeholder.warning(f"Keep trying! The model detected letter {predicted_letter} (Confidence: {confidence:.2f}%)")
                            else:
                                feedback_placeholder.info("No letter detected. Please try again.")
                else:
                    feedback_placeholder.info("No hand detected. Please show your hand to the camera.")
                
                # Display the frame
                camera_placeholder.image(rgb_frame, use_column_width=True)
                
                # Add a small delay to prevent overwhelming the UI
                time.sleep(0.01)
                
        finally:
            # Release the camera when done
            cap.release()
            camera_placeholder.empty()
            feedback_placeholder.empty()
            
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Camera Practice"])

if page == "Home":
    home_page()
else:
    camera_page() 