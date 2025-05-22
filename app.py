import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import time
from utils.model_handler import ASLModelHandler
import random

# Set page config
st.set_page_config(
    page_title="ASL Recognition",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0 !important;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: black;
    }
    .title-container {
        background-color: transparent;
        padding: 1rem 2rem;
        text-align: center;
        color: white;
    }
    .camera-page-content {
        padding-top: 2rem;
    }
    .feature-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
        color: black;
    }
    .feature-container h4 {
        color: #2c3e50;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .feature-container:hover {
        transform: translateY(-5px);
    }
    .metric-card {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .cta-button {
        background-color: #2ecc71;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .cta-button:hover {
        background-color: #27ae60;
        transform: scale(1.05);
    }
    /* Style for Streamlit buttons */
    .stButton button {
        color: white !important;
        background-color: #2980b9 !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 5px !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background-color: #3498db !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
if 'current_word' not in st.session_state:
    st.session_state.current_word = ''
if 'current_letter_index' not in st.session_state:
    st.session_state.current_letter_index = 0
if 'word_completed' not in st.session_state:
    st.session_state.word_completed = False

# List of simple words for practice
SIMPLE_WORDS = [
    "cat", "dog", "hat", "sun", "run", "jump", "play", "book",
    "tree", "fish", "bird", "star", "moon", "cake", "ball", "home",
    "toy", "car", "bus", "pen", "cup", "box", "red", "blue"
]

def get_random_word():
    return random.choice(SIMPLE_WORDS)

def home_page():
    # Title Section
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.title("ASL Recognition AI")
    st.markdown("""
    <h3 style='text-align: center; color: white;'>
        Learn American Sign Language with the power of Artificial Intelligence
    </h3>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h2>96.1%</h2>
                <p>Mean Average Precision</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h2>88.6%</h2>
                <p>Precision Score</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h2>92.1%</h2>
                <p>Recall Rate</p>
            </div>
        """, unsafe_allow_html=True)

    # Main Features Section
    st.markdown("### 🌟 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-container">
                <h4>🎯 Real-time Detection</h4>
                <p>Experience instant hand sign recognition through your webcam with advanced computer vision technology.</p>
            </div>
            
            <div class="feature-container">
                <h4>🤖 AI-Powered Learning</h4>
                <p>Our system uses state-of-the-art machine learning models trained on extensive ASL datasets.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="feature-container">
                <h4>📊 Accurate Feedback</h4>
                <p>Get immediate feedback on your hand signs with detailed accuracy metrics.</p>
            </div>
            
            <div class="feature-container">
                <h4>🎓 Interactive Practice</h4>
                <p>Practice ASL letters A-Z with our interactive camera system and improve your skills.</p>
            </div>
        """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("### 🔍 How It Works")
    st.markdown("""
        <div class="feature-container">
            <ol>
                <li>Select "Camera Practice" from the sidebar</li>
                <li>Choose a letter you want to practice</li>
                <li>Show your ASL sign to the camera</li>
                <li>Get instant feedback on your sign accuracy</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)


    # Call to Action
    st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <a href="#" class="cta-button" onclick="document.querySelector('.streamlit-expanderHeader').click()">
                Start Learning ASL Now! 👋
            </a>
        </div>
    """, unsafe_allow_html=True)

def camera_page():
    st.markdown('<div class="camera-page-content">', unsafe_allow_html=True)
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
                    
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            rgb_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                        
                        if st.session_state.model_handler:
                            predicted_letter, confidence = st.session_state.model_handler.predict_letter(frame)
                            
                            if predicted_letter:
                                if predicted_letter == selected_letter:
                                    feedback_placeholder.success(f"Correct! This is the letter {predicted_letter} (Confidence: {min(confidence * 100, 100):.2f}%)")
                                else:
                                    feedback_placeholder.warning(f"Keep trying! The model detected letter {predicted_letter} (Confidence: {min(confidence * 100, 100):.2f}%)")
                            else:
                                feedback_placeholder.info("No letter detected. Please try again.")
                else:
                    feedback_placeholder.info("No hand detected. Please show your hand to the camera.")
                
                # Display the frame
                camera_placeholder.image(rgb_frame, use_column_width=True)
                
                # Add a small delay
                time.sleep(0.01)
                
        finally:
            # Release the camera
            cap.release()
            camera_placeholder.empty()
            feedback_placeholder.empty()
    st.markdown('</div>', unsafe_allow_html=True)

def word_practice_page():
    st.markdown('<div class="camera-page-content">', unsafe_allow_html=True)
    st.title("ASL Word Practice 📝")
    
    # Initialize or reset word
    if not st.session_state.current_word or st.session_state.word_completed:
        st.session_state.current_word = get_random_word()
        st.session_state.current_letter_index = 0
        st.session_state.word_completed = False
    
    # Display current word and progress
    st.markdown(f"### Current Word: **{st.session_state.current_word.upper()}**")
    progress = st.session_state.current_letter_index / len(st.session_state.current_word)
    st.progress(progress)
    
    # Display which letter to sign
    current_letter = st.session_state.current_word[st.session_state.current_letter_index]
    st.markdown(f"### Sign the letter: **{current_letter.upper()}**")
    st.markdown(f"Letter {st.session_state.current_letter_index + 1} of {len(st.session_state.current_word)}")
    
    # New word button
    if st.button("Get New Word"):
        st.session_state.current_word = get_random_word()
        st.session_state.current_letter_index = 0
        st.session_state.word_completed = False
        st.experimental_rerun()
    
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
                    
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            rgb_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                        
                        if st.session_state.model_handler:
                            predicted_letter, confidence = st.session_state.model_handler.predict_letter(frame)
                            
                            if predicted_letter:
                                if predicted_letter.lower() == current_letter.lower():
                                    feedback_placeholder.success(f"Correct! Moving to next letter... (Confidence: {min(confidence * 100, 100):.2f}%)")
                                    time.sleep(1)  # Give user time to see the success message
                                    st.session_state.current_letter_index += 1
                                    
                                    # Check if word is completed
                                    if st.session_state.current_letter_index >= len(st.session_state.current_word):
                                        st.session_state.word_completed = True
                                        st.balloons()
                                        feedback_placeholder.success("🎉 Congratulations! Word completed! Click 'Get New Word' to continue.")
                                        break
                                    st.experimental_rerun()
                                else:
                                    feedback_placeholder.warning(f"Keep trying! The model detected letter {predicted_letter} (Confidence: {min(confidence * 100, 100):.2f}%)")
                            else:
                                feedback_placeholder.info("No letter detected. Please try again.")
                else:
                    feedback_placeholder.info("No hand detected. Please show your hand to the camera.")
                
                # Display the frame
                camera_placeholder.image(rgb_frame, use_column_width=True)
                
                # Add a small delay
                time.sleep(0.01)
                
        finally:
            # Release the camera
            cap.release()
            camera_placeholder.empty()
            feedback_placeholder.empty()
    st.markdown('</div>', unsafe_allow_html=True)

def toggle_camera():
    st.session_state.camera_running = not st.session_state.camera_running

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Home", "Camera Practice", "Word Practice"])

if page == "Home":
    home_page()
elif page == "Camera Practice":
    camera_page()
else:
    word_practice_page() 