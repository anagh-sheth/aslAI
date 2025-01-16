import cv2
import streamlit as st
import mediapipe as mp
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow with your API key
rf = Roboflow(api_key="iFjPUVPeTAGflBUC7OvK")

# Load the project and version
project = rf.workspace("anagh").project("american-sign-language-letters-1hqg4")  # Roboflow 100 Sign Language Project
model = project.version(1).model


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="iFjPUVPeTAGflBUC7OvK"
)
 

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, #f0f8ff, #e6f2ff);
    }
    .main-title {
        font-size: 2.5rem;
        color: #007bff;
        font-weight: bold;
        text-align: center;
    }
    .description {
        font-size: 1.2rem;
        color: #333;
        text-align: center;
        margin-top: -15px;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .feature-card {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px;
        text-align: center;
    }
    .feature-card h3 {
        color: #007bff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set up page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Camera"])

# Page 1: Home
if page == "Home":
    st.markdown('<h1 class="main-title">Learn American Sign Language with AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Master ASL through interactive lessons and real time feedback</p>', unsafe_allow_html=True)

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
    st.title("ASL Letter Detection")
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
    
