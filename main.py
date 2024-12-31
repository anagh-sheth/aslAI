import cv2
import streamlit as st
import mediapipe as mp

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
    st.title("Face and Hand Detection")

    # Placeholder for the video frame
    frame_placeholder = st.empty()

    # Add a button to stop the video
    stop_button = st.button("Stop")

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Initialize Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Check if the camera is opened
    if not cap.isOpened():
        st.error("Could not open the camera.")
    else:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame. Exiting.")
                break

            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Detect hands
            hand_results = hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert frame to RGB for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

    # Release the camera and clean up
    cap.release()
    cv2.destroyAllWindows()
