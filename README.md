# ASL Recognition Web Application

This web application uses artificial intelligence to help users learn and practice American Sign Language (ASL) through real-time hand sign detection.

## Features

- Real-time ASL letter detection using computer vision
- Interactive learning experience with immediate feedback
- Practice mode for ASL letters A-Z
- User-friendly interface built with Streamlit
- High-accuracy hand tracking using MediaPipe

## Technical Details

- Mean Average Precision (mAP): 96.1%
- Precision: 88.6%
- Recall: 92.1%

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Navigate to the Camera Practice page using the sidebar
2. Select a letter you want to practice
3. Click "Start Camera" to begin
4. Make the ASL sign for the selected letter
5. Get real-time feedback on your sign

## Requirements

- Python 3.7+
- Webcam
- Internet connection (for initial setup)

## Technologies Used

- Streamlit
- OpenCV
- MediaPipe
- Roboflow
- NumPy
- PIL (Python Imaging Library)

## Note

Make sure your webcam is properly connected and accessible before starting the camera practice session. 