from roboflow import Roboflow
import os
import cv2
import numpy as np

class ASLModelHandler:
    def __init__(self, api_key=None):
        """
        Initialize the ASL model handler.
        :param api_key: Roboflow API key
        """
        if api_key is None:
            api_key = os.getenv('ROBOFLOW_API_KEY')
            if api_key is None:
                raise ValueError("No Roboflow API key provided. Please set ROBOFLOW_API_KEY environment variable.")
        
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace("anagh").project("american-sign-language-letters-1hqg4")
        self.model = self.project.version(1).model
        
    def predict_letter(self, frame):
        """
        Predict the ASL letter from a frame.
        :param frame: numpy array of the image frame
        :return: predicted letter and confidence score
        """
        # Save frame temporarily
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        try:
            # Get prediction
            prediction = self.model.predict(temp_path, confidence=40, overlap=30)
            
            if prediction:
                # Get the prediction with highest confidence
                best_pred = max(prediction, key=lambda x: x['confidence'])
                letter = best_pred['class']
                confidence = best_pred['confidence']
                return letter, confidence
            
            return None, 0.0
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def process_hand_landmarks(self, landmarks):
        """
        Process hand landmarks to extract features for classification.
        :param landmarks: MediaPipe hand landmarks
        :return: processed features
        """
        # Convert landmarks to a format suitable for the model
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features) 