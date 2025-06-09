"""Camera interface for hand tracking."""
import cv2
import numpy as np

class Camera:
    def __init__(self):
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # drop old frames
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        print("Camera initialized successfully")
        
    def read(self):
        """Read a frame from the camera."""
        return self.cap.read()
        
    def release(self):
        """Release the camera resources."""
        self.cap.release()
        
    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.release() 