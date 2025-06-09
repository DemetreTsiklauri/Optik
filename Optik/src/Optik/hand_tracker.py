import cv2
import mediapipe as mp
import time

class HandTracker:
    def __init__(self):
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.frame = None
        self.last_landmarks = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30  # Target 30 FPS
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize the camera with retries"""
        max_retries = 3
        for i in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    # Set camera properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    return True
            except Exception as e:
                print(f"Camera initialization attempt {i+1} failed: {e}")
                if self.cap is not None:
                    self.cap.release()
                time.sleep(1)
        return False

    def next_frame(self):
        """Get the next frame with timing control"""
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return self.frame

        if self.cap is None or not self.cap.isOpened():
            if not self.initialize_camera():
                return None

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return None

        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame_time = current_time
        return self.frame

    def get_landmarks(self, frame):
        """Get hand landmarks with error handling"""
        if frame is None:
            return None

        try:
            results = self.hands.process(frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                self.last_landmarks = landmarks
                return landmarks
            return self.last_landmarks
        except Exception as e:
            print(f"Error processing hand landmarks: {e}")
            return self.last_landmarks

    def close(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        self.hands.close() 