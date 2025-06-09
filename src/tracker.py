#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import time

class HandTracker:
    def __init__(self, camera_index=0):
        print("Initializing hand tracker...")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # Fastest model
        )
        
        # Drawing settings
        self.mp_drawing = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        self.draw_spec = self.mp_drawing.DrawingSpec(
            color=(0,255,0),
            thickness=1,
            circle_radius=1
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        self._init_camera()
        
        # Performance tracking
        self.last_frame_time = 0
        self.fps = 0
        self.processing_time = 0
        
        print("Hand tracker initialized successfully")

    def _init_camera(self):
        """Initialize camera with error handling"""
        try:
            # Verify camera is working
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise Exception("Could not read from camera")
                
            print("Camera initialized successfully")
            
        except Exception as e:
            print(f"Error initializing camera: {str(e)}")
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            raise

    def get_landmarks(self):
        """Return the first hand's landmark list (21 normalized points) or None."""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Update FPS calculation
        current_time = cv2.getTickCount()
        if self.last_frame_time > 0:
            self.fps = cv2.getTickFrequency() / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        # Process frame
        start_time = cv2.getTickCount()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)
        
        # Calculate processing time
        self.processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        # Return landmarks if found
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0].landmark
        return None

    def draw_hands(self, frame):
        """Overlay the hand landmarks on the frame (optional for debugging)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    self.hand_connections,
                    self.draw_spec,
                    self.draw_spec
                )
        
        # Draw FPS and processing time
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Process: {self.processing_time*1000:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def close(self):
        """Clean up resources"""
        try:
            if self.cap is not None:
                self.cap.release()
            self.mp_hands.close()
        except Exception as e:
            print(f"Error closing hand tracker: {str(e)}")
