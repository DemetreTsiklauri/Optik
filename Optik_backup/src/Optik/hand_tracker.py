#!/usr/bin/env python3
import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, camera_index=0):
        # open camera
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Enable two hand tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing     = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        self.draw_spec      = self.mp_drawing.DrawingSpec(color=(0,255,0),
                                                          thickness=2,
                                                          circle_radius=2)
        self.results = None

    def next_frame(self):
        """Grab the next frame from camera, or return None."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_landmarks(self, frame):
        """Return the first hand's landmark list (21 normalized points) or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_hands.process(rgb)
        self.results = res
        if res.multi_hand_landmarks:
            # Return the first hand's landmarks
            return res.multi_hand_landmarks[0].landmark
        return None

    def draw_hands(self, frame):
        """Overlay the hand landmarks on the frame (optional for debugging)."""
        if self.results and self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    self.hand_connections,
                    self.draw_spec,
                    self.draw_spec
                )
