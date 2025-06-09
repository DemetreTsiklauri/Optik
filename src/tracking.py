"""Low‑level MediaPipe hand tracking and landmark helpers."""
from __future__ import annotations

import cv2
import mediapipe as mp
from typing import Tuple, List

mp_hands = mp.solutions.hands

Pose = List[Tuple[int, int]]  # 21 landmark screen‑pixel integer coords

class HandTracker:
    def __init__(self, camera: int = 0, max_hands: int = 1, draw: bool = False):
        self.cap = cv2.VideoCapture(camera)
        self.draw = draw
        self.hands = mp_hands.Hands(
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        # Get frame size once for mapping
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        landmarks: Pose | None = None
        if results.multi_hand_landmarks:
            # convert normalized coords to pixel ints
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = [
                (
                    int(lm.x * self.frame_w),
                    int(lm.y * self.frame_h),
                )
                for lm in hand_landmarks.landmark
            ]
            if self.draw:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        return frame, landmarks

    def release(self):
        self.cap.release() 