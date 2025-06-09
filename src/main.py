"""Main application for hand tracking with gesture controls."""
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import sys
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os
from dotenv import load_dotenv
import speech_recognition as sr
import threading
from collections import deque
import math

from .tracking import HandTracker
from .smoothing import EMASmoother, KalmanSmoother
from .dictation import SpeechDictation

# Load environment variables
load_dotenv()

# Get configuration from environment variables
CAMERA_ID = int(os.getenv('CAMERA_ID', 0))
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', 960))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', 540))
UI_WIDTH = int(os.getenv('UI_WIDTH', 1280))
UI_HEIGHT = int(os.getenv('UI_HEIGHT', 720))
UI_TITLE = os.getenv('UI_TITLE', 'Hand Control System')
UI_FPS = int(os.getenv('UI_FPS', 30))

# Drag smoothing settings
SMOOTH_WINDOW = 6  # Number of deltas to average
SMOOTH_ALPHA = 0.65  # Smoothing factor (0.5-0.7 is sweet spot)
PINCH_THRESHOLD = 0.05  # Reduced from 0.1 for more precise clicking
DRAG_DELAY = 0.3  # Seconds to hold pinch before drag starts
SCROLL_SPEED = 0.1  # Slower scrolling
SCROLL_AMOUNT = 3  # Scroll amount
BOX_GROW = 0.08  # How fast the box expands
BOX_DECAY = 0.001  # How fast the box shrinks

class SimpleGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI_TITLE)
        self.setGeometry(100, 100, UI_WIDTH, UI_HEIGHT)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C2C2C;
            }
            QLabel {
                color: #FFFFFF;
                background-color: #1E1E1E;
                border: 2px solid #3C3C3C;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #3C3C3C;
                color: #FFFFFF;
                border: 2px solid #4C4C4C;
                border-radius: 5px;
                padding: 8px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
            }
            QPushButton:pressed {
                background-color: #5C5C5C;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create video display with border
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(640, 480)
        self.display.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                border: 2px solid #3C3C3C;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.display)
        
        # Create control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Tracking")
        self.start_button.clicked.connect(self.start_tracking)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Tracking")
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(button_layout)
        
        # Create status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                border: 2px solid #3C3C3C;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        main_layout.addWidget(self.status_label)
        
        # Initialize MediaPipe with optimized settings
        print("Initializing MediaPipe...")
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # Use lite model for better performance
        )
        
        # Initialize camera
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(CAMERA_ID)
        if not self.cap.isOpened():
            self.status_label.setText("Error: Could not open camera")
            return
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize hand tracker
        self.tracker = HandTracker(camera=CAMERA_ID, max_hands=1, draw=True)
        
        # Initialize smoothers
        self.position_smoother = EMASmoother(alpha=SMOOTH_ALPHA)
        self.kalman_smoother = KalmanSmoother()
        
        # Initialize speech dictation
        self.dictation = SpeechDictation()
        
        # Initialize gesture states
        self.click_state = False
        self.enter_state = False
        self.scroll_state = False
        self.drag_state = False
        self.dictation_state = False
        
        # Initialize drag control
        self.pinch_start_time = 0
        self.drag_buf = deque(maxlen=SMOOTH_WINDOW)
        self.anchor_hand = (0, 0)
        self.anchor_cur = (0, 0)
        
        # Initialize scroll control
        self.last_scroll_time = time.time()
        
        # Initialize dynamic bounding box
        self.min_x, self.max_x = 0.0, 1.0
        self.min_y, self.max_y = 0.0, 1.0
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize tracking state
        self.is_tracking = False
        
    def start_tracking(self):
        """Start hand tracking."""
        if not self.is_tracking:
            self.is_tracking = True
            self.timer.start(30)  # ~30 FPS
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Tracking started")
            
    def stop_tracking(self):
        """Stop hand tracking."""
        if self.is_tracking:
            self.is_tracking = False
            self.timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Tracking stopped")
            
    def _smooth_cursor(self, target_x: float, target_y: float) -> tuple[int, int]:
        """Smoothly interpolate cursor position."""
        # Get current cursor position
        current_x, current_y = pyautogui.position()
        
        # Apply exponential moving average smoothing
        smooth_x, smooth_y = self.position_smoother(target_x, target_y)
        
        # Apply Kalman filter for additional smoothing
        kalman_x, kalman_y = self.kalman_smoother(smooth_x, smooth_y)
        
        return int(kalman_x), int(kalman_y)
        
    def _update_drag_filtered(self, target_x: float, target_y: float):
        """Update cursor position with direct movement."""
        # Add to drag buffer
        self.drag_buf.append((target_x, target_y))
        
        # Calculate average position
        avg_x = sum(x for x, _ in self.drag_buf) / len(self.drag_buf)
        avg_y = sum(y for _, y in self.drag_buf) / len(self.drag_buf)
        
        # Move cursor
        pyautogui.dragTo(int(avg_x), int(avg_y), duration=0.01, button='left')
        
    def _update_bounding_box(self, x: float, y: float):
        """Update the dynamic bounding box."""
        # Expand box if hand moves outside
        if x < self.min_x:
            self.min_x -= BOX_GROW * (self.min_x - x)
        if x > self.max_x:
            self.max_x += BOX_GROW * (x - self.max_x)
        if y < self.min_y:
            self.min_y -= BOX_GROW * (self.min_y - y)
        if y > self.max_y:
            self.max_y += BOX_GROW * (y - self.max_y)
            
        # Slowly shrink box when hand stays inside
        self.min_x = max(0, self.min_x + BOX_DECAY)
        self.max_x = min(1, self.max_x - BOX_DECAY)
        self.min_y = max(0, self.min_y + BOX_DECAY)
        self.max_y = min(1, self.max_y - BOX_DECAY)
        
    def _map_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Map hand coordinates to screen coordinates with stretch-band edge for all directions."""
        # Normalize coordinates to 0-1 range
        norm_x = float(x)  # No longer inverting X coordinate
        norm_y = float(y)  # Keep Y as is
        
        # Apply stretch-band to all edges
        # Bottom edge (60-100%) - increased boost area and strength
        if norm_y > 0.60:
            extra = (norm_y - 0.60) / 0.40  # 0-1 inside the band
            norm_y = 0.60 + extra * 0.8  # amplify by +80%
            
        # Top edge (0-5%) - weakened boost
        elif norm_y < 0.05:
            extra = (0.05 - norm_y) / 0.05  # 0-1 inside the band
            norm_y = 0.05 - extra * 0.2  # reduced to +20%
            
        # Right edge (95-100%) - weakened boost
        if norm_x > 0.95:
            extra = (norm_x - 0.95) / 0.05  # 0-1 inside the band
            norm_x = 0.95 + extra * 0.2  # reduced to +20%
        # Left edge (0-5%) - weakened boost
        elif norm_x < 0.05:
            extra = (0.05 - norm_x) / 0.05  # 0-1 inside the band
            norm_x = 0.05 - extra * 0.2  # reduced to +20%
        
        # Map to screen coordinates
        screen_x = norm_x * pyautogui.size()[0]
        screen_y = norm_y * pyautogui.size()[1]
        
        # Clamp to screen bounds
        screen_x = max(0, min(screen_x, pyautogui.size()[0]))
        screen_y = max(0, min(screen_y, pyautogui.size()[1]))
        
        return int(screen_x), int(screen_y)
        
    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        
    def _is_extended(self, tip: tuple[float, float], pip: tuple[float, float]) -> bool:
        """Check if a finger is extended."""
        return tip[1] < pip[1]
        
    def _is_curled(self, tip: tuple[float, float], pip: tuple[float, float]) -> bool:
        """Check if a finger is curled."""
        return tip[1] > pip[1]
            
    def update_frame(self):
        try:
            if not self.is_tracking:
                return
                
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.setText("Error reading frame")
                return
                
            # Flip frame horizontally for display
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    
                    # Get all needed landmark points
                    thumb_tip = (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y)
                    index_tip = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
                    middle_tip = (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y)
                    ring_tip = (hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y)
                    pinky_tip = (hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y)
                    
                    # Get PIP joints for extension check
                    index_pip = (hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y)
                    middle_pip = (hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y)
                    ring_pip = (hand_landmarks.landmark[14].x, hand_landmarks.landmark[14].y)
                    pinky_pip = (hand_landmarks.landmark[18].x, hand_landmarks.landmark[18].y)
                    thumb_pip = (hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y)
                    
                    # Get cursor point from middle knuckle
                    cursor_pt = (hand_landmarks.landmark[14].x, hand_landmarks.landmark[14].y)
                    
                    # Update dynamic bounding box
                    self._update_bounding_box(cursor_pt[0], cursor_pt[1])
                    
                    # Map to screen coordinates with dynamic box and non-linear boost
                    target_x, target_y = self._map_to_screen(*cursor_pt)
                    
                    # Check for drag gesture first
                    if self._distance(index_tip, thumb_tip) < PINCH_THRESHOLD:
                        current_time = time.time()
                        if not self.drag_state:
                            if self.pinch_start_time == 0:
                                self.pinch_start_time = current_time
                                self.anchor_hand = cursor_pt
                                self.anchor_cur = pyautogui.position()
                            elif current_time - self.pinch_start_time >= DRAG_DELAY:
                                pyautogui.mouseDown()
                                self.drag_state = True
                    else:
                        if self.drag_state:
                            pyautogui.mouseUp()
                            self.drag_state = False
                        self.pinch_start_time = 0
                    
                    # Update cursor position
                    if self.drag_state:
                        # Use smooth dragging
                        self._update_drag_filtered(target_x, target_y)
                    else:
                        # Use normal cursor movement
                        screen_x, screen_y = self._smooth_cursor(target_x, target_y)
                        pyautogui.moveTo(screen_x, screen_y, _pause=False)
                    
                    # Left click: pointer and thumb touch with tighter threshold
                    if self._distance(index_tip, thumb_tip) < PINCH_THRESHOLD:
                        if not self.click_state:
                            pyautogui.click()
                            self.click_state = True
                    else:
                        self.click_state = False
                    
                    # Enter: only thumb extended
                    if (self._is_extended(thumb_tip, thumb_pip) and
                        not self._is_extended(index_tip, index_pip) and
                        not self._is_extended(middle_tip, middle_pip) and
                        not self._is_extended(ring_tip, ring_pip) and
                        not self._is_extended(pinky_tip, pinky_pip)):
                        if not self.enter_state:
                            pyautogui.press('enter')
                            self.enter_state = True
                    else:
                        self.enter_state = False
                    
                    # Dictation: index, middle, and ring fingers curled, others extended
                    if (self._is_curled(index_tip, index_pip) and
                        self._is_curled(middle_tip, middle_pip) and
                        self._is_curled(ring_tip, ring_pip) and
                        self._is_extended(thumb_tip, thumb_pip) and
                        self._is_extended(pinky_tip, pinky_pip)):
                        if not self.dictation_state:
                            # Start dictation in a separate thread to prevent lag
                            threading.Thread(target=self.dictation.start, daemon=True).start()
                            self.dictation_state = True
                    else:
                        if self.dictation_state:
                            # Stop dictation in a separate thread
                            threading.Thread(target=self.dictation.stop, daemon=True).start()
                            self.dictation_state = False
                    
                    # Scroll: pointer and middle extended (down) or pointer+middle+ring (up)
                    if (self._is_extended(index_tip, index_pip) and
                        self._is_extended(middle_tip, middle_pip) and
                        not self._is_extended(pinky_tip, pinky_pip)):
                        
                        current_time = time.time()
                        if current_time - self.last_scroll_time >= SCROLL_SPEED:
                            if self._is_extended(ring_tip, ring_pip):  # Scroll up
                                pyautogui.scroll(SCROLL_AMOUNT)
                            else:  # Scroll down
                                pyautogui.scroll(-SCROLL_AMOUNT)
                            self.last_scroll_time = current_time
                        
                        if not self.scroll_state:
                            self.scroll_state = True
                    else:
                        self.scroll_state = False
                    
                    # Draw stretch-band visualization
                    h, w = frame.shape[:2]
                    # Draw band thresholds
                    band_top = int(0.05 * h)
                    band_bottom = int(0.60 * h)  # Changed to 60%
                    band_left = int(0.05 * w)
                    band_right = int(0.95 * w)
                    
                    # Draw horizontal bands
                    cv2.line(frame, (0, band_top), (w, band_top), (0, 255, 0), 2)
                    cv2.line(frame, (0, band_bottom), (w, band_bottom), (0, 255, 0), 2)
                    # Draw vertical bands
                    cv2.line(frame, (band_left, 0), (band_left, h), (0, 255, 0), 2)
                    cv2.line(frame, (band_right, 0), (band_right, h), (0, 255, 0), 2)
                    
                    # Draw stretch areas
                    # Bottom stretch (increased area and strength)
                    for i in range(band_bottom, h, 5):
                        y_norm = float(i) / h
                        if y_norm > 0.60:
                            extra = (y_norm - 0.60) / 0.40
                            y_stretched = 0.60 + extra * 0.8
                            x_pos = int(y_stretched * w)
                            cv2.line(frame, (0, i), (x_pos, i), (0, 0, 255), 1)
                    
                    # Top stretch (weakened)
                    for i in range(0, band_top, 5):
                        y_norm = float(i) / h
                        if y_norm < 0.05:
                            extra = (0.05 - y_norm) / 0.05
                            y_stretched = 0.05 - extra * 0.2  # reduced to +20%
                            x_pos = int(y_stretched * w)
                            cv2.line(frame, (0, i), (x_pos, i), (0, 0, 255), 1)
                    
                    # Right stretch (weakened)
                    for i in range(band_right, w, 5):
                        x_norm = float(i) / w
                        if x_norm > 0.95:
                            extra = (x_norm - 0.95) / 0.05
                            x_stretched = 0.95 + extra * 0.2  # reduced to +20%
                            y_pos = int(x_stretched * h)
                            cv2.line(frame, (i, 0), (i, y_pos), (0, 0, 255), 1)
                    
                    # Left stretch (weakened)
                    for i in range(0, band_left, 5):
                        x_norm = float(i) / w
                        if x_norm < 0.05:
                            extra = (0.05 - x_norm) / 0.05
                            x_stretched = 0.05 - extra * 0.2  # reduced to +20%
                            y_pos = int(x_stretched * h)
                            cv2.line(frame, (i, 0), (i, y_pos), (0, 0, 255), 1)
                    
                self.status_label.setText("Hand detected")
            else:
                self.status_label.setText("No hand detected")
            
            # Convert to Qt image
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale and display
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.display.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Error in update_frame: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close event."""
        self.cap.release()
        self.mp_hands.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = SimpleGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 