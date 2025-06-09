import pyautogui
import numpy as np
import time
from pykalman import KalmanFilter

class HandController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x = None
        self.prev_y = None
        self.smooth_factor = 0.9
        self.click_cooldown = 0.5
        self.drag_cooldown = 0.2
        self.enter_cooldown = 1.0
        self.dictation_cooldown = 2.0
        self.last_click = 0
        self.last_drag = 0
        self.last_enter = 0
        self.last_dictation = 0
        self.dragging = False
        self.bottom_screen_triggered = False
        self.dynamic_min_y = 1.0
        self.dynamic_max_y = 0.0
        self.pinch_threshold = 0.04
        self.enter_threshold = 0.08
        self.dictation_hold_time = 1.0
        self.dictation_start = None
        self.hand_detected = False
        self.hand_detection_timeout = 0.3  # Reduced timeout for faster response
        self.last_hand_detection = 0
        self.bounds_margin = 0.1  # Margin for bounds checking
        self.velocity_threshold = 0.5  # Threshold for velocity-based smoothing
        
        # Initialize Kalman filter with more aggressive smoothing
        self.kf = KalmanFilter(
            initial_state_mean=[0, 0, 0, 0],  # [x, y, dx, dy]
            n_dim_obs=2,  # We only observe x and y
            n_dim_state=4,  # State includes position and velocity
            transition_matrices=np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 0.8, 0],  # Reduced velocity persistence
                [0, 0, 0, 0.8]
            ]),
            observation_matrices=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ]),
            transition_covariance=0.005 * np.eye(4),  # Reduced process noise
            observation_covariance=0.05 * np.eye(2),  # Reduced measurement noise
            initial_state_covariance=np.eye(4)
        )
        self.kf_state = None
        self.kf_covariance = None
        self.last_positions = []  # Store last N positions for velocity calculation
        self.max_position_history = 5

    def _is_hand_detected(self, landmarks):
        if landmarks is None or len(landmarks) < 21:
            return False
        
        # Check if hand landmarks are within reasonable bounds with margin
        x_coords = [x for x, _ in landmarks]
        y_coords = [y for _, y in landmarks]
        
        # Check if any landmark is outside the bounds (with margin)
        if (min(x_coords) < -self.bounds_margin or max(x_coords) > 1 + self.bounds_margin or
            min(y_coords) < -self.bounds_margin or max(y_coords) > 1 + self.bounds_margin):
            return False
            
        return True

    def _calculate_velocity(self, x, y):
        """Calculate velocity based on recent positions"""
        self.last_positions.append((x, y))
        if len(self.last_positions) > self.max_position_history:
            self.last_positions.pop(0)
        
        if len(self.last_positions) < 2:
            return 0, 0
            
        dx = (self.last_positions[-1][0] - self.last_positions[0][0]) / len(self.last_positions)
        dy = (self.last_positions[-1][1] - self.last_positions[0][1]) / len(self.last_positions)
        return dx, dy

    def _apply_velocity_smoothing(self, x, y, dx, dy):
        """Apply additional smoothing based on velocity"""
        velocity = np.sqrt(dx*dx + dy*dy)
        if velocity > self.velocity_threshold:
            # If velocity is high, reduce the movement
            smoothing = 0.7
        else:
            # Normal smoothing for low velocity
            smoothing = 0.9
            
        if self.prev_x is not None and self.prev_y is not None:
            x = self.prev_x * smoothing + x * (1 - smoothing)
            y = self.prev_y * smoothing + y * (1 - smoothing)
            
        self.prev_x, self.prev_y = x, y
        return x, y

    def _detect_gestures(self, landmarks, now):
        gestures = {
            'click': False,
            'drag': False,
            'enter': False,
            'dictation': False,
            'scroll_up': False,
            'scroll_down': False,
            'right_click': False
        }
        
        # Click gesture (pinch)
        pinch = np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[8])) < self.pinch_threshold
        if pinch and now - self.last_click > self.click_cooldown:
            gestures['click'] = True
            self.last_click = now

        # Drag gesture (hold pinch)
        if pinch and not self.dragging and now - self.last_drag > self.drag_cooldown:
            gestures['drag'] = True
            self.last_drag = now

        # Enter gesture (fist)
        fingers_curled = all(np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[0])) < self.enter_threshold 
                           for i in [8,12,16,20])
        palm_facing = landmarks[0][1] < landmarks[9][1]
        if fingers_curled and palm_facing and now - self.last_enter > self.enter_cooldown:
            gestures['enter'] = True
            self.last_enter = now

        # Dictation gesture (thumb and pinky out, rest curled)
        thumb_out = np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[2])) > 0.08
        pinky_out = np.linalg.norm(np.array(landmarks[20]) - np.array(landmarks[18])) > 0.08
        rest_curled = all(np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[0])) < 0.07 
                         for i in [8,12,16])
        if thumb_out and pinky_out and rest_curled:
            if self.dictation_start is None:
                self.dictation_start = now
            elif now - self.dictation_start > self.dictation_hold_time and now - self.last_dictation > self.dictation_cooldown:
                gestures['dictation'] = True
                self.last_dictation = now
                self.dictation_start = None
        else:
            self.dictation_start = None

        # Scroll gestures (index and middle finger pointing up/down)
        index_up = landmarks[8][1] < landmarks[6][1]
        middle_up = landmarks[12][1] < landmarks[10][1]
        if index_up and middle_up:
            if landmarks[8][1] < landmarks[12][1]:
                gestures['scroll_up'] = True
            else:
                gestures['scroll_down'] = True

        # Right click gesture (ring and pinky finger pointing up)
        ring_up = landmarks[16][1] < landmarks[14][1]
        pinky_up = landmarks[20][1] < landmarks[18][1]
        if ring_up and pinky_up and not index_up and not middle_up:
            gestures['right_click'] = True

        return gestures

    def update(self, landmarks):
        now = time.time()
        hand_detected = self._is_hand_detected(landmarks)
        
        # Update hand detection state
        if hand_detected:
            self.hand_detected = True
            self.last_hand_detection = now
        elif now - self.last_hand_detection > self.hand_detection_timeout:
            self.hand_detected = False
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            # Clear position history when hand is lost
            self.last_positions.clear()
            return

        if not self.hand_detected:
            return

        # Process hand tracking
        y = landmarks[8][1]
        self.dynamic_min_y = min(self.dynamic_min_y, y)
        self.dynamic_max_y = max(self.dynamic_max_y, y)
        mapped_y = (y - self.dynamic_min_y) / max(0.01, (self.dynamic_max_y - self.dynamic_min_y))
        mapped_y = np.clip(mapped_y, 0, 1)
        x = landmarks[8][0]

        # Calculate velocity
        dx, dy = self._calculate_velocity(x, mapped_y)

        # Apply Kalman filter
        if self.kf_state is None:
            self.kf_state = self.kf.initial_state_mean
            self.kf_covariance = self.kf.initial_state_covariance
        
        measurement = np.array([x, mapped_y])
        self.kf_state, self.kf_covariance = self.kf.filter_update(
            filtered_state_mean=self.kf_state,
            filtered_state_covariance=self.kf_covariance,
            observation=measurement
        )
        
        smoothed_x, smoothed_y = self.kf_state[0], self.kf_state[1]
        
        # Apply additional velocity-based smoothing
        smoothed_x, smoothed_y = self._apply_velocity_smoothing(smoothed_x, smoothed_y, dx, dy)
        
        screen_x = int(smoothed_x * self.screen_w)
        screen_y = int(smoothed_y * self.screen_h)
        
        # Move cursor
        pyautogui.moveTo(screen_x, screen_y)

        # Process gestures
        gestures = self._detect_gestures(landmarks, now)
        
        # Execute gestures
        if gestures['click']:
            pyautogui.click()
        if gestures['drag']:
            pyautogui.mouseDown()
            self.dragging = True
        elif not gestures['drag'] and self.dragging:
            pyautogui.mouseUp()
            self.dragging = False
        if gestures['enter']:
            pyautogui.press('enter')
        if gestures['dictation']:
            print("Dictation gesture triggered!")
        if gestures['scroll_up']:
            pyautogui.scroll(10)
        if gestures['scroll_down']:
            pyautogui.scroll(-10)
        if gestures['right_click']:
            pyautogui.rightClick()

        # Bottom screen gesture
        if mapped_y > 0.95 and not self.bottom_screen_triggered:
            print("Bottom screen gesture triggered!")
            self.bottom_screen_triggered = True
        elif mapped_y <= 0.95:
            self.bottom_screen_triggered = False

    def close(self):
        if self.dragging:
            pyautogui.mouseUp()
            self.dragging = False 