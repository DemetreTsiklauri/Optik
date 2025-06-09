#!/usr/bin/env python3
import pyautogui, math, time, threading
import numpy as np
from collections import deque
import traceback
from filterpy.kalman import KalmanFilter
from typing import List, Tuple

# Constants for fine-tuning
PINCH_THRESH = 0.05
DRAG_DELAY = 0.1
SCROLL_PX = 9
SCROLL_HZ = 120
BOX_GROW = 0.1
KAL_Q, KAL_R = 0.03, 2.5     # cursor smoothness
VELOCITY_THRESHOLD = 0.01
DOUBLE_CLICK_DELAY = 0.3
SCROLL_SMOOTHING = 0.8
DRAG_SMOOTHING = 0.7
POSITION_SMOOTHING = 0.8   # Reduced for more direct movement
VELOCITY_BLEND = 0.8      # Reduced for more direct movement
POSITION_BLEND = 0.8      # Reduced for more direct movement
GESTURE_HISTORY_SIZE = 3
VELOCITY_WINDOW = 2       # Even more direct response
POSITION_HISTORY_SIZE = 3  # Even more direct response
DICTATION_SILENCE_TIMEOUT = 2.0
DEADZONE_THRESHOLD = 0.001  # Increased for more stable movement
SPEED_MULTIPLIER = 1.5      # Increased for faster movement
BOX_RESET_RATE = 0.1
BASE_SENSITIVITY = 1.0
EDGE_SENSITIVITY = 1.5

# Disable pyautogui delays
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_extended(landmarks, finger_tip_idx):
    """Check if a finger is extended"""
    return landmarks[finger_tip_idx].y < landmarks[finger_tip_idx - 2].y

class HandController:
    def __init__(self):
        print("Initializing hand controller...")
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Cursor smoothing
        self.last_x = None
        self.last_y = None
        self.alpha = 0.65  # Smoothing factor (0.5-0.7 is sweet spot)
        
        # Gesture states
        self.click_state = False
        self.enter_state = False
        self.scroll_state = False
        self.drag_state = False
        self.dictation_state = False
        
        # Drag control
        self.pinch_start_time = 0
        self.pinch_threshold = 0.3  # seconds to hold pinch before drag starts
        
        # Scroll control
        self.last_scroll_time = time.time()
        self.scroll_speed = 0.005  # Faster updates for smoother scrolling
        
        print("Hand controller initialized successfully")

    def _smooth_position(self, x: float, y: float) -> Tuple[float, float]:
        """Apply exponential moving average smoothing."""
        if self.last_x is None:
            self.last_x, self.last_y = x, y
            return x, y
            
        smooth_x = self.alpha * x + (1 - self.alpha) * self.last_x
        smooth_y = self.alpha * y + (1 - self.alpha) * self.last_y
        self.last_x, self.last_y = smooth_x, smooth_y
        return smooth_x, smooth_y

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _is_extended(tip: Tuple[float, float], pip: Tuple[float, float]) -> bool:
        return tip[1] < pip[1]

    def _get_landmark(self, landmarks, idx: int) -> Tuple[float, float]:
        if not landmarks or idx >= len(landmarks):
            return (0.0, 0.0)
        return (landmarks[idx].x, landmarks[idx].y)

    def _to_screen_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized coordinates (0-1) to screen coordinates, with horizontal inversion."""
        screen_x = int((1.0 - x) * self.screen_width)  # Invert x coordinate
        screen_y = int(y * self.screen_height)
        return screen_x, screen_y

    def process(self, landmarks):
        """Given 21 landmark points, execute gestures & return cursor point."""
        if not landmarks or len(landmarks) < 21:
            return self._to_screen_coords(0.5, 0.5)

        try:
            # Get cursor point from middle knuckle and smooth it
            cursor_pt = self._get_landmark(landmarks, 14)  # Middle knuckle
            smooth_x, smooth_y = self._smooth_position(*cursor_pt)
            
            # Get all needed landmark points
            thumb_tip = self._get_landmark(landmarks, 4)
            index_tip = self._get_landmark(landmarks, 8)
            middle_tip = self._get_landmark(landmarks, 12)
            ring_tip = self._get_landmark(landmarks, 16)
            pinky_tip = self._get_landmark(landmarks, 20)
            
            # Get PIP joints for extension check
            index_pip = self._get_landmark(landmarks, 6)
            middle_pip = self._get_landmark(landmarks, 10)
            ring_pip = self._get_landmark(landmarks, 14)
            pinky_pip = self._get_landmark(landmarks, 18)
            thumb_pip = self._get_landmark(landmarks, 2)

            # Left click: pointer and thumb touch
            if self._distance(index_tip, thumb_tip) < 0.1:
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

            # Scroll: pointer and middle extended (down) or pointer+middle+ring (up)
            if (self._is_extended(index_tip, index_pip) and
                self._is_extended(middle_tip, middle_pip) and
                not self._is_extended(pinky_tip, pinky_pip)):
                
                current_time = time.time()
                if current_time - self.last_scroll_time >= self.scroll_speed:
                    if self._is_extended(ring_tip, ring_pip):  # Scroll up
                        pyautogui.scroll(2)  # Increased scroll amount
                    else:  # Scroll down
                        pyautogui.scroll(-2)  # Increased scroll amount
                    self.last_scroll_time = current_time
                
                if not self.scroll_state:
                    self.scroll_state = True
            else:
                self.scroll_state = False

            # Drag: pointer and thumb pinch held for threshold time
            if self._distance(index_tip, thumb_tip) < 0.1:
                current_time = time.time()
                if not self.drag_state:
                    if self.pinch_start_time == 0:
                        self.pinch_start_time = current_time
                    elif current_time - self.pinch_start_time >= self.pinch_threshold:
                        pyautogui.mouseDown()
                        self.drag_state = True
            else:
                if self.drag_state:
                    pyautogui.mouseUp()
                    self.drag_state = False
                self.pinch_start_time = 0

            # Dictation mode: trigger on gesture release
            dictation_gesture = (self._is_extended(pinky_tip, pinky_pip) and
                               self._is_extended(thumb_tip, thumb_pip) and
                               not self._is_extended(index_tip, index_pip) and
                               not self._is_extended(middle_tip, middle_pip) and
                               not self._is_extended(ring_tip, ring_pip))
            
            if dictation_gesture:
                self.dictation_state = True
            elif self.dictation_state:  # Gesture was released
                try:
                    from .voice_mode import VoiceDictation
                    if not VoiceDictation.active():
                        threading.Thread(target=VoiceDictation.start, daemon=True).start()
                except Exception as e:
                    print(f"Dictation error: {str(e)}")
                self.dictation_state = False

        except Exception as e:
            print(f"Error in gesture processing: {str(e)}")
            return self._to_screen_coords(0.5, 0.5)

        return self._to_screen_coords(smooth_x, smooth_y)

    def start(self):
        """Start the hand controller."""
        print("Starting hand controller...")
        print("Hand controller started successfully")

    def stop(self):
        """Stop the hand controller."""
        print("Stopping hand controller...")
        print("Hand controller stopped successfully")

    def close(self):
        """Clean up resources."""
        self.stop()

    def _grow_box(self, pt):
        if pt.x < self.minx: self.minx -= BOX_GROW*(self.minx-pt.x)
        if pt.x > self.maxx: self.maxx += BOX_GROW*(pt.x-self.maxx)
        if pt.y < self.miny: self.miny -= BOX_GROW*(self.miny-pt.y)
        if pt.y > self.maxy: self.maxy += BOX_GROW*(pt.y-self.maxy)
        self.minx, self.maxx = max(0,self.minx), min(1,self.maxx)
        self.miny, self.maxy = max(0,self.miny), min(1,self.maxy)

    def _handle_pinch(self, lm):
        touching = dist(lm[4], lm[8]) < PINCH_THRESH
        now = time.time()
        if touching and self.pinch_t0 is None:
            self.pinch_t0    = now
            self.anchor_hand = self._map(lm[5].x, lm[5].y)
            self.anchor_cur  = pyautogui.position()

        if touching and not self.drag and now - self.pinch_t0 >= DRAG_DELAY:
            pyautogui.mouseDown(); self.drag = True

        if not touching:
            if self.drag:
                pyautogui.mouseUp(); self.drag = False
            elif self.pinch_t0 and now - self.pinch_t0 < DRAG_DELAY:
                pyautogui.click()
            self.pinch_t0 = None

    def _update_drag_raw(self, lm):
        hx,hy = self._map(lm[5].x, lm[5].y)
        dx,dy = hx-self.anchor_hand[0], hy-self.anchor_hand[1]
        nx,ny = self.anchor_cur[0]+dx, self.anchor_cur[1]+dy
        pyautogui.dragTo(nx, ny, duration=0, button='left', _pause=False)
        self.kf.x[:2] = [nx, ny]

    def _reset_pinch(self):
        if self.drag: pyautogui.mouseUp(); self.drag=False
        self.pinch_t0 = None

    def _detect_scroll(self, lm):
        idx   = lm[8].y  < lm[6].y
        mid   = lm[12].y < lm[10].y
        ring  = lm[16].y < lm[14].y
        pink  = lm[20].y < lm[18].y
        if idx and mid and not ring and not pink:
            self.scroll_direction = "down"
            self.target_scroll = -self.scroll_speed
        elif idx and mid and ring and not pink:
            self.scroll_direction = "up"
            self.target_scroll = self.scroll_speed
        else:
            self.scroll_direction = None
            self.target_scroll = 0

    def _detect_gestures(self, landmarks):
        """Detect various gestures and trigger corresponding actions"""
        try:
            # Pinch gesture (click and drag)
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            pinch_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            
            if pinch_distance < PINCH_THRESH:
                now = time.time()
                if now - self.last_click_time > self.click_cooldown:
                    if not self.is_dragging:
                        pyautogui.mouseDown()
                        self.is_dragging = True
                        self.drag_start_time = now
                    self.last_click_time = now
            else:
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
            
            # Enter gesture (all fingers curled except thumb)
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # More precise enter gesture detection
            fingers_curled = all(tip.y > thumb_tip.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip])
            if fingers_curled:
                now = time.time()
                if now - self.enter_gesture_time > self.enter_gesture_cooldown:
                    pyautogui.press('enter')
                    self.enter_gesture_time = now
            
            # Scroll gestures with improved detection
            index_up = index_tip.y < landmarks[6].y
            middle_up = middle_tip.y < landmarks[10].y
            ring_up = ring_tip.y < landmarks[14].y
            pinky_up = pinky_tip.y < landmarks[18].y
            
            # Scroll down: index and middle up, others down
            if index_up and middle_up and not ring_up and not pinky_up:
                self.scroll_direction = "down"
                self.target_scroll = -self.scroll_speed
                pyautogui.scroll(-self.scroll_speed)  # Direct scroll
            # Scroll up: index, middle, and ring up, pinky down
            elif index_up and middle_up and ring_up and not pinky_up:
                self.scroll_direction = "up"
                self.target_scroll = self.scroll_speed
                pyautogui.scroll(self.scroll_speed)  # Direct scroll
            else:
                self.scroll_direction = None
                self.target_scroll = 0
            
            # Voice transcription gesture (pinky and thumb out, others curled)
            pinky_out = pinky_tip.x > landmarks[19].x + 0.03
            thumb_out = thumb_tip.x < landmarks[3].x - 0.03
            index_cur = index_tip.y > landmarks[6].y
            mid_cur = middle_tip.y > landmarks[10].y
            ring_cur = ring_tip.y > landmarks[14].y
            
            if pinky_out and thumb_out and index_cur and mid_cur and ring_cur:
                now = time.time()
                if not self.speech_active and now - self.voice_search_time > self.voice_search_cooldown:
                    pyautogui.hotkey('command', 'space')
                    self.speech_active = True
                    self.voice_search_time = now
            else:
                if self.speech_active:
                    pyautogui.hotkey('command', 'space')
                    self.speech_active = False
                    
        except Exception as e:
            print(f"Error in gesture detection: {str(e)}")
            traceback.print_exc()
