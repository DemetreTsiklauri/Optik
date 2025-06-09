"""Map hand landmarks to OS mouse / gesture actions."""
import math
import time
import threading
from typing import List, Tuple
import pyautogui

# Disable pyautogui delays
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Landmarks indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
MIDDLE_KNUCKLE = 14  # Changed to middle knuckle for cursor
RING_TIP = 16
PINKY_TIP = 20

class GestureController:
    def __init__(self):
        self.click_state = False
        self.enter_state = False
        self.scroll_state = False
        self.drag_state = False
        self.dictation_state = False
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Drag control
        self.pinch_start_time = 0
        self.pinch_threshold = 0.3  # seconds to hold pinch before drag starts
        
        # Scroll control
        self.last_scroll_time = time.time()
        self.scroll_speed = 0.005  # Faster updates for smoother scrolling
        
        # Cursor smoothing
        self.last_x = None
        self.last_y = None
        self.alpha = 0.65  # Smoothing factor (0.5-0.7 is sweet spot)

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
            cursor_pt = self._get_landmark(landmarks, MIDDLE_KNUCKLE)
            smooth_x, smooth_y = self._smooth_position(*cursor_pt)
            
            # Get all needed landmark points
            thumb_tip = self._get_landmark(landmarks, THUMB_TIP)
            index_tip = self._get_landmark(landmarks, INDEX_TIP)
            middle_tip = self._get_landmark(landmarks, MIDDLE_TIP)
            ring_tip = self._get_landmark(landmarks, RING_TIP)
            pinky_tip = self._get_landmark(landmarks, PINKY_TIP)
            
            # Get PIP joints for extension check
            index_pip = self._get_landmark(landmarks, INDEX_TIP - 2)
            middle_pip = self._get_landmark(landmarks, MIDDLE_TIP - 2)
            ring_pip = self._get_landmark(landmarks, RING_TIP - 2)
            pinky_pip = self._get_landmark(landmarks, PINKY_TIP - 2)
            thumb_pip = self._get_landmark(landmarks, THUMB_TIP - 2)

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
                    from .dictation import SpeechDictation
                    if not SpeechDictation.active():
                        threading.Thread(target=SpeechDictation.start, daemon=True).start()
                except Exception as e:
                    print(f"Dictation error: {str(e)}")
                self.dictation_state = False

        except Exception as e:
            print(f"Error in gesture processing: {str(e)}")
            return self._to_screen_coords(0.5, 0.5)

        return self._to_screen_coords(smooth_x, smooth_y) 