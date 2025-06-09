#!/usr/bin/env python3
import pyautogui, math, time, threading
import numpy as np
from filterpy.kalman_filter import KalmanFilter

# ── tuning ────────────────────────────────────────────
PINCH_THRESH = 0.05
DRAG_DELAY   = 0.25  # 0.25s hold for drag
SCROLL_SPEED = 3.0
BOX_GROW     = 0.25
KAL_Q, KAL_R = 0.015, 3.5  # Adjusted for smoother movement (lower Q, higher R)
BASE_SENSITIVITY = 1.2  # Increased base sensitivity
EDGE_SENSITIVITY = 1.8  # Increased edge sensitivity
SCROLL_SMOOTH = 0.1
MIN_MOVEMENT = 0.2  # Reduced for better responsiveness
BOTTOM_BOOST_THRESHOLD = 0.6  # Start boosting earlier
BOTTOM_BOOST_MULTIPLIER = 2.0  # Stronger boost for bottom
BOX_RESET_RATE = 0.1  # Rate at which box recenters

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_finger_up(lm, finger_tip, finger_pip):
    """Check if a finger is extended (tip above PIP joint)"""
    return lm[finger_tip].y < lm[finger_pip].y

def is_finger_down(lm, finger_tip, finger_pip):
    """Check if a finger is curled (tip below PIP joint)"""
    return lm[finger_tip].y > lm[finger_pip].y

class HandController:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.sw, self.sh = pyautogui.size()
        # Start with wider range
        self.minx, self.maxx = 0.0, 1.0
        self.miny, self.maxy = 0.0, 1.0
        self.target_minx, self.target_maxx = 0.0, 1.0
        self.target_miny, self.target_maxy = 0.0, 1.0

        # Kalman filter setup for optimal smoothness
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        self.kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        self.kf.P *= 1000
        self.kf.R = np.eye(2) * KAL_R
        self.kf.Q = np.eye(4) * KAL_Q
        self.kf.x = np.array([0,0,0,0], float)

        # pinch/drag state
        self.pinch_t0    = None
        self.drag        = False
        self.anchor_hand = None
        self.anchor_cur  = None
        self.last_pos    = None

        # scroll state
        self.scroll_mode = None
        self.scroll_amount = 0
        self.target_scroll = 0
        self.alive       = True
        threading.Thread(target=self._scroll_loop, daemon=True).start()

        # gesture state
        self.last_fist_time = 0
        self.fist_cooldown = 0.5

    def close(self):
        self.alive = False

    def update(self, lm):
        if not lm or len(lm) < 21:
            return
        kn = lm[5]  # Index finger MCP (knuckle)
        self._grow_box(kn)
        self._detect_gestures(lm)

        if self.scroll_mode:
            self._reset_pinch()
        else:
            self._handle_pinch(lm)

        if not self.drag:
            self._move_filtered(kn.x, kn.y)
        else:
            self._update_drag_raw(lm)

    def _grow_box(self, pt):
        """Dynamically expand the tracking box based on hand position"""
        # Update target bounds
        if pt.x < self.target_minx: self.target_minx = pt.x
        if pt.x > self.target_maxx: self.target_maxx = pt.x
        if pt.y < self.target_miny: self.target_miny = pt.y
        if pt.y > self.target_maxy: self.target_maxy = pt.y

        # Smoothly interpolate current bounds to target bounds
        self.minx += (self.target_minx - self.minx) * BOX_RESET_RATE
        self.maxx += (self.target_maxx - self.maxx) * BOX_RESET_RATE
        self.miny += (self.target_miny - self.miny) * BOX_RESET_RATE
        self.maxy += (self.target_maxy - self.maxy) * BOX_RESET_RATE

        # Ensure minimum size
        min_size = 0.2
        if self.maxx - self.minx < min_size:
            center = (self.maxx + self.minx) / 2
            self.minx = center - min_size/2
            self.maxx = center + min_size/2
        if self.maxy - self.miny < min_size:
            center = (self.maxy + self.miny) / 2
            self.miny = center - min_size/2
            self.maxy = center + min_size/2

        # Clamp to screen bounds
        self.minx, self.maxx = max(0, self.minx), min(1, self.maxx)
        self.miny, self.maxy = max(0, self.miny), min(1, self.maxy)

    def _get_dynamic_sensitivity(self, x, y):
        """Calculate sensitivity based on position (higher at edges)"""
        # Calculate distance from center (0.5, 0.5)
        dx = abs(x - 0.5)
        dy = abs(y - 0.5)
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Smoothly interpolate between base and edge sensitivity
        # Add extra boost at edges for better control
        edge_factor = min(1.0, dist * 2.5)  # Increased edge factor
        return BASE_SENSITIVITY + (EDGE_SENSITIVITY - BASE_SENSITIVITY) * edge_factor

    def _nonlinear_map(self, x, y):
        """
        Apply nonlinear mapping to improve control and reach.
        Makes it easier to reach screen edges while maintaining precision in center.
        """
        # Center the coordinates
        nx = x - 0.5
        ny = y - 0.5
        
        # Apply different nonlinear transformations for x and y
        # Use stronger power for y-axis to improve bottom reach
        nx = math.copysign(abs(nx) ** 1.2, nx)
        ny = math.copysign(abs(ny) ** 1.5, ny)  # Increased power for y-axis
        
        # Get dynamic sensitivity
        sensitivity = self._get_dynamic_sensitivity(x, y)
        
        # Apply sensitivity and recenter
        nx = (nx * sensitivity) + 0.5
        ny = (ny * sensitivity) + 0.5
        
        # Add extra boost for bottom area
        if y > BOTTOM_BOOST_THRESHOLD:  # If hand is in bottom third
            ny = 0.5 + (ny - 0.5) * BOTTOM_BOOST_MULTIPLIER  # Stronger boost for bottom movement
        
        # Clamp to screen bounds with slight overshoot for better edge behavior
        nx = max(-0.05, min(1.05, nx))
        ny = max(-0.05, min(1.05, ny))
        
        # Map to screen coordinates
        px = nx * self.sw
        py = ny * self.sh
        return px, py

    def _move_filtered(self, x, y):
        mx, my = self._nonlinear_map(x, y)
        
        # If we have a last position, check if movement is significant
        if self.last_pos is not None:
            dx = mx - self.last_pos[0]
            dy = my - self.last_pos[1]
            # If movement is very small, maintain last position
            if abs(dx) < MIN_MOVEMENT and abs(dy) < MIN_MOVEMENT:
                return
        
        # Update Kalman filter
        self.kf.predict()
        self.kf.update(np.array([mx, my]))
        
        # Get smoothed position
        cx, cy = self.kf.x[:2]
        pyautogui.moveTo(float(cx), float(cy), _pause=False)
        self.last_pos = (cx, cy)

    def _handle_pinch(self, lm):
        """Handle pinch gesture for clicking and dragging"""
        touching = dist(lm[4], lm[8]) < PINCH_THRESH
        now = time.time()
        
        if touching and self.pinch_t0 is None:
            self.pinch_t0 = now
            self.anchor_hand = self._nonlinear_map(lm[5].x, lm[5].y)
            self.anchor_cur = pyautogui.position()
            
        if touching and not self.drag and now - self.pinch_t0 >= DRAG_DELAY:
            pyautogui.mouseDown()
            self.drag = True
            
        if not touching:
            if self.drag:
                pyautogui.mouseUp()
                self.drag = False
            elif self.pinch_t0 and now - self.pinch_t0 < DRAG_DELAY:
                pyautogui.click()
            self.pinch_t0 = None

    def _update_drag_raw(self, lm):
        """Handle dragging with improved edge behavior"""
        hx, hy = self._nonlinear_map(lm[5].x, lm[5].y)
        dx, dy = hx - self.anchor_hand[0], hy - self.anchor_hand[1]
        
        # Calculate new position
        nx, ny = self.anchor_cur[0] + dx, self.anchor_cur[1] + dy
        
        # Add stability during drag
        if self.last_pos is not None:
            # If movement is very small, maintain last position
            if abs(nx - self.last_pos[0]) < MIN_MOVEMENT and abs(ny - self.last_pos[1]) < MIN_MOVEMENT:
                return
        
        # Apply Kalman filtering to drag movement
        self.kf.predict()
        self.kf.update(np.array([nx, ny]))
        nx, ny = self.kf.x[:2]
        
        # Ensure we can still move in both axes even at screen edges
        # by allowing slight overshoot that gets clamped by pyautogui
        nx = max(-10, min(self.sw + 10, nx))
        ny = max(-10, min(self.sh + 10, ny))
        
        # Use dragTo for original drag behavior
        pyautogui.dragTo(nx, ny, duration=0, button='left', _pause=False)
        
        self.last_pos = (nx, ny)

    def _reset_pinch(self):
        if self.drag:
            pyautogui.mouseUp(); self.drag = False
        self.pinch_t0 = None

    def _detect_gestures(self, lm):
        """Detect scroll gestures and reset drag if needed"""
        # Check for scroll gestures
        index_up = is_finger_up(lm, 8, 6)
        middle_up = is_finger_up(lm, 12, 10)
        ring_up = is_finger_up(lm, 16, 14)
        pinky_up = is_finger_up(lm, 20, 18)

        # Two fingers up = scroll down
        if index_up and middle_up and not ring_up and not pinky_up:
            self.scroll_mode = "down"
            self.target_scroll = -SCROLL_SPEED
            self._reset_pinch()  # Reset drag when scroll starts
        # Three fingers up = scroll up
        elif index_up and middle_up and ring_up and not pinky_up:
            self.scroll_mode = "up"
            self.target_scroll = SCROLL_SPEED
            self._reset_pinch()  # Reset drag when scroll starts
        else:
            self.scroll_mode = None
            self.target_scroll = 0

    def _scroll_loop(self):
        dt = 1.0 / 60  # 60 FPS for smooth scrolling
        while self.alive:
            if self.scroll_mode:
                # Smoothly interpolate current scroll amount to target
                self.scroll_amount += (self.target_scroll - self.scroll_amount) * SCROLL_SMOOTH
                pyautogui.scroll(int(self.scroll_amount))
            time.sleep(dt)

