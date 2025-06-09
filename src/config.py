"""Central configuration values for Optik."""

import math

# Cursor control
SMOOTHER_WINDOW = 5            # for MovingAverageSmoother
USE_KALMAN = True              # Set True for Kalman filter
KALMAN_R = 0.03                # measurement noise
KALMAN_Q = 0.0001              # process noise

# Gesture thresholds
TOUCH_DIST_PX = 40             # max pixel distance between tips counted as "touch" (tuned at runtime)

# Dictation settings
DICTATION_TIMEOUT = 5          # seconds of silence to auto‑stop

# Performance
FPS = 30                       # camera FPS target 