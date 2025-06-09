"""Cursor smoothing filters."""
from collections import deque
from typing import Tuple, Deque
import numpy as np
from filterpy.kalman import KalmanFilter

class ExpSmoother:
    def __init__(self, alpha=0.65):  # 0.5-0.7 is sweet spot
        self.alpha = alpha
        self.x = self.y = None

    def __call__(self, x, y):
        if self.x is None:
            self.x, self.y = x, y
        else:
            self.x = self.alpha * x + (1-self.alpha) * self.x
            self.y = self.alpha * y + (1-self.alpha) * self.y
        return self.x, self.y

class EMASmoother:
    """Exponential Moving Average smoother for responsive cursor movement."""
    
    def __init__(self, alpha: float = 0.7):  # Higher alpha = more responsive
        self.alpha = alpha
        self.x = None
        self.y = None
        
    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        if self.x is None or self.y is None:
            self.x = x
            self.y = y
        else:
            self.x = self.alpha * x + (1 - self.alpha) * self.x
            self.y = self.alpha * y + (1 - self.alpha) * self.y
        return self.x, self.y

class MovingAverageSmoother:
    """Simple moving‑average smoother."""

    def __init__(self, window: int = 5):
        self.buf: Deque[Tuple[float, float]] = deque(maxlen=window)

    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        self.buf.append((x, y))
        avg_x = sum(p[0] for p in self.buf) / len(self.buf)
        avg_y = sum(p[1] for p in self.buf) / len(self.buf)
        return avg_x, avg_y

class KalmanSmoother:
    """2‑D Kalman filter smoother."""

    def __init__(self):
        self.kf = KalmanFilter(
            dim_x=4,  # State: [x, y, dx, dy]
            dim_z=2   # Measurement: [x, y]
        )
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0.95, 0],  # Added velocity damping
            [0, 0, 0, 0.95]   # Added velocity damping
        ])  # State transition matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])  # Measurement matrix
        self.kf.R *= 0.05  # Reduced measurement noise for more smoothing
        self.kf.Q *= 0.01  # Reduced process noise for more smoothing
        self.kf.P *= 1000  # Initial covariance

    def __call__(self, x: float, y: float):
        """Update the filter with new measurements."""
        self.kf.predict()
        self.kf.update(np.array([x, y]))
        return self.kf.x[0], self.kf.x[1]  # Return filtered x, y position 