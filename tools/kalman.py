import numpy as np


class KalmanTracker:
    def __init__(self, dt=1, p = 1, r=1, q=0.1):
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.P = np.eye(4) * p
        self.H = np.eye(4)[:2]
        self.I = np.eye(4)
        self.R = np.eye(2) * r
        self.Q = np.eye(4) * q
        self.state = np.zeros((4, 1))

        self.missing = 0
        self.old_state = []
        self.temperature = 3

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measure, kconf):
        measure = np.array(measure).reshape(2, 1)
        y = measure - self.H @ self.state
        K = self.P @ self.H.T @ np.linalg.pinv(self.H @ self.P @ self.H.T + self.R/kconf**self.temperature)
        self.state = self.state + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.state
