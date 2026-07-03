import numpy as np
import pandas as pd

class GazeEstimator:
    def __init__(self, frame_width=640, frame_height=480):
        self.width = frame_width
        self.height = frame_height

    def estimate_gaze(self, frame):
        """
        Simulate gaze estimation: return horizontal and vertical angles.
        In production, use MediaPipe or OpenCV face mesh.
        """
        # Dummy: random gaze angles between -30 and 30 degrees
        gaze_x = np.random.uniform(-30, 30)
        gaze_y = np.random.uniform(-20, 20)
        return gaze_x, gaze_y