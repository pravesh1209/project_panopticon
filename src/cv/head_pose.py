import numpy as np

class HeadPoseEstimator:
    def estimate_pose(self, frame):
        # Dummy: head pose (yaw, pitch, roll)
        yaw = np.random.uniform(-15, 15)
        pitch = np.random.uniform(-10, 10)
        roll = np.random.uniform(-5, 5)
        return yaw, pitch, roll