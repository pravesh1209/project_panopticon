import numpy as np

class FaceTracker:
    def __init__(self):
        self.face_present = True

    def track_face(self, frame):
        """
        Return face presence (bool) and bounding box.
        Simulated: face is present 95% of the time.
        """
        present = np.random.random() < 0.95
        if present:
            bbox = (100, 100, 200, 200)  # x,y,w,h
        else:
            bbox = None
        return present, bbox