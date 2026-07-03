import numpy as np

class ObjectDetector:
    def __init__(self, classes=['phone', 'book', 'notebook', 'document', 'person']):
        self.classes = classes

    def detect_objects(self, frame):
        """
        Return list of detected objects with bounding boxes.
        Simulated: returns random objects sometimes.
        """
        # Simulate detection: 0 or 1 object
        detected = []
        if np.random.random() < 0.1:
            obj = np.random.choice(self.classes)
            detected.append({'class': obj, 'confidence': np.random.uniform(0.7, 0.99)})
        return detected