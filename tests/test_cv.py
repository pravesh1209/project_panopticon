import pytest
from src.cv.gaze_estimator import GazeEstimator

def test_gaze():
    ge = GazeEstimator()
    gaze_x, gaze_y = ge.estimate_gaze(None)
    assert -30 <= gaze_x <= 30
    assert -20 <= gaze_y <= 20