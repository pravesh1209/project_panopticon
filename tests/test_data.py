import pytest
import pandas as pd
from src.data.loader import load_video_telemetry
from src.data.cleaner import impute_video_data

def test_load_video():
    df = load_video_telemetry('data/synthetic/video_telemetry.csv')
    assert 'timestamp' in df.columns
    assert len(df) > 0

def test_impute_video():
    df = pd.DataFrame({'eye_gaze_angle': [1, np.nan, 3], 'head_pose_x': [0, np.nan, 2], 'head_pose_y': [0, np.nan, 1], 'face_detected': [1, np.nan, 1]})
    cleaned = impute_video_data(df)
    assert cleaned['eye_gaze_angle'].isna().sum() == 0
    assert cleaned['face_detected'].dtype == int