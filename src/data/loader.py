"""
Data Loader Module
Loads CSV files into pandas DataFrames with proper datetime parsing.
"""
import pandas as pd
from pathlib import Path

def load_video_telemetry(filepath):
    """
    Load video telemetry CSV.
    Columns: timestamp, eye_gaze_angle, head_pose_x, head_pose_y, face_detected
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Video telemetry file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df

def load_audio_telemetry(filepath):
    """
    Load audio telemetry CSV.
    Columns: timestamp, audio_db
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Audio telemetry file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df

def load_system_events(filepath):
    """
    Load system events CSV.
    Columns: timestamp, event_type, event_count (or tab_switches, copy_paste)
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"System events file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df

def load_labels(filepath):
    """
    Load ground truth labels CSV.
    Columns: timestamp, is_cheating
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Labels file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df