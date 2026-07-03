"""
Data Cleaning Module
Handles missing values caused by dropped video frames and asynchronous gaps.
"""
import pandas as pd
import numpy as np

def impute_video_data(df):
    """
    Impute missing video sensor data:
        - Forward-fill eye_gaze_angle, head_pose_x, head_pose_y.
        - Fill any remaining NaNs with 0.
        - Fill face_detected with 0 and convert to int.
    """
    df_clean = df.copy()
    for col in ['eye_gaze_angle', 'head_pose_x', 'head_pose_y']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].ffill()
            # Fill any initial NaN with 0
            df_clean[col] = df_clean[col].fillna(0)
    if 'face_detected' in df_clean.columns:
        df_clean['face_detected'] = df_clean['face_detected'].fillna(0).astype(int)
    return df_clean

def impute_audio_data(df):
    """
    Interpolate missing audio_db values linearly over time.
    If all values are missing, fill with median of entire column.
    """
    df_clean = df.copy()
    if 'audio_db' in df_clean.columns:
        df_clean['audio_db'] = df_clean['audio_db'].interpolate(method='linear', limit_direction='both')
        if df_clean['audio_db'].isna().any():
            median = df_clean['audio_db'].median()
            df_clean['audio_db'] = df_clean['audio_db'].fillna(median)
    return df_clean

def impute_event_data(df):
    """
    Fill missing system event columns (tab_switches, copy_paste, event_count) with 0.
    """
    df_clean = df.copy()
    for col in ['tab_switches', 'copy_paste', 'event_count']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    return df_clean

def impute_labels(df):
    """
    Fill missing is_cheating labels with 0 (assuming no cheating when missing).
    """
    df_clean = df.copy()
    if 'is_cheating' in df_clean.columns:
        df_clean['is_cheating'] = df_clean['is_cheating'].fillna(0).astype(int)
    return df_clean

def clean_merged_data(merged_df):
    """
    Apply all imputation steps in the correct order.
    """
    df = impute_video_data(merged_df)
    df = impute_audio_data(df)
    df = impute_event_data(df)
    df = impute_labels(df)
    return df