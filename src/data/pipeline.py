"""
Complete Data Pipeline
Orchestrates loading, merging, cleaning, and feature engineering.
"""
from .loader import load_video_telemetry, load_audio_telemetry, load_system_events, load_labels
from .merger import merge_all
from .cleaner import clean_merged_data
from .features import create_all_features

def build_full_pipeline(video_path, audio_path, events_path, labels_path=None):
    """
    Execute the entire preprocessing pipeline from raw CSVs to fully engineered DataFrame.
    
    Parameters:
        video_path (str): path to video_telemetry.csv
        audio_path (str): path to audio_telemetry.csv
        events_path (str): path to system_events.csv
        labels_path (str, optional): path to labels.csv
    
    Returns:
        pd.DataFrame: cleaned and feature-engineered data ready for modeling.
    """
    # Load raw data
    video_df = load_video_telemetry(video_path)
    audio_df = load_audio_telemetry(audio_path)
    events_df = load_system_events(events_path)
    labels_df = load_labels(labels_path) if labels_path else None
    
    # Merge asynchronously
    merged = merge_all(video_df, audio_df, events_df, labels_df)
    
    # Clean missing values
    cleaned = clean_merged_data(merged)
    
    # Engineer features
    featured = create_all_features(cleaned)
    
    return featured