"""
Asynchronous Merge Module
Uses pandas merge_asof to align system events to the nearest video timestamp.
"""
import pandas as pd

def merge_async(video_df, events_df):
    """
    Perform an asynchronous merge: for each video timestamp, take the most recent
    event that occurred at or before that time (backward direction).
    
    Parameters:
        video_df (pd.DataFrame): sorted by timestamp, with regular intervals.
        events_df (pd.DataFrame): sorted by timestamp, irregularly spaced.
    
    Returns:
        pd.DataFrame: video rows with event columns appended.
    """
    # Ensure dataframes are sorted by timestamp
    video_sorted = video_df.sort_values('timestamp').reset_index(drop=True)
    events_sorted = events_df.sort_values('timestamp').reset_index(drop=True)
    
    # If events_df is empty, return video_df with empty event columns
    if events_sorted.empty:
        for col in ['event_type', 'event_count']:
            if col not in video_sorted.columns:
                video_sorted[col] = 0
        return video_sorted
    
    # Perform asof merge (backward)
    merged = pd.merge_asof(
        video_sorted,
        events_sorted,
        on='timestamp',
        direction='backward',
        allow_exact_matches=True
    )
    return merged

def merge_all(video_df, audio_df, events_df, labels_df=None):
    """
    Merge all data sources into a single DataFrame aligned on video timestamps.
    
    Steps:
        1. Merge video and audio on timestamp (both have same rate).
        2. Asynchronously merge system events.
        3. Optionally merge labels.
    """
    # Merge video and audio (outer join to keep all video rows, even if audio missing)
    df = pd.merge(video_df, audio_df, on='timestamp', how='outer')
    
    # If system events exist, merge asynchronously
    if events_df is not None and not events_df.empty:
        # First, pivot events to have tab_switches and copy_paste columns if needed
        # Our synthetic generator produces columns: event_type, event_count
        # We'll aggregate by timestamp: sum counts per event type
        if 'event_type' in events_df.columns and 'event_count' in events_df.columns:
            # Pivot to get separate columns for each event type
            pivot = events_df.pivot_table(
                index='timestamp',
                columns='event_type',
                values='event_count',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            # Rename columns to match expected names
            pivot.columns = ['timestamp'] + [f'{col}' for col in pivot.columns if col != 'timestamp']
            # Ensure we have tab_switches and copy_paste, even if absent
            for col in ['tab_switches', 'copy_paste']:
                if col not in pivot.columns:
                    pivot[col] = 0
            events_aggregated = pivot
        else:
            # Assume events_df already has tab_switches and copy_paste columns
            events_aggregated = events_df.copy()
        
        # Merge asynchronously
        df = merge_async(df, events_aggregated)
    else:
        # No events; add zeros
        df['tab_switches'] = 0
        df['copy_paste'] = 0
        df['event_count'] = 0
    
    # Merge labels if provided
    if labels_df is not None:
        df = pd.merge(df, labels_df, on='timestamp', how='left')
    
    return df