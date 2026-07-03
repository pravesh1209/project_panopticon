"""
Synthetic Data Generator for Project Panopticon
-----------------------------------------------
Generates realistic exam proctoring telemetry with asynchronous events, missing values,
and ground truth cheating labels.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import random
from pathlib import Path

def generate_video_telemetry(timestamps, cheat_periods):
    """
    Generate eye gaze angle, head pose, and face presence.
    Returns DataFrame with columns: timestamp, eye_gaze_angle, head_pose_x, head_pose_y, face_detected
    """
    n = len(timestamps)
    # Normal behavior: gaze around center (0 degrees) with small noise
    # Cheating periods: gaze deviates (>15 degrees) for long durations
    gaze = np.random.normal(0, 5, n)  # degrees, centered at 0 (looking at screen)
    head_x = np.random.normal(0, 3, n)
    head_y = np.random.normal(0, 3, n)
    face_detected = np.ones(n, dtype=int)

    # Introduce missing face detection chunks (bad internet)
    missing_chunks = []
    for _ in range(random.randint(2, 5)):
        start = random.randint(0, n-50)
        duration = random.randint(20, 100)
        missing_chunks.append((start, min(start+duration, n)))
    for s, e in missing_chunks:
        face_detected[s:e] = 0
        gaze[s:e] = np.nan
        head_x[s:e] = np.nan
        head_y[s:e] = np.nan

    # Cheating periods: increase gaze deviation and head movement
    for (start_idx, end_idx) in cheat_periods:
        gaze[start_idx:end_idx] = np.random.choice([-20, 20], size=end_idx-start_idx) + np.random.normal(0, 5, end_idx-start_idx)
        head_x[start_idx:end_idx] = np.random.normal(10, 5, end_idx-start_idx)
        head_y[start_idx:end_idx] = np.random.normal(10, 5, end_idx-start_idx)
        # face might still be present, but could be missing sometimes

    df = pd.DataFrame({
        'timestamp': timestamps,
        'eye_gaze_angle': gaze,
        'head_pose_x': head_x,
        'head_pose_y': head_y,
        'face_detected': face_detected
    })
    return df

def generate_audio_telemetry(timestamps, cheat_periods):
    """
    Audio dB level: normal quiet (40-50 dB), cheating episodes may have multiple voices or noise.
    """
    n = len(timestamps)
    audio = np.random.uniform(40, 50, n)  # dB
    for (s, e) in cheat_periods:
        # Increase volume, maybe multiple voices
        audio[s:e] = np.random.uniform(55, 70, e-s)
        # also add some random spikes
    # Missing audio chunks (interpolation later)
    # Simulate dropouts: same as face missing for simplicity
    # Add noise spikes (sneezes)
    for _ in range(random.randint(1, 10)):
        idx = random.randint(0, n-1)
        audio[idx] = np.random.uniform(70, 90)
    df = pd.DataFrame({
        'timestamp': timestamps,
        'audio_db': audio
    })
    return df

def generate_system_events(timestamps, cheat_periods):
    """
    System events: tab switches and copy/paste events.
    These occur asynchronously, only at certain timestamps.
    We'll create a list of dicts with timestamp and event type.
    """
    events = []
    # Normal student may occasionally switch tabs innocently
    for t in timestamps:
        if random.random() < 0.01:  # 1% chance per second
            events.append({'timestamp': t, 'event_type': 'tab_switch', 'event_count': 1})
        if random.random() < 0.005:
            events.append({'timestamp': t, 'event_type': 'copy_paste', 'event_count': 1})

    # During cheating periods, increase event frequency
    for (s_idx, e_idx) in cheat_periods:
        for i in range(s_idx, e_idx):
            t = timestamps[i]
            if random.random() < 0.10:  # 10% chance per second
                events.append({'timestamp': t, 'event_type': 'tab_switch', 'event_count': random.randint(1, 3)})
            if random.random() < 0.08:
                events.append({'timestamp': t, 'event_type': 'copy_paste', 'event_count': random.randint(1, 2)})

    # Sort by timestamp
    events.sort(key=lambda x: x['timestamp'])
    df = pd.DataFrame(events)
    return df

def generate_labels(timestamps, cheat_periods):
    """
    Binary labels: 1 if cheating, else 0.
    """
    labels = np.zeros(len(timestamps), dtype=int)
    for (s, e) in cheat_periods:
        labels[s:e] = 1
    return pd.DataFrame({'timestamp': timestamps, 'is_cheating': labels})

def generate_synthetic_data(duration_seconds=600, output_dir='data/synthetic/', random_seed=42):
    """
    Generate all synthetic CSV files and save to output_dir.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Generate timestamps at 1 Hz
    start_time = datetime.now().replace(microsecond=0) - timedelta(seconds=duration_seconds)
    timestamps = [start_time + timedelta(seconds=i) for i in range(duration_seconds)]

    # Define cheating periods: about 5% of total time, in chunks of 10-30 seconds
    total_cheat_seconds = int(0.05 * duration_seconds)
    cheat_periods = []
    remaining = total_cheat_seconds
    while remaining > 0:
        chunk = min(random.randint(10, 30), remaining)
        start_idx = random.randint(0, duration_seconds - chunk)
        # Ensure no overlap with existing cheat periods
        overlap = False
        for (s, e) in cheat_periods:
            if not (start_idx + chunk <= s or start_idx >= e):
                overlap = True
                break
        if not overlap:
            cheat_periods.append((start_idx, start_idx + chunk))
            remaining -= chunk
        else:
            # try again with different start
            continue
        if remaining <= 0:
            break

    # Generate data
    video_df = generate_video_telemetry(timestamps, cheat_periods)
    audio_df = generate_audio_telemetry(timestamps, cheat_periods)
    events_df = generate_system_events(timestamps, cheat_periods)
    labels_df = generate_labels(timestamps, cheat_periods)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save to CSV
    video_df.to_csv(os.path.join(output_dir, 'video_telemetry.csv'), index=False)
    audio_df.to_csv(os.path.join(output_dir, 'audio_telemetry.csv'), index=False)
    events_df.to_csv(os.path.join(output_dir, 'system_events.csv'), index=False)
    labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)

    print(f"✅ Synthetic data generated in {output_dir}")
    print(f"   - video_telemetry.csv: {len(video_df)} rows")
    print(f"   - audio_telemetry.csv: {len(audio_df)} rows")
    print(f"   - system_events.csv: {len(events_df)} rows")
    print(f"   - labels.csv: {len(labels_df)} rows")
    print(f"   - Cheating periods: {len(cheat_periods)} (total {total_cheat_seconds} seconds)")

    return {
        'video_df': video_df,
        'audio_df': audio_df,
        'events_df': events_df,
        'labels_df': labels_df
    }

if __name__ == "__main__":
    # If run as script, generate 10 minutes of data
    generate_synthetic_data(duration_seconds=600, output_dir='data/synthetic/')