import pandas as pd
import numpy as np

class RiskEngine:
    """
    Rule-based risk scoring using derived features.
    """
    def __init__(self, config=None):
        self.rules = [
            {'feature': 'tab_switch_density', 'threshold': 1, 'weight': 15},
            {'feature': 'copy_paste_density', 'threshold': 1, 'weight': 25},
            {'feature': 'gaze_deviation', 'threshold': 15, 'duration': 5, 'weight': 25},  # sustained >5s
            {'feature': 'audio_db', 'threshold': 60, 'weight': 10},
            {'feature': 'face_detected', 'threshold': 0, 'duration': 3, 'weight': 20},  # missing face >3s
        ]
        self.risk_score_col = 'risk_score'

    def compute_risk(self, df):
        """
        Apply rules and compute total risk score per row.
        """
        df_risk = df.copy()
        total_risk = np.zeros(len(df_risk))
        for rule in self.rules:
            feat = rule['feature']
            if feat not in df_risk.columns:
                continue
            thresh = rule['threshold']
            weight = rule['weight']
            duration = rule.get('duration', 0)
            # For duration-based rules, we need rolling sum of violations
            if duration > 0:
                violation = (df_risk[feat] > thresh).astype(int)
                sustained = violation.rolling(window=duration, min_periods=1).sum() >= duration
                total_risk += sustained.astype(int) * weight
            else:
                total_risk += (df_risk[feat] > thresh).astype(int) * weight
        df_risk[self.risk_score_col] = total_risk
        # Categorize risk
        df_risk['risk_category'] = pd.cut(
            df_risk[self.risk_score_col],
            bins=[-1, 20, 50, 80, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        return df_risk

    def get_risk_timeline(self, df):
        """
        Return a summary of risk events.
        """
        risky = df[df[self.risk_score_col] > 20].copy()
        if risky.empty:
            return pd.DataFrame()
        risky['timestamp_diff'] = risky['timestamp'].diff().dt.total_seconds()
        # identify continuous risk periods
        risky['new_event'] = (risky['timestamp_diff'] > 5) | risky['timestamp_diff'].isna()
        risky['event_id'] = risky['new_event'].cumsum()
        timeline = risky.groupby('event_id').agg(
            start=('timestamp', 'min'),
            end=('timestamp', 'max'),
            max_risk=('risk_score', 'max'),
            duration=('timestamp', lambda x: (x.max() - x.min()).total_seconds())
        ).reset_index(drop=True)
        return timeline