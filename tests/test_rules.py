import pytest
import pandas as pd
from src.rules.risk_engine import RiskEngine

def test_risk_engine():
    df = pd.DataFrame({'tab_switch_density': [2, 0], 'copy_paste_density': [0, 3], 'gaze_deviation': [20, 5], 'audio_db': [50, 70], 'face_detected': [1, 0], 'timestamp': pd.to_datetime(['2021-01-01 00:00:00', '2021-01-01 00:00:01'])})
    engine = RiskEngine()
    df_risk = engine.compute_risk(df)
    assert 'risk_score' in df_risk.columns
    assert df_risk['risk_score'].iloc[0] > 0