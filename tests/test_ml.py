import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.ml.train import get_features_target
from src.ml.evaluate import evaluate_model
from src.ml.threshold_optimizer import optimize_threshold

def test_evaluate():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier().fit(X, y)
    metrics = evaluate_model(model, X, y, threshold=0.5)
    assert 'precision' in metrics
    assert 'recall' in metrics