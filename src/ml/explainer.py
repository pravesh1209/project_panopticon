"""
SHAP Explainability Module
"""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def explain_model(model, X_train, X_test, sample_size=100):
    """
    Generate SHAP explanations for a trained model.
    Handles tree-based models (RandomForest, XGBoost, LightGBM, CatBoost) and linear models.
    """
    X_train_sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
    X_test_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    
    # Try TreeExplainer first (works for all tree ensembles)
    # If it fails, fall back to LinearExplainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.LinearExplainer(model, X_train_sample)
    
    shap_values = explainer.shap_values(X_test_sample)
    return explainer, shap_values, X_test_sample

def plot_shap_summary(shap_values, X_test_sample, feature_names=None, save_path=None):
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_shap_force(explainer, shap_values, X_test_sample, idx=0):
    shap.force_plot(explainer.expected_value, shap_values[idx], X_test_sample.iloc[idx], matplotlib=True)