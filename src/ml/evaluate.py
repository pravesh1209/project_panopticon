"""
Model Evaluation Module
Computes classification metrics and confusion matrices at specified thresholds.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a single model at a given decision threshold.
    
    Returns a dictionary with metrics:
        - threshold, precision, recall, f1, roc_auc, pr_auc
        - tn, fp, fn, tp (confusion matrix components)
        - false_positives, false_negatives (counts)
    """
    # Get predicted probabilities for positive class
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For models without predict_proba (e.g., some SVM), use decision_function
        y_proba = model.decision_function(X_test)
        # Normalize to [0,1] if needed (not necessary for threshold)
        # We'll rely on the default behavior; but to be safe, we can use a sigmoid.
        # For simplicity, we assume predict_proba is available.

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Avoid division by zero
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'false_positives': fp,
        'false_negatives': fn
    }
    return metrics

def evaluate_all_models(trained_models, X_test, y_test, thresholds=None):
    """
    Evaluate all trained models across a list of thresholds.
    
    Parameters:
        trained_models (dict): Model name -> model object.
        X_test, y_test: Test data.
        thresholds (list): List of threshold values to evaluate.
    
    Returns:
        pd.DataFrame: Results for every (model, threshold) combination.
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    results = []
    for name, model in trained_models.items():
        print(f"Evaluating {name}...")
        for thresh in thresholds:
            metrics = evaluate_model(model, X_test, y_test, threshold=thresh)
            metrics['model'] = name
            results.append(metrics)
    return pd.DataFrame(results)

def print_confusion_matrix(model, X_test, y_test, threshold=0.5):
    """Helper to print confusion matrix for a model at a given threshold."""
    metrics = evaluate_model(model, X_test, y_test, threshold)
    print(f"Threshold: {threshold:.2f}")
    print(f"Confusion Matrix: [[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")
    print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")