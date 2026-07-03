"""
Threshold Optimization Module
Finds an optimal decision threshold that meets a minimum precision requirement.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

def optimize_threshold(model, X_test, y_test, min_precision=0.90):
    """
    Find the highest threshold that yields precision >= min_precision.
    
    If no threshold meets the precision requirement, the threshold that gives
    the highest precision is selected (and its actual precision will be lower).
    
    Parameters:
        model: Trained classifier with predict_proba.
        X_test, y_test: Test data.
        min_precision (float): Minimum acceptable precision (e.g., 0.90).
    
    Returns:
        dict: {'threshold': float, 'precision': float, 'recall': float}
    """
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model does not support predict_proba.")

    # Compute precision, recall, and thresholds from precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # The last element of precisions and recalls correspond to threshold 1.0 (no positives)
    # We ignore the last point because it has no associated threshold.
    # The thresholds array is length n_thresholds, while precisions[:-1] and recalls[:-1] are length n_thresholds.
    # So we align: for each threshold i, precision = precisions[i], recall = recalls[i].

    # Find indices where precision >= min_precision
    valid = precisions[:-1] >= min_precision
    if not np.any(valid):
        # If no threshold meets the requirement, choose the one with highest precision
        best_idx = np.argmax(precisions[:-1])
        print(f"Warning: No threshold achieves precision >= {min_precision}. Choosing highest precision {precisions[best_idx]:.3f}.")
    else:
        # Among valid thresholds, choose the one with the highest recall
        valid_indices = np.where(valid)[0]
        # For ties in recall, the lower threshold (which gives higher recall) is already chosen
        best_idx = valid_indices[np.argmax(recalls[valid_indices])]

    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.99

    # Compute metrics at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    return {
        'threshold': optimal_threshold,
        'precision': precision,
        'recall': recall
    }

def optimize_threshold_custom(model, X_test, y_test, metric='precision', target_value=0.90, direction='max'):
    """
    More flexible threshold optimizer: maximize recall while maintaining metric >= target_value,
    or minimize false positives, etc. For now, we keep the simple version above.
    """
    # This function is a placeholder for future extension.
    pass