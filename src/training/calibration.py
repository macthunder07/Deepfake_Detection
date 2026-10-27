"""
Threshold Calibration using Youden's J Statistic on Validation Set Probabilities.
Location: D:\\Deepfake Detection\\src\\training\\calibration.py
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def calibrate_threshold_youden(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Finds the decision threshold that maximizes Youden's J statistic
    (Sensitivity + Specificity - 1) on validation probabilities.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])
    return best_threshold


def apply_calibrated_decision(y_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Applies a calibrated probability threshold to produce binary predictions (0 or 1)."""
    return (y_probs >= threshold).astype(int)