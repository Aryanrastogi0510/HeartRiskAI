"""Unit tests for model training specifications, evaluation metrics,
clinical specificity calculation, and artifact loading.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.evaluate_models import calculate_classification_metrics
from src.train_models import get_model_specifications
from src.config import MODELS_DIR


def test_calculate_classification_metrics_accuracy_and_specificity():
    """Verifies that accuracy, sensitivity, and specificity are correctly computed."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # 3 TN, 1 FP, 1 FN, 3 TP
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.8, 0.85, 0.9])

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        model_name="TestModel",
        experiment_name="TestExp",
        feature_count=5,
        training_time_sec=0.5,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["true_negatives"] == 3
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 3
    # Specificity = TN / (TN + FP) = 3 / 4 = 0.75
    assert metrics["specificity"] == 0.75
    # Sensitivity (Recall) = TP / (TP + FN) = 3 / 4 = 0.75
    assert metrics["recall_sensitivity"] == 0.75
    assert metrics["roc_auc"] > 0.8
    assert metrics["pr_auc"] > 0.7


def test_calculate_classification_metrics_handles_all_zeros_prediction():
    """Verifies zero-division protection when a model predicts all zeros."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["true_negatives"] == 2
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 2
    assert metrics["precision"] == 0.0
    assert metrics["recall_sensitivity"] == 0.0
    assert metrics["specificity"] == 1.0


def test_model_specifications_covers_all_four_models():
    """Verifies that all 4 required models are specified with balanced weighting."""
    specs = get_model_specifications(scale_pos_weight=5.5)

    required_models = ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
    for m in required_models:
        assert m in specs
        assert "estimator" in specs[m]
        assert "param_grid" in specs[m]
        assert len(specs[m]["param_grid"]) > 0

    # Check class weighting is configured
    assert specs["Logistic Regression"]["estimator"].class_weight == "balanced"
    assert specs["Random Forest"]["estimator"].class_weight == "balanced"
    assert specs["SVM"]["estimator"].class_weight == "balanced"
    assert specs["XGBoost"]["estimator"].scale_pos_weight == 5.5
