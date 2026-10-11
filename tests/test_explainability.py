"""Unit tests for Model Explainability, SHAP feature attributions,
input schema alignment, and fallback handling.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from src.explainability import ClinicalModelExplainer
from src.config import MODELS_DIR


@pytest.fixture
def sample_explainer():
    """Initializes ClinicalModelExplainer using saved best_model.joblib."""
    return ClinicalModelExplainer()


def test_explainer_initialization(sample_explainer):
    """Verifies explainer loads model, features, and configures an explainer engine."""
    assert sample_explainer.model is not None
    assert len(sample_explainer.features) > 0
    assert sample_explainer.explainer_type in ("LinearExplainer", "TreeExplainer", "KernelExplainer")
    assert sample_explainer.background_sample is not None


def test_format_input_instance_alignment(sample_explainer):
    """Verifies that input instances are mapped to exact expected features in identical order."""
    patient_dict = {
        "age": 55.0,
        "sysBP": 140.0,
        "glucose": 95.0,
        "male": 1,
        "cigsPerDay": 10.0,
        "BPMeds": 0,
        "extra_unneeded_col": 999.0,  # Should be filtered out
    }
    df_aligned = sample_explainer.format_input_instance(patient_dict)

    assert list(df_aligned.columns) == sample_explainer.features
    assert "extra_unneeded_col" not in df_aligned.columns
    assert df_aligned.shape == (1, len(sample_explainer.features))


def test_format_input_instance_missing_features(sample_explainer):
    """Verifies that an exception is raised when required model features are missing."""
    incomplete_patient = {"age": 50.0}  # Missing other model features
    with pytest.raises(ValueError, match="missing required model features"):
        sample_explainer.format_input_instance(incomplete_patient)


def test_explain_instance_output_validity(sample_explainer):
    """Verifies structure and numerical validity of local SHAP explanation."""
    train_df = pd.read_csv("data/processed/train.csv")
    sample_inst = train_df[sample_explainer.features].iloc[0].to_dict()

    exp = sample_explainer.explain_instance(sample_inst)

    assert "predicted_risk_probability" in exp
    assert 0.0 <= exp["predicted_risk_probability"] <= 1.0
    assert "shap_attributions" in exp
    assert "causality_disclaimer" in exp

    # Every feature must have a corresponding numerical attribution
    for f in sample_explainer.features:
        assert f in exp["shap_attributions"]
        assert isinstance(exp["shap_attributions"][f], float)


def test_plot_generation(sample_explainer, tmp_path):
    """Verifies that waterfall and global importance plots render without errors."""
    train_df = pd.read_csv("data/processed/train.csv")
    sample_inst = train_df[sample_explainer.features].iloc[0].to_dict()

    waterfall_file = tmp_path / "test_waterfall.png"
    sample_explainer.plot_waterfall(sample_inst, output_path=waterfall_file)
    assert waterfall_file.exists()
    assert waterfall_file.stat().st_size > 0

    global_file = tmp_path / "test_global.png"
    sample_explainer.plot_global_importance(output_path=global_file)
    assert global_file.exists()
    assert global_file.stat().st_size > 0


def test_compare_with_expert_rules():
    """Verifies structured clinical contrast between SHAP attribution and expert rules."""
    mock_shap = {
        "predicted_risk_probability": 0.42,
        "risk_increasing_factors": ["sysBP (+0.12)", "age (+0.08)"],
        "risk_decreasing_factors": ["cigsPerDay (-0.05)"],
    }
    mock_rules = ["[Blood Pressure] Stage 2 Hypertension", "[Tobacco] Heavy Tobacco Burden"]

    comparison = ClinicalModelExplainer.compare_with_expert_rules(mock_shap, mock_rules)

    assert "statistical_ml_summary" in comparison
    assert "expert_system_summary" in comparison
    assert "synthesis_and_contrast" in comparison
    assert comparison["statistical_ml_summary"]["predicted_risk_probability"] == 0.42
    assert len(comparison["expert_system_summary"]["triggered_guideline_rules"]) == 2
