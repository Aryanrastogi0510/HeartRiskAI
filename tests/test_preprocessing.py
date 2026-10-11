"""Unit tests for data preprocessing pipeline, leakage prevention,
class weighting, training SMOTE, and mandatory feature preservation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_audit import DataAuditor, DataAuditException
from src.data_preprocessing import (
    CardiovascularDataPreprocessor,
    DataPreprocessingException,
    stratified_split,
    compute_class_weights,
    apply_training_smote,
    enforce_mandatory_features,
)
from src.config import (
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    MANDATORY_FEATURES,
)


@pytest.fixture
def sample_cardio_df() -> pd.DataFrame:
    """Creates synthetic cardiovascular DataFrame matching Framingham schema."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "male": np.random.choice([0, 1], size=n, p=[0.55, 0.45]),
        "age": np.random.randint(32, 70, size=n),
        "education": np.random.choice([1.0, 2.0, 3.0, 4.0, np.nan], size=n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        "currentSmoker": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
        "cigsPerDay": np.random.choice([0.0, 10.0, 20.0, np.nan], size=n, p=[0.5, 0.25, 0.2, 0.05]),
        "BPMeds": np.random.choice([0.0, 1.0, np.nan], size=n, p=[0.9, 0.08, 0.02]),
        "prevalentStroke": np.random.choice([0, 1], size=n, p=[0.98, 0.02]),
        "prevalentHyp": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "diabetes": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "totChol": np.random.normal(235, 45, size=n),
        "sysBP": np.random.normal(132, 22, size=n),
        "diaBP": np.random.normal(82, 12, size=n),
        "BMI": np.random.normal(25.8, 4.0, size=n),
        "heartRate": np.random.normal(75, 12, size=n),
        "glucose": np.random.choice([80.0, 100.0, 140.0, np.nan], size=n, p=[0.7, 0.15, 0.05, 0.1]),
        "TenYearCHD": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    })
    # Add a few nulls in numerical columns
    df.loc[0, "totChol"] = np.nan
    df.loc[1, "BMI"] = np.nan
    return df


def test_stratified_split_preserves_class_ratio(sample_cardio_df):
    """Verifies stratified split isolates test set and maintains class proportions."""
    X_train, X_test, y_train, y_test = stratified_split(
        sample_cardio_df, target_col="TenYearCHD", test_size=0.20, random_state=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert "TenYearCHD" not in X_train.columns
    assert "TenYearCHD" not in X_test.columns

    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_preprocessor_prevents_target_leakage(sample_cardio_df):
    """Verifies that preprocessor strictly raises an exception if target is in feature matrix."""
    preprocessor = CardiovascularDataPreprocessor()
    with pytest.raises(DataPreprocessingException, match="Data Leakage Violation"):
        preprocessor.fit(sample_cardio_df)  # Contains TenYearCHD


def test_preprocessor_fits_on_train_only(sample_cardio_df):
    """Verifies that preprocessor statistics are derived strictly from training data."""
    X_train, X_test, y_train, y_test = stratified_split(
        sample_cardio_df, target_col="TenYearCHD", test_size=0.20, random_state=42
    )

    preprocessor = CardiovascularDataPreprocessor(scaler_type="standard")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Imputed sets must have zero null values
    assert X_train_trans.isnull().sum().sum() == 0
    assert X_test_trans.isnull().sum().sum() == 0

    # Training numerical features must have approximately zero mean under StandardScaler
    for num_col in NUMERICAL_FEATURES:
        assert abs(X_train_trans[num_col].mean()) < 1e-6
        assert abs(X_train_trans[num_col].std() - 1.0) < 0.1


def test_preprocessor_preserves_feature_names(sample_cardio_df):
    """Verifies that output DataFrame preserves full feature names in correct order."""
    X = sample_cardio_df.drop(columns=["TenYearCHD"])
    preprocessor = CardiovascularDataPreprocessor()
    X_trans = preprocessor.fit_transform(X)

    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    assert list(X_trans.columns) == expected_cols
    assert X_trans.shape == (100, 15)


def test_compute_class_weights(sample_cardio_df):
    """Verifies balanced class weights computation for imbalanced binary target."""
    y = sample_cardio_df["TenYearCHD"]
    weights_info = compute_class_weights(y)

    assert "class_weight_dict" in weights_info
    assert "scale_pos_weight" in weights_info

    w_dict = weights_info["class_weight_dict"]
    assert w_dict[1] > w_dict[0]  # Minority class must have higher weight
    assert weights_info["scale_pos_weight"] > 1.0


def test_apply_training_smote(sample_cardio_df):
    """Verifies SMOTE resamples training minority class without touching test set."""
    X_train, _, y_train, _ = stratified_split(
        sample_cardio_df, target_col="TenYearCHD", test_size=0.20, random_state=42
    )
    preprocessor = CardiovascularDataPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)

    initial_pos = int((y_train == 1).sum())
    initial_neg = int((y_train == 0).sum())

    X_res, y_res = apply_training_smote(X_train_trans, y_train, random_state=42)

    # Resampled dataset must have equal class counts under auto strategy
    assert (y_res == 1).sum() == initial_neg
    assert len(X_res) == initial_neg * 2
    assert list(X_res.columns) == list(X_train_trans.columns)


def test_enforce_mandatory_features():
    """Verifies that mandatory clinical features (age, sex) are never dropped."""
    selected = ["sysBP", "totChol", "glucose"]
    enforced = enforce_mandatory_features(selected)

    for m in MANDATORY_FEATURES:
        assert m in enforced
    # Check that original features remain present
    for s in selected:
        assert s in enforced


def test_preprocessor_serialization(sample_cardio_df, tmp_path):
    """Verifies that saving and loading preprocessor preserves exact transformation results."""
    X_train, X_test, _, _ = stratified_split(
        sample_cardio_df, target_col="TenYearCHD", test_size=0.20, random_state=42
    )
    preprocessor = CardiovascularDataPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)

    save_file = tmp_path / "test_preprocessor.joblib"
    preprocessor.save(save_file)
    assert save_file.exists()

    loaded_preprocessor = CardiovascularDataPreprocessor.load(save_file)
    X_test_trans_original = preprocessor.transform(X_test)
    X_test_trans_loaded = loaded_preprocessor.transform(X_test)

    pd.testing.assert_frame_equal(X_test_trans_original, X_test_trans_loaded)
