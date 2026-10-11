"""Unit tests for Genetic Algorithm feature selection, chromosome operations,
mandatory feature retention, and convergence tracking.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.genetic_algorithm import (
    GeneticAlgorithmFeatureSelector,
    compare_ga_vs_full_features,
)
from src.config import MANDATORY_FEATURES


@pytest.fixture
def mock_ga_data():
    """Generates synthetic dataset for rapid GA unit testing."""
    np.random.seed(42)
    n = 120
    feature_names = [
        "age", "male", "sysBP", "diaBP", "totChol",
        "glucose", "cigsPerDay", "BMI", "heartRate", "diabetes"
    ]
    X = pd.DataFrame(
        np.random.randn(n, len(feature_names)),
        columns=feature_names,
    )
    # Synthetic target with strong correlation to age and sysBP
    logits = 0.8 * X["age"] + 0.6 * X["sysBP"] - 0.5 * X["male"]
    probs = 1 / (1 + np.exp(-logits))
    y = pd.Series((probs > 0.5).astype(int), name="TenYearCHD")
    return X, y, feature_names


def test_chromosome_encoding_decoding(mock_ga_data):
    """Verifies binary chromosome encoding and decoding to feature lists."""
    _, _, feature_names = mock_ga_data
    ga = GeneticAlgorithmFeatureSelector(
        feature_names=feature_names,
        mandatory_features=["age", "male"],
    )

    selected = ["age", "sysBP", "totChol"]
    chrom = ga.features_to_chromosome(selected)

    assert len(chrom) == len(feature_names)
    assert chrom[feature_names.index("age")] == 1
    assert chrom[feature_names.index("sysBP")] == 1
    assert chrom[feature_names.index("totChol")] == 1
    assert chrom[feature_names.index("diaBP")] == 0

    decoded = ga.chromosome_to_features(chrom)
    # Mandatory features must be present
    assert "age" in decoded
    assert "male" in decoded
    assert "sysBP" in decoded
    assert "totChol" in decoded


def test_mandatory_features_never_dropped(mock_ga_data):
    """Verifies that mandatory clinical features (age, sex) are retained even if chromosome bit is 0."""
    _, _, feature_names = mock_ga_data
    ga = GeneticAlgorithmFeatureSelector(
        feature_names=feature_names,
        mandatory_features=["age", "male"],
    )

    # All-zeros chromosome
    zero_chrom = np.zeros(len(feature_names), dtype=int)
    repaired = ga.repair_chromosome(zero_chrom)

    assert repaired[feature_names.index("age")] == 1
    assert repaired[feature_names.index("male")] == 1

    features = ga.chromosome_to_features(repaired)
    assert "age" in features
    assert "male" in features


def test_population_initialization(mock_ga_data):
    """Verifies population initialization constraints, sizes, and baselines."""
    _, _, feature_names = mock_ga_data
    pop_size = 20
    ga = GeneticAlgorithmFeatureSelector(
        feature_names=feature_names,
        mandatory_features=["age", "male"],
        pop_size=pop_size,
        random_state=42,
    )
    pop = ga.initialize_population()

    assert len(pop) == pop_size
    for ind in pop:
        assert len(ind) == len(feature_names)
        # All individuals must have mandatory bits set
        assert ind[feature_names.index("age")] == 1
        assert ind[feature_names.index("male")] == 1
        # No individual can have 0 features
        assert np.sum(ind) >= 2


def test_crossover_and_mutation(mock_ga_data):
    """Verifies crossover and mutation produce valid offspring."""
    _, _, feature_names = mock_ga_data
    ga = GeneticAlgorithmFeatureSelector(
        feature_names=feature_names,
        mandatory_features=["age", "male"],
        crossover_rate=1.0,
        mutation_rate=0.2,
        random_state=42,
    )
    p1 = np.ones(len(feature_names), dtype=int)
    p2 = np.zeros(len(feature_names), dtype=int)
    p2 = ga.repair_chromosome(p2)

    c1, c2 = ga.crossover(p1, p2)
    assert len(c1) == len(feature_names)
    assert len(c2) == len(feature_names)
    assert c1[feature_names.index("age")] == 1
    assert c2[feature_names.index("male")] == 1

    mutated = ga.mutate(c1)
    assert len(mutated) == len(feature_names)
    assert mutated[feature_names.index("age")] == 1
    assert mutated[feature_names.index("male")] == 1


def test_fitness_evaluation_and_parsimony_penalty(mock_ga_data):
    """Verifies fitness rewards ROC-AUC and penalizes unnecessarily large subsets."""
    X, y, feature_names = mock_ga_data
    ga = GeneticAlgorithmFeatureSelector(
        feature_names=feature_names,
        mandatory_features=["age", "male"],
        parsimony_penalty=0.05,
        cv_folds=3,
        random_state=42,
    )

    # 1. Evaluate with 3 features
    chrom_compact = np.zeros(len(feature_names), dtype=int)
    chrom_compact[feature_names.index("age")] = 1
    chrom_compact[feature_names.index("male")] = 1
    chrom_compact[feature_names.index("sysBP")] = 1

    fitness_compact, auc_compact = ga.evaluate_chromosome(chrom_compact, X, y)
    expected_penalty_compact = 0.05 * (3 / len(feature_names))
    assert abs(fitness_compact - (auc_compact - expected_penalty_compact)) < 1e-6

    # 2. Evaluate with all 10 features
    chrom_all = np.ones(len(feature_names), dtype=int)
    fitness_all, auc_all = ga.evaluate_chromosome(chrom_all, X, y)
    expected_penalty_all = 0.05 * (10 / len(feature_names))
    assert abs(fitness_all - (auc_all - expected_penalty_all)) < 1e-6


def test_ga_reproducibility(mock_ga_data):
    """Verifies that two GA runs with identical seeds yield identical optimization results."""
    X, y, feature_names = mock_ga_data
    ga1 = GeneticAlgorithmFeatureSelector(
        feature_names=feature_names,
        mandatory_features=["age", "male"],
        pop_size=10,
        n_generations=5,
        cv_folds=3,
        random_state=42,
    )
    ga1.fit(X, y, verbose=False)

    ga2 = GeneticAlgorithmFeatureSelector(
        feature_names=feature_names,
        mandatory_features=["age", "male"],
        pop_size=10,
        n_generations=5,
        cv_folds=3,
        random_state=42,
    )
    ga2.fit(X, y, verbose=False)

    assert ga1.best_features_ == ga2.best_features_
    assert ga1.best_fitness_ == ga2.best_fitness_
    assert ga1.best_auc_ == ga2.best_auc_


def test_compare_ga_vs_full_features(mock_ga_data):
    """Verifies comparative validation function computes metrics for both feature sets."""
    X, y, feature_names = mock_ga_data
    selected = ["age", "male", "sysBP"]

    comparison = compare_ga_vs_full_features(
        X, y, selected_features=selected, all_features=feature_names, cv_folds=3, random_state=42
    )

    assert "full_features" in comparison
    assert "ga_selected_features" in comparison
    assert comparison["full_features"]["feature_count"] == len(feature_names)
    assert comparison["ga_selected_features"]["feature_count"] == len(selected)
    assert "mean_cv_roc_auc" in comparison["full_features"]
    assert "auc_difference" in comparison
