"""Unit tests for the Clinical Expert System.
Verifies explicit IF-THEN rules, boundary conditions, input validation,
distinction between missing and negative data, and clinical combinations.
"""

import pytest
from src.expert_system import CardiovascularExpertSystem, SeverityLevel


@pytest.fixture
def expert():
    """Instantiates the CardiovascularExpertSystem."""
    return CardiovascularExpertSystem()


def test_hypertensive_crisis_rule(expert):
    """Verifies that SBP >= 180 or DBP >= 120 triggers critical hypertensive crisis alert."""
    patient = {"sysBP": 185.0, "diaBP": 95.0}
    res = expert.evaluate(patient)

    triggered_ids = [r["rule_id"] for r in res.triggered_rules]
    assert "RULE_BP_CRISIS" in triggered_ids
    assert res.has_critical_alerts is True
    assert "CRITICAL" in res.qualitative_risk_tier


def test_hypertension_stage2_and_stage1_boundaries(expert):
    """Verifies ACC/AHA stage 1 and stage 2 hypertension boundary conditions."""
    # Stage 2: SBP = 145, DBP = 85
    p_stage2 = {"sysBP": 145.0, "diaBP": 85.0}
    res2 = expert.evaluate(p_stage2)
    ids2 = [r["rule_id"] for r in res2.triggered_rules]
    assert "RULE_BP_STAGE2" in ids2
    assert "RULE_BP_STAGE1" not in ids2

    # Stage 1: SBP = 135, DBP = 85
    p_stage1 = {"sysBP": 135.0, "diaBP": 85.0}
    res1 = expert.evaluate(p_stage1)
    ids1 = [r["rule_id"] for r in res1.triggered_rules]
    assert "RULE_BP_STAGE1" in ids1
    assert "RULE_BP_STAGE2" not in ids1

    # Elevated: SBP = 125, DBP = 75
    p_elev = {"sysBP": 125.0, "diaBP": 75.0}
    res_elev = expert.evaluate(p_elev)
    ids_elev = [r["rule_id"] for r in res_elev.triggered_rules]
    assert "RULE_BP_ELEVATED" in ids_elev


def test_cholesterol_rules(expert):
    """Verifies ATP III total cholesterol thresholds."""
    # High: totChol = 250
    res_high = expert.evaluate({"totChol": 250.0})
    ids_high = [r["rule_id"] for r in res_high.triggered_rules]
    assert "RULE_CHOL_HIGH" in ids_high
    assert "RULE_CHOL_BORDERLINE" not in ids_high

    # Borderline: totChol = 220
    res_borderline = expert.evaluate({"totChol": 220.0})
    ids_border = [r["rule_id"] for r in res_borderline.triggered_rules]
    assert "RULE_CHOL_BORDERLINE" in ids_border
    assert "RULE_CHOL_HIGH" not in ids_border

    # Desirable: totChol = 180
    res_normal = expert.evaluate({"totChol": 180.0})
    ids_norm = [r["rule_id"] for r in res_normal.triggered_rules]
    assert "RULE_CHOL_HIGH" not in ids_norm
    assert "RULE_CHOL_BORDERLINE" not in ids_norm


def test_glucose_and_diabetes_rules(expert):
    """Verifies ADA glycemia thresholds and distinction from diagnosed diabetes."""
    # Diagnosed diabetes
    res_diab = expert.evaluate({"diabetes": 1, "glucose": 110.0})
    ids_diab = [r["rule_id"] for r in res_diab.triggered_rules]
    assert "RULE_DIABETES_DIAGNOSED" in ids_diab
    assert "RULE_GLUCOSE_IMPAIRED" not in ids_diab  # Suppressed when diabetes diagnosed

    # Fasting hyperglycemia without prior diagnosis
    res_hyper = expert.evaluate({"diabetes": 0, "glucose": 135.0})
    ids_hyper = [r["rule_id"] for r in res_hyper.triggered_rules]
    assert "RULE_GLUCOSE_DIABETIC_RANGE" in ids_hyper

    # Impaired fasting glucose (prediabetes)
    res_prediab = expert.evaluate({"diabetes": 0, "glucose": 112.0})
    ids_prediab = [r["rule_id"] for r in res_prediab.triggered_rules]
    assert "RULE_GLUCOSE_IMPAIRED" in ids_prediab


def test_tobacco_consumption_rules(expert):
    """Verifies CDC tobacco smoking severity tiers."""
    # Heavy: cigsPerDay = 25
    res_heavy = expert.evaluate({"currentSmoker": 1, "cigsPerDay": 25.0})
    ids_heavy = [r["rule_id"] for r in res_heavy.triggered_rules]
    assert "RULE_TOBACCO_HEAVY" in ids_heavy

    # Active moderate: cigsPerDay = 10
    res_mod = expert.evaluate({"currentSmoker": 1, "cigsPerDay": 10.0})
    ids_mod = [r["rule_id"] for r in res_mod.triggered_rules]
    assert "RULE_TOBACCO_ACTIVE" in ids_mod
    assert "RULE_TOBACCO_HEAVY" not in ids_mod

    # Non-smoker: cigsPerDay = 0
    res_none = expert.evaluate({"currentSmoker": 0, "cigsPerDay": 0.0})
    ids_none = [r["rule_id"] for r in res_none.triggered_rules]
    assert "RULE_TOBACCO_ACTIVE" not in ids_none
    assert "RULE_TOBACCO_HEAVY" not in ids_none


def test_missing_data_distinction(expert):
    """Verifies missing fields are reported as missing and NOT treated as negative findings."""
    incomplete_patient = {
        "age": 50,
        "male": 1,
        "sysBP": 135.0,
        "diaBP": 85.0,
        # totChol, glucose, and cigsPerDay are completely omitted
    }
    res = expert.evaluate(incomplete_patient)

    assert "totChol" in res.missing_fields
    assert "glucose" in res.missing_fields
    assert "cigsPerDay" in res.missing_fields

    # No cholesterol or diabetes rules should be falsely triggered
    triggered_ids = [r["rule_id"] for r in res.triggered_rules]
    assert "RULE_CHOL_HIGH" not in triggered_ids
    assert "RULE_CHOL_BORDERLINE" not in triggered_ids
    assert "RULE_GLUCOSE_DIABETIC_RANGE" not in triggered_ids

    # Missing notice must appear in educational guidance
    guidance_str = " ".join(res.educational_guidance)
    assert "Information Note" in guidance_str
    assert "totChol" in guidance_str


def test_input_validation_and_physiological_bounds(expert):
    """Verifies physiological plausibility validation filters out impossible values."""
    implausible_patient = {
        "sysBP": 450.0,  # Exceeds max plausible bound (300)
        "diaBP": 25.0,   # Below min plausible bound (40)
        "male": 5,       # Invalid binary value
    }
    res = expert.evaluate(implausible_patient)

    assert "sysBP" in res.invalid_fields
    assert "diaBP" in res.invalid_fields
    assert "male" in res.invalid_fields

    # Because sysBP 450 is rejected as invalid, it is replaced with None and doesn't trigger crisis
    triggered_ids = [r["rule_id"] for r in res.triggered_rules]
    assert "RULE_BP_CRISIS" not in triggered_ids


def test_combined_multimorbid_patient(expert):
    """Verifies complex multi-risk patient profile evaluation."""
    complex_patient = {
        "age": 62,
        "male": 1,
        "sysBP": 155.0,
        "diaBP": 94.0,
        "totChol": 265.0,
        "diabetes": 1,
        "currentSmoker": 1,
        "cigsPerDay": 20.0,
        "BPMeds": 1,
        "BMI": 32.5,
        "prevalentStroke": 1,
    }
    res = expert.evaluate(complex_patient)

    triggered_ids = [r["rule_id"] for r in res.triggered_rules]
    assert "RULE_BP_STAGE2" in triggered_ids
    assert "RULE_CHOL_HIGH" in triggered_ids
    assert "RULE_DIABETES_DIAGNOSED" in triggered_ids
    assert "RULE_TOBACCO_HEAVY" in triggered_ids
    assert "RULE_OBESITY_CLASS" in triggered_ids
    assert "RULE_HISTORY_STROKE" in triggered_ids
    assert res.has_critical_alerts is True


def test_optimal_healthy_patient(expert):
    """Verifies that an optimal profile with no risk factors triggers zero warning rules."""
    healthy_patient = {
        "age": 32,
        "male": 0,
        "sysBP": 112.0,
        "diaBP": 72.0,
        "totChol": 165.0,
        "glucose": 82.0,
        "diabetes": 0,
        "currentSmoker": 0,
        "cigsPerDay": 0.0,
        "BPMeds": 0,
        "BMI": 21.8,
        "heartRate": 68.0,
        "prevalentStroke": 0,
        "prevalentHyp": 0,
    }
    res = expert.evaluate(healthy_patient)

    assert len(res.triggered_rules) == 0
    assert len(res.identified_risk_factors) == 0
    assert "OPTIMAL_OR_UNASSESSED" in res.qualitative_risk_tier
    assert res.has_critical_alerts is False
