"""
test_hybrid.py — Tests for hybrid.py: Phase 1 (DT) + Phase 2 (CSP) combined.

Run from the dt-csp-fairness-tool/ directory:
    python -m pytest tests/test_hybrid.py -v

Covers:
  - run_hybrid: return structure, IDI counting, budget edge cases,
                phase splitting, DT rule extraction, model behaviour extremes,
                multi-value sensitive features, structured discrimination regions
"""

import numpy as np
import pandas as pd

from src.hybrid import run_hybrid


# ------------------------------------------------------------------ #
# Mock model helpers
# ------------------------------------------------------------------ #

class MockModel:
    """Minimal stand-in for a Keras model."""
    def __init__(self, predict_fn):
        self._fn = predict_fn

    def predict(self, x, verbose=0):
        val = self._fn(x[0])
        return np.array([[val]])


def always_discriminatory_model():
    """Binary sensitive feature at col 0: age=1→0.9, age=0→0.1. Always IDI."""
    return MockModel(lambda row: 0.9 if row[0] >= 1 else 0.1)


def never_discriminatory_model():
    """Constant 0.5: no pair ever exceeds the 0.05 threshold."""
    return MockModel(lambda _: 0.5)


def feature1_structured_model():
    """
    Discriminatory only when feature1 (col 1) > 0.5.
    age=1 and feature1>0.5 → 0.9, else → 0.1.
    The hybrid's Phase 1 DT should learn this split and guide Phase 2.
    """
    return MockModel(lambda row: 0.9 if (row[0] >= 1 and row[1] > 0.5) else 0.1)


def multivalue_sensitive_model():
    """race (col 0): predicts 0.9 if race==0 else 0.1. Always IDI when flipped from 0."""
    return MockModel(lambda row: 0.9 if row[0] == 0 else 0.1)


# ------------------------------------------------------------------ #
# Dataset factories
# ------------------------------------------------------------------ #

def binary_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":      rng.integers(0, 2, size=n).astype(float),
        "feature1": rng.uniform(0, 1, size=n),
    })


def multivalue_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "race":     rng.integers(0, 5, size=n).astype(float),
        "feature1": rng.uniform(0, 1, size=n),
    })


binary_config    = {"sensitive": ["age"]}
multivalue_config = {"sensitive": ["race"]}


# ================================================================== #
# run_hybrid — return structure
# ================================================================== #

class TestRunHybridStructure:

    def test_returns_correct_keys(self):
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=never_discriminatory_model(), X=X)
        assert set(result.keys()) == {"idi_ratio", "idi_count", "inputs_generated", "dt_rules"}

    def test_dt_rules_is_list(self):
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=always_discriminatory_model(), X=X)
        assert isinstance(result["dt_rules"], list)

    def test_idi_count_is_int(self):
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=never_discriminatory_model(), X=X)
        assert isinstance(result["idi_count"], int)

    def test_inputs_generated_is_int(self):
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=never_discriminatory_model(), X=X)
        assert isinstance(result["inputs_generated"], int)

    def test_idi_ratio_is_float(self):
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=never_discriminatory_model(), X=X)
        assert isinstance(result["idi_ratio"], float)


# ================================================================== #
# run_hybrid — budget and counts
# ================================================================== #

class TestRunHybridCounts:

    def test_budget_zero_returns_zeros(self):
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=0, model=always_discriminatory_model(), X=X)
        assert result["idi_ratio"] == 0.0
        assert result["idi_count"] == 0
        assert result["inputs_generated"] == 0
        assert result["dt_rules"] == []

    def test_inputs_generated_at_most_budget(self):
        """
        inputs_generated <= budget (Phase 2 CSP may terminate early if
        the solver exhausts the feasible space).
        """
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=never_discriminatory_model(), X=X)
        assert result["inputs_generated"] <= 20

    def test_inputs_generated_at_least_seed_n(self):
        """
        Phase 1 always generates max(1, int(seed_ratio * budget)) inputs.
        Even if Phase 2 produces zero (budget=1, remaining=0), inputs_generated >= 1.
        """
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=1, model=always_discriminatory_model(), X=X)
        assert result["inputs_generated"] >= 1

    def test_idi_ratio_consistency(self):
        """idi_ratio must equal idi_count / inputs_generated."""
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=always_discriminatory_model(), X=X)
        if result["inputs_generated"] > 0:
            expected = result["idi_count"] / result["inputs_generated"]
            assert abs(result["idi_ratio"] - expected) < 1e-9

    def test_idi_ratio_bounds(self):
        """idi_ratio must be in [0.0, 1.0]."""
        X = binary_dataset(n=100)
        for model in [always_discriminatory_model(), never_discriminatory_model()]:
            r = run_hybrid(binary_config, budget=20, model=model, X=X)
            assert 0.0 <= r["idi_ratio"] <= 1.0

    def test_idi_count_non_negative(self):
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=never_discriminatory_model(), X=X)
        assert result["idi_count"] >= 0

    def test_idi_count_at_most_inputs_generated(self):
        X = binary_dataset(n=200)
        result = run_hybrid(binary_config, budget=30, model=always_discriminatory_model(), X=X)
        assert result["idi_count"] <= result["inputs_generated"]


# ================================================================== #
# run_hybrid — model behaviour
# ================================================================== #

class TestRunHybridModelBehaviour:

    def test_never_discriminatory_gives_zero_idi(self):
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=never_discriminatory_model(), X=X)
        assert result["idi_count"] == 0
        assert result["idi_ratio"] == 0.0

    def test_always_discriminatory_gives_all_idi(self):
        """When every pair is discriminatory, idi_count == inputs_generated."""
        X = binary_dataset(n=200)
        result = run_hybrid(binary_config, budget=20, model=always_discriminatory_model(), X=X)
        assert result["idi_count"] == result["inputs_generated"]
        assert abs(result["idi_ratio"] - 1.0) < 1e-9

    def test_idi_threshold_below_not_counted(self):
        """diff = 0.04 (clearly below 0.05 threshold) → idi_count == 0."""
        model_below = MockModel(lambda row: 0.54 if row[0] >= 1 else 0.50)
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=15, model=model_below, X=X)
        assert result["idi_count"] == 0

    def test_idi_threshold_above_counted(self):
        """diff = 0.10 (clearly above 0.05 threshold) → all inputs are IDIs."""
        model_above = MockModel(lambda row: 0.60 if row[0] >= 1 else 0.50)
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=15, model=model_above, X=X)
        assert result["idi_count"] == result["inputs_generated"]

    def test_multivalue_sensitive_feature_no_crash(self):
        """Multi-value sensitive feature flipping must not crash."""
        X = multivalue_dataset(n=200)
        result = run_hybrid(multivalue_config, budget=15, model=multivalue_sensitive_model(), X=X)
        assert 0.0 <= result["idi_ratio"] <= 1.0

    def test_structured_model_finds_idis(self):
        """
        feature1_structured_model only discriminates for feature1 > 0.5.
        With budget=50 and a large enough dataset, the hybrid should find some IDIs.
        We assert >= 1 to confirm the hybrid is functioning, not just passing through.
        """
        rng = np.random.default_rng(10)
        X = pd.DataFrame({
            "age":      rng.integers(0, 2, size=500).astype(float),
            "feature1": rng.uniform(0, 1, size=500),
        })
        result = run_hybrid(binary_config, budget=50, model=feature1_structured_model(), X=X)
        assert result["idi_count"] >= 1, (
            f"Hybrid should find at least 1 IDI on a structured dataset, got {result['idi_count']}"
        )


# ================================================================== #
# run_hybrid — phase splitting
# ================================================================== #

class TestRunHybridPhaseSplit:

    def test_budget_1_only_seed_phase(self):
        """
        seed_n = max(1, int(0.15 * 1)) = 1, remaining = 0.
        Phase 2 gets budget=0 → only Phase 1 runs.
        """
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=1, model=always_discriminatory_model(), X=X)
        assert result["inputs_generated"] == 1
        assert result["idi_count"] in (0, 1)

    def test_seed_ratio_1_no_csp_phase(self):
        """
        seed_ratio=1.0 → seed_n=budget, remaining=0.
        Phase 2 never runs. All inputs come from Phase 1.
        With always-discriminatory model, idi_count == budget.
        """
        X = binary_dataset(n=100)
        result = run_hybrid(
            binary_config, budget=20,
            model=always_discriminatory_model(), X=X,
            seed_ratio=1.0
        )
        assert result["inputs_generated"] == 20
        assert result["idi_count"] == 20

    def test_seed_ratio_0_minimal_seed_phase(self):
        """
        seed_ratio=0.0 → seed_n = max(1, 0) = 1.
        Most of the budget goes to Phase 2 (CSP).
        inputs_generated should be close to budget.
        """
        X = binary_dataset(n=200)
        result = run_hybrid(
            binary_config, budget=20,
            model=always_discriminatory_model(), X=X,
            seed_ratio=0.0
        )
        assert result["inputs_generated"] <= 20
        assert result["inputs_generated"] >= 1

    def test_dt_rules_populated_when_phase1_finds_idis(self):
        """
        When Phase 1 finds discriminatory inputs, the DT should learn rules.
        With always-discriminatory model, rules list should be non-empty.
        """
        X = binary_dataset(n=200)
        result = run_hybrid(binary_config, budget=30, model=always_discriminatory_model(), X=X)
        assert len(result["dt_rules"]) >= 1

    def test_dt_rules_is_empty_list_when_no_phase1_idis(self):
        """
        When Phase 1 finds zero IDIs, the DT is trained on all-zero labels.
        DT predicts all negative → extract_rules returns [].
        Phase 2 runs unconstrained (rules=[]).
        """
        X = binary_dataset(n=100)
        result = run_hybrid(binary_config, budget=20, model=never_discriminatory_model(), X=X)
        assert result["dt_rules"] == []

    def test_phase2_receives_rules_from_phase1(self):
        """
        Indirect test: with feature1_structured_model, Phase 1 should discover
        some IDIs (feature1 > 0.5 region) and extract rules. Phase 2 constrained
        to that region should find more IDIs than random chance.
        With budget=80 and seed_ratio=0.25 (20 seed samples), the DT has enough
        signal to learn the feature1 > 0.5 split.
        """
        rng = np.random.default_rng(99)
        X = pd.DataFrame({
            "age":      rng.integers(0, 2, size=500).astype(float),
            "feature1": rng.uniform(0, 1, size=500),
        })
        result = run_hybrid(
            binary_config, budget=80,
            model=feature1_structured_model(), X=X,
            seed_ratio=0.25
        )
        # Should find a meaningful number of IDIs given the clear discrimination signal
        assert result["idi_count"] >= 20, (
            f"Hybrid with DT rules should find >= 20 IDIs on a structured dataset, "
            f"got {result['idi_count']}"
        )
