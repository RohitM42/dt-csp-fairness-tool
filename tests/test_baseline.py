"""
test_baseline.py — Unit tests for baseline.py random search.

Run from the dt-csp-fairness-tool/ directory:
    python -m pytest tests/test_baseline.py -v

Covers:
  - run_baseline: return structure, inputs_generated, idi_ratio consistency,
                  bounds, always/never discriminatory, multivalue sensitive,
                  threshold boundary, deduplication, budget edge cases
"""

import numpy as np
import pandas as pd

from src.baseline import run_baseline


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
    """Sensitive feature at col 0: value>=1 → 0.9, else → 0.1. Always discriminatory."""
    return MockModel(lambda row: 0.9 if row[0] >= 1 else 0.1)


def never_discriminatory_model():
    """Constant 0.5: no pair ever exceeds the 0.05 threshold."""
    return MockModel(lambda _: 0.5)


def multivalue_sensitive_model():
    """race (col 0): predicts 0.9 if race==0 else 0.1."""
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


binary_config     = {"sensitive": ["age"]}
multivalue_config = {"sensitive": ["race"]}


# ================================================================== #
# run_baseline
# ================================================================== #

class TestRunBaseline:

    def test_returns_correct_keys(self):
        X = binary_dataset(n=50)
        result = run_baseline(binary_config, budget=20, model=always_discriminatory_model(), X=X)
        assert set(result.keys()) == {"idi_ratio", "idi_count", "inputs_generated"}

    def test_inputs_generated_equals_budget(self):
        X = binary_dataset(n=100)
        result = run_baseline(binary_config, budget=30, model=always_discriminatory_model(), X=X)
        assert result["inputs_generated"] == 30

    def test_idi_ratio_consistency(self):
        """idi_ratio must equal idi_count / inputs_generated."""
        X = binary_dataset(n=100)
        result = run_baseline(binary_config, budget=30, model=always_discriminatory_model(), X=X)
        expected = result["idi_count"] / result["inputs_generated"]
        assert abs(result["idi_ratio"] - expected) < 1e-9

    def test_idi_ratio_bounds(self):
        """idi_ratio must always be in [0.0, 1.0]."""
        X = binary_dataset(n=100)
        for model in [always_discriminatory_model(), never_discriminatory_model()]:
            r = run_baseline(binary_config, budget=30, model=model, X=X)
            assert 0.0 <= r["idi_ratio"] <= 1.0

    def test_always_discriminatory_model_gives_high_ratio(self):
        """Every pair is discriminatory; unique IDIs should be nearly == budget."""
        X = binary_dataset(n=2000)
        result = run_baseline(binary_config, budget=50, model=always_discriminatory_model(), X=X)
        assert result["idi_count"] >= 45
        assert result["idi_ratio"] >= 0.9

    def test_never_discriminatory_model_gives_ratio_0(self):
        X = binary_dataset(n=100)
        result = run_baseline(binary_config, budget=30, model=never_discriminatory_model(), X=X)
        assert result["idi_count"] == 0
        assert result["idi_ratio"] == 0.0

    def test_multivalue_sensitive_feature(self):
        """Multi-value sensitive feature must not crash and return valid structure."""
        X = multivalue_dataset(n=200)
        result = run_baseline(multivalue_config, budget=30, model=multivalue_sensitive_model(), X=X)
        assert result["inputs_generated"] == 30
        assert 0.0 <= result["idi_ratio"] <= 1.0

    def test_threshold_boundary_below(self):
        """diff = 0.04, clearly below 0.05 → no IDIs."""
        model_below = MockModel(lambda row: 0.54 if row[0] >= 1 else 0.50)
        X = binary_dataset(n=2000)
        result = run_baseline(binary_config, budget=30, model=model_below, X=X)
        assert result["idi_count"] == 0

    def test_threshold_boundary_above(self):
        """diff = 0.10, clearly above 0.05 → IDIs found."""
        model_above = MockModel(lambda row: 0.60 if row[0] >= 1 else 0.50)
        X = binary_dataset(n=2000)
        result = run_baseline(binary_config, budget=30, model=model_above, X=X)
        assert result["idi_count"] >= 28

    def test_deduplication(self):
        """
        Same discriminatory input seen multiple times must count only once.
        Use a tiny dataset (2 rows) with a large budget so repeats are guaranteed.
        """
        X = pd.DataFrame({
            "age":      [0.0, 1.0],
            "feature1": [0.5, 0.5],
        })
        result = run_baseline(binary_config, budget=100, model=always_discriminatory_model(), X=X)
        # Only 2 unique rows exist, so at most 2 unique discriminatory inputs
        assert result["idi_count"] <= 2

    def test_budget_1(self):
        """Budget of 1 must not crash and return sensible results."""
        X = binary_dataset(n=50)
        result = run_baseline(binary_config, budget=1, model=always_discriminatory_model(), X=X)
        assert result["inputs_generated"] == 1
        assert result["idi_count"] in (0, 1)

    def test_idi_count_in_valid_range(self):
        X = binary_dataset(n=200)
        result = run_baseline(binary_config, budget=50, model=always_discriminatory_model(), X=X)
        assert 0 <= result["idi_count"] <= 50
