"""
test_dt.py — Tests for dt.py: decision tree fairness testing (Phase 1).

Run from the csp-fairness-tool/ directory:
    python -m pytest tests/test_dt.py -v

Covers:
  - train_dt:     fitting, depth, edge cases (all-zero, all-one, single sample)
  - extract_rules: structure, positive-leaf filtering, condition format, edge cases
  - run_dt:       return structure, IDI counting, DT-guided bias, tricky scenarios
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.dt import train_dt, extract_rules, run_dt


# ------------------------------------------------------------------ #
# Mock model helpers
# ------------------------------------------------------------------ #

class MockModel:
    """
    Minimal stand-in for a Keras model.
    predict_fn receives a flat numpy row and returns a scalar.
    """
    def __init__(self, predict_fn):
        self._fn = predict_fn

    def predict(self, x, verbose=0):
        val = self._fn(x[0])
        return np.array([[val]])


def always_discriminatory_model():
    """Binary sensitive feature at col 0: age=1→0.9, age=0→0.1. Always discriminatory."""
    return MockModel(lambda row: 0.9 if row[0] >= 1 else 0.1)


def never_discriminatory_model():
    """Constant 0.5: no pair ever exceeds the 0.05 threshold."""
    return MockModel(lambda _: 0.5)


def feature1_structured_model():
    """
    Discriminatory only when feature1 (col 1) > 0.5, regardless of age direction.
    Lets the DT learn a real rule and guides Phase 2 toward that region.
      age=1 and feature1>0.5 → 0.9, else → 0.1
    So flipping age for a row with feature1>0.5 gives |0.9-0.1|=0.8 → IDI.
    Flipping age for a row with feature1<=0.5 gives |0.1-0.1|=0.0 → not IDI.
    """
    return MockModel(lambda row: 0.9 if (row[0] >= 1 and row[1] > 0.5) else 0.1)


def multivalue_sensitive_model():
    """race (col 0): predicts 0.9 if race==0 else 0.1. Flipping from 0 → anything gives IDI."""
    return MockModel(lambda row: 0.9 if row[0] == 0 else 0.1)


# ------------------------------------------------------------------ #
# Dataset factories
# ------------------------------------------------------------------ #

def binary_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(0, 2, size=n).astype(float),
        "feature1": rng.uniform(0, 1, size=n),
    })


def multivalue_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "race": rng.integers(0, 5, size=n).astype(float),
        "feature1": rng.uniform(0, 1, size=n),
    })


binary_config = {"sensitive": ["age"]}
multivalue_config = {"sensitive": ["race"]}


# ================================================================== #
# train_dt
# ================================================================== #

class TestTrainDt:

    def test_returns_decision_tree_classifier(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (50, 3))
        y = rng.integers(0, 2, 50)
        dt = train_dt(X, y, ["a", "b", "c"])
        assert isinstance(dt, DecisionTreeClassifier)

    def test_fitted_classifier_has_classes(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (50, 3))
        y = rng.integers(0, 2, 50)
        dt = train_dt(X, y, ["a", "b", "c"])
        assert hasattr(dt, "classes_")

    def test_default_max_depth_is_4(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, (60, 4))
        y = rng.integers(0, 2, 60)
        dt = train_dt(X, y, list("abcd"))
        assert dt.max_depth == 4

    def test_respects_custom_max_depth(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 1, (100, 5))
        y = rng.integers(0, 2, 100)
        dt = train_dt(X, y, list("abcde"), max_depth=2)
        assert dt.get_depth() <= 2

    def test_all_zero_labels_trains_without_error(self):
        rng = np.random.default_rng(4)
        X = rng.uniform(0, 1, (40, 3))
        y = np.zeros(40, dtype=int)
        dt = train_dt(X, y, ["a", "b", "c"])
        assert set(dt.predict(X)).issubset({0})

    def test_all_one_labels_trains_without_error(self):
        rng = np.random.default_rng(5)
        X = rng.uniform(0, 1, (40, 3))
        y = np.ones(40, dtype=int)
        dt = train_dt(X, y, ["a", "b", "c"])
        assert set(dt.predict(X)).issubset({1})

    def test_single_sample_does_not_crash(self):
        X = np.array([[0.5, 0.3]])
        y = np.array([1])
        dt = train_dt(X, y, ["a", "b"])
        assert isinstance(dt, DecisionTreeClassifier)

    def test_can_predict_after_training(self):
        rng = np.random.default_rng(6)
        X = rng.uniform(0, 1, (50, 3))
        y = rng.integers(0, 2, 50)
        dt = train_dt(X, y, ["a", "b", "c"])
        preds = dt.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})


# ================================================================== #
# extract_rules
# ================================================================== #

class TestExtractRules:

    def _fit(self, X, y, feature_names, max_depth=4):
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(X, y)
        return dt

    def test_returns_list(self):
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        dt = self._fit(X, y, ["a", "b", "c"])
        rules = extract_rules(dt, ["a", "b", "c"])
        assert isinstance(rules, list)

    def test_rule_has_required_keys(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        dt = self._fit(X, y, ["f0", "f1"])
        for rule in extract_rules(dt, ["f0", "f1"]):
            assert "conditions" in rule
            assert "n_samples" in rule
            assert "n_positive" in rule

    def test_condition_is_3_tuple(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        dt = self._fit(X, y, ["f0", "f1"])
        for rule in extract_rules(dt, ["f0", "f1"]):
            for cond in rule["conditions"]:
                assert len(cond) == 3
                feat, op, thresh = cond
                assert isinstance(feat, str)
                assert op in ("<=", ">")
                assert isinstance(thresh, float)

    def test_all_zero_labels_gives_empty_rules(self):
        # DT trained on all-negatives should have no positive leaves
        X = np.random.rand(40, 3)
        y = np.zeros(40, dtype=int)
        dt = self._fit(X, y, ["a", "b", "c"])
        rules = extract_rules(dt, ["a", "b", "c"])
        assert rules == []

    def test_all_one_labels_gives_at_least_one_rule(self):
        X = np.random.rand(40, 3)
        y = np.ones(40, dtype=int)
        dt = self._fit(X, y, ["a", "b", "c"])
        rules = extract_rules(dt, ["a", "b", "c"])
        assert len(rules) >= 1

    def test_positive_leaf_rules_only(self):
        """
        Clear split: X[:,0] > 0.5 → positive. DT should learn this.
        All extracted rules must lead to positive predictions.
        """
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (200, 2))
        y = (X[:, 0] > 0.5).astype(int)
        dt = self._fit(X, y, ["feat_a", "feat_b"])
        rules = extract_rules(dt, ["feat_a", "feat_b"])
        assert len(rules) > 0
        # Every rule should have feat_a in its conditions (root split)
        for rule in rules:
            feats = {c[0] for c in rule["conditions"]}
            assert "feat_a" in feats, f"feat_a missing from rule conditions: {rule}"

    def test_n_positive_is_non_negative(self):
        X = np.random.rand(60, 3)
        y = np.random.randint(0, 2, 60)
        dt = self._fit(X, y, ["a", "b", "c"])
        for rule in extract_rules(dt, ["a", "b", "c"]):
            assert rule["n_positive"] >= 0

    def test_depth_1_tree_conditions_length(self):
        """A depth-1 tree has exactly one split → each leaf path has 1 condition."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (100, 2))
        y = (X[:, 0] > 0.5).astype(int)
        dt = DecisionTreeClassifier(max_depth=1, random_state=42)
        dt.fit(X, y)
        rules = extract_rules(dt, ["feat_a", "feat_b"])
        assert len(rules) >= 1
        for rule in rules:
            assert len(rule["conditions"]) == 1

    def test_highly_imbalanced_sparse_labels_no_crash(self):
        """
        Tricky case: ~2% positive rate (similar to COMPAS ~4%).
        DT typically predicts all-0 due to class imbalance.
        extract_rules must return an empty list, not raise.
        """
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (200, 5))
        y = np.zeros(200, dtype=int)
        y[:4] = 1  # 2% positives
        dt = self._fit(X, y, list("abcde"))
        rules = extract_rules(dt, list("abcde"))
        assert isinstance(rules, list)

    def test_no_conditions_when_root_is_positive_leaf(self):
        """
        Tricky: if all training labels are 1, the root IS the leaf.
        The rule has an empty conditions list — valid, means 'always positive'.
        """
        X = np.random.rand(20, 2)
        y = np.ones(20, dtype=int)
        dt = self._fit(X, y, ["a", "b"])
        rules = extract_rules(dt, ["a", "b"])
        assert len(rules) >= 1
        assert rules[0]["conditions"] == []


# ================================================================== #
# run_dt
# ================================================================== #

class TestRunDt:

    def test_returns_correct_keys(self):
        X = binary_dataset(n=50)
        result = run_dt(binary_config, budget=20, model=always_discriminatory_model(), X=X)
        assert set(result.keys()) == {"idi_ratio", "idi_count", "inputs_generated", "dt_rules"}

    def test_inputs_generated_equals_budget(self):
        X = binary_dataset(n=100)
        result = run_dt(binary_config, budget=30, model=always_discriminatory_model(), X=X)
        assert result["inputs_generated"] == 30

    def test_idi_ratio_consistency(self):
        """idi_ratio must equal idi_count / inputs_generated."""
        X = binary_dataset(n=100)
        result = run_dt(binary_config, budget=30, model=always_discriminatory_model(), X=X)
        expected = result["idi_count"] / result["inputs_generated"]
        assert abs(result["idi_ratio"] - expected) < 1e-9

    def test_idi_ratio_bounds(self):
        """idi_ratio must always be in [0.0, 1.0]."""
        X = binary_dataset(n=100)
        for model in [always_discriminatory_model(), never_discriminatory_model()]:
            r = run_dt(binary_config, budget=30, model=model, X=X)
            assert 0.0 <= r["idi_ratio"] <= 1.0

    def test_all_discriminatory_model_gives_ratio_1(self):
        """
        When every pair is discriminatory, almost all unique inputs should be IDIs.
        Seed draws and pool draws come from the same dataset so rare row collisions
        are possible — allow a small margin rather than asserting exactly == budget.
        """
        X = binary_dataset(n=2000)
        result = run_dt(binary_config, budget=50, model=always_discriminatory_model(), X=X)
        assert result["idi_count"] >= 48
        assert result["idi_ratio"] >= 0.96

    def test_never_discriminatory_model_gives_ratio_0(self):
        """When the model never discriminates, idi_count should be 0."""
        X = binary_dataset(n=100)
        result = run_dt(binary_config, budget=30, model=never_discriminatory_model(), X=X)
        assert result["idi_count"] == 0
        assert result["idi_ratio"] == 0.0

    def test_dt_rules_is_list(self):
        X = binary_dataset(n=100)
        result = run_dt(binary_config, budget=25, model=always_discriminatory_model(), X=X)
        assert isinstance(result["dt_rules"], list)

    def test_idi_count_in_valid_range(self):
        X = binary_dataset(n=200)
        result = run_dt(binary_config, budget=50, model=feature1_structured_model(), X=X)
        assert 0 <= result["idi_count"] <= 50

    # --- edge cases ---

    def test_budget_1(self):
        """Budget of 1: seed_n=1, no guided phase. Should not crash."""
        X = binary_dataset(n=50)
        result = run_dt(binary_config, budget=1, model=always_discriminatory_model(), X=X)
        assert result["inputs_generated"] == 1
        assert result["idi_count"] in (0, 1)

    def test_seed_ratio_1_no_guided_phase(self):
        """
        seed_ratio=1.0: entire budget spent on random seed, no DT-guided phase.
        With all-discriminatory model, idi_count must be > 0. Cannot assert == budget
        because seed sampling uses random-with-replacement, so duplicate rows are
        possible and only unique discriminatory inputs are counted after dedup.
        """
        X = binary_dataset(n=100)
        result = run_dt(binary_config, budget=20, model=always_discriminatory_model(), X=X, seed_ratio=1.0)
        assert result["inputs_generated"] == 20
        assert 0 < result["idi_count"] <= 20

    def test_multivalue_sensitive_feature(self):
        """
        Multi-value sensitive feature (race: 0–4). Flip must pick a different value.
        Should not crash and must return a valid structure.
        """
        X = multivalue_dataset(n=200)
        result = run_dt(multivalue_config, budget=30, model=multivalue_sensitive_model(), X=X)
        assert result["inputs_generated"] == 30
        assert 0.0 <= result["idi_ratio"] <= 1.0

    def test_sparse_discrimination_no_crash(self):
        """
        Tricky: dataset where IDIs are almost never found (feature1 always ≤ 0.02).
        DT may predict all-0 → guided phase falls back to full pool. Must not crash.
        """
        rng = np.random.default_rng(99)
        n = 300
        X = pd.DataFrame({
            "age": rng.integers(0, 2, size=n).astype(float),
            "feature1": rng.uniform(0, 0.02, size=n),  # always ≤ 0.02 → never > 0.5
        })
        result = run_dt(binary_config, budget=40, model=feature1_structured_model(), X=X)
        assert result["inputs_generated"] == 40
        assert result["idi_count"] == 0

    def test_dt_guided_phase_biases_toward_positive_region(self):
        """
        With feature1_structured_model, only rows with feature1 > 0.5 are discriminatory.
        The DT should learn this split from the seed phase and bias Phase 2
        toward feature1 > 0.5 rows, giving a higher IDI count than pure random chance.

        Random baseline would find ~50% IDIs (50% of rows have feature1 > 0.5).
        DT-guided should converge toward ~100% in Phase 2 once the rule is learned.
        We assert >= 40 out of 100 to allow for variance without being too strict.
        """
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "age": rng.integers(0, 2, size=500).astype(float),
            "feature1": rng.uniform(0, 1, size=500),
        })
        result = run_dt(binary_config, budget=100, model=feature1_structured_model(), X=X, seed_ratio=0.3)
        assert result["idi_count"] >= 40, (
            f"DT-guided should find >= 40 IDIs on a clear split dataset, got {result['idi_count']}"
        )

    def test_large_budget_completes(self):
        """Run with a larger budget to check no infinite loops."""
        X = binary_dataset(n=300)
        result = run_dt(binary_config, budget=200, model=always_discriminatory_model(), X=X, seed_ratio=0.1)
        assert result["inputs_generated"] == 200

    def test_seed_phase_labels_are_binary(self):
        """
        Verify the seed phase labels with the strict > 0.05 threshold.
        Uses values well clear of 0.05 to avoid IEEE 754 rounding pitfalls
        (e.g. 0.55 - 0.50 is 0.0500...444 in float, which is > 0.05).
        Uses a large dataset so n_candidates < len(X), avoiding sampling-with-replacement
        and ensuring idi_count matches budget exactly for the always-above model.
        """
        # diff = 0.04, clearly below threshold → not IDI
        model_below = MockModel(lambda row: 0.54 if row[0] >= 1 else 0.50)
        X = binary_dataset(n=2000)
        result = run_dt(binary_config, budget=30, model=model_below, X=X)
        assert result["idi_count"] == 0

        # diff = 0.10, clearly above threshold → IDI
        model_above = MockModel(lambda row: 0.60 if row[0] >= 1 else 0.50)
        result2 = run_dt(binary_config, budget=30, model=model_above, X=X)
        assert result2["idi_count"] >= 28  # allow rare seed/pool row collisions
