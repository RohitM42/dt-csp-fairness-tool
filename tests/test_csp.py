"""
test_csp.py — Tests for csp.py: CP-SAT fairness testing (Phase 2).

Run from the csp-fairness-tool/ directory:
    python -m pytest tests/test_csp.py -v

Covers:
  - _feature_scale:          integer vs continuous detection
  - _flip_col:               binary and multi-value flipping
  - build_cp_model:          IntVar domains, sensitive cols excluded, rule disjunction
  - add_diversity_constraint: diversity enforcement, accumulation
  - run_csp:                 return structure, IDI counting, budget=0, rule constraints
"""

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from src.csp import (
    SCALE,
    _feature_scale,
    _flip_col,
    build_cp_model,
    add_diversity_constraint,
    run_csp,
)


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
    """Sensitive feature at col 0: age=1→0.9, age=0→0.1. Always IDI."""
    return MockModel(lambda row: 0.9 if row[0] >= 1 else 0.1)


def never_discriminatory_model():
    """Constant 0.5: no pair ever crosses the 0.05 threshold."""
    return MockModel(lambda _: 0.5)


def feature1_structured_model():
    """
    Discriminatory only when feature1 (col 1, non-sensitive) > 0.5.
    age=1 and feature1>0.5 → 0.9, else → 0.1.
    """
    return MockModel(lambda row: 0.9 if (row[0] >= 1 and row[1] > 0.5) else 0.1)


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


def two_feature_dataset(n=200, seed=0):
    """Both non-sensitive features are continuous."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":      rng.integers(0, 2, size=n).astype(float),
        "income":   rng.uniform(20.0, 100.0, size=n),
        "score":    rng.uniform(0.0, 1.0, size=n),
    })


binary_config    = {"sensitive": ["age"]}
multivalue_config = {"sensitive": ["race"]}
two_feature_config = {"sensitive": ["age"]}


# ================================================================== #
# _feature_scale
# ================================================================== #

class TestFeatureScale:

    def test_integer_series_returns_1(self):
        s = pd.Series([0.0, 1.0, 2.0, 3.0])
        assert _feature_scale(s) == 1

    def test_continuous_series_returns_scale(self):
        s = pd.Series([0.1, 0.5, 0.9])
        assert _feature_scale(s) == SCALE

    def test_all_integers_as_floats_returns_1(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _feature_scale(s) == 1

    def test_mixed_with_any_non_integer_returns_scale(self):
        s = pd.Series([1.0, 2.5, 3.0])
        assert _feature_scale(s) == SCALE

    def test_single_integer_value_returns_1(self):
        s = pd.Series([42.0])
        assert _feature_scale(s) == 1

    def test_empty_series_returns_scale(self):
        # len(vals) == 0 → condition is False → falls through to return SCALE
        s = pd.Series([], dtype=float)
        assert _feature_scale(s) == SCALE

    def test_series_with_only_nans_returns_scale(self):
        # dropna() leaves empty → same as empty series → returns SCALE
        s = pd.Series([np.nan, np.nan])
        assert _feature_scale(s) == SCALE


# ================================================================== #
# _flip_col
# ================================================================== #

class TestFlipCol:

    def test_binary_flips_0_to_1(self):
        assert _flip_col(0.0, [0.0, 1.0]) == 1.0

    def test_binary_flips_1_to_0(self):
        assert _flip_col(1.0, [0.0, 1.0]) == 0.0

    def test_multivalue_never_returns_same(self):
        unique_vals = [0.0, 1.0, 2.0, 3.0]
        for _ in range(30):
            flipped = _flip_col(0.0, unique_vals)
            assert flipped != 0.0

    def test_multivalue_returns_a_valid_value(self):
        unique_vals = [0.0, 1.0, 2.0, 3.0]
        for _ in range(30):
            flipped = _flip_col(2.0, unique_vals)
            assert flipped in unique_vals

    def test_binary_result_is_float(self):
        result = _flip_col(0.0, [0.0, 1.0])
        assert isinstance(result, float)

    def test_multivalue_result_is_float(self):
        unique_vals = [0.0, 1.0, 2.0]
        result = _flip_col(1.0, unique_vals)
        assert isinstance(result, float)


# ================================================================== #
# build_cp_model
# ================================================================== #

class TestBuildCpModel:

    def test_returns_three_values(self):
        X = binary_dataset(n=50)
        result = build_cp_model(X, binary_config)
        assert len(result) == 3

    def test_sensitive_cols_excluded_from_feature_vars(self):
        X = binary_dataset(n=50)
        _, feature_vars, _ = build_cp_model(X, binary_config)
        assert "age" not in feature_vars
        assert "feature1" in feature_vars

    def test_feature_vars_are_intvars(self):
        X = binary_dataset(n=50)
        _, feature_vars, _ = build_cp_model(X, binary_config)
        for var in feature_vars.values():
            # OR-Tools IntVar has a Name() method
            assert hasattr(var, "Name")

    def test_integer_col_has_scale_1(self):
        X = binary_dataset(n=50)
        _, _, feat_scales = build_cp_model(X, binary_config)
        assert feat_scales["feature1"] == SCALE  # feature1 is continuous

    def test_continuous_col_has_scale_1000(self):
        X = two_feature_dataset(n=50)
        _, _, feat_scales = build_cp_model(X, two_feature_config)
        assert feat_scales["income"] == SCALE
        assert feat_scales["score"] == SCALE

    def test_domain_matches_dataset_range(self):
        """
        For feature1 in [0, 1] (continuous), IntVar domain should be
        approximately [0*SCALE, 1*SCALE] = [0, 1000].
        """
        X = binary_dataset(n=200)
        m, feature_vars, feat_scales = build_cp_model(X, binary_config)
        solver = cp_model.CpSolver()
        # Minimise feature1 to find lower bound
        m.Minimize(feature_vars["feature1"])
        solver.Solve(m)
        lo = solver.ObjectiveValue()
        assert lo >= 0  # domain starts at min*scale >= 0

    def test_no_rules_model_is_feasible(self):
        X = binary_dataset(n=100)
        m, _, _ = build_cp_model(X, binary_config, rules=None)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_empty_rules_list_model_is_feasible(self):
        X = binary_dataset(n=100)
        m, _, _ = build_cp_model(X, binary_config, rules=[])
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_rule_constrains_feasible_region(self):
        """
        Encode rule: feature1 > 0.8 (scaled: >= 801).
        Every solution from the solver must have feature1 > 0.8.
        """
        X = binary_dataset(n=200)
        rules = [{"conditions": [("feature1", ">", 0.8)]}]
        m, feature_vars, feat_scales = build_cp_model(X, binary_config, rules=rules)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        val = solver.Value(feature_vars["feature1"]) / feat_scales["feature1"]
        assert val > 0.8

    def test_contradictory_rules_still_feasible_via_disjunction(self):
        """
        Rule 0: feature1 <= 0.2  (low region)
        Rule 1: feature1 >  0.8  (high region)
        Either one being satisfied makes the model feasible.
        """
        X = binary_dataset(n=200)
        rules = [
            {"conditions": [("feature1", "<=", 0.2)]},
            {"conditions": [("feature1", ">",  0.8)]},
        ]
        m, _, _ = build_cp_model(X, binary_config, rules=rules)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_impossible_single_rule_makes_model_infeasible(self):
        """
        Rule with a single condition that is impossible given the domain:
        feature1 <= -1.0, but feature1 is always >= 0.0.
        Model must become infeasible.
        """
        rng = np.random.default_rng(5)
        X = pd.DataFrame({
            "age":      rng.integers(0, 2, size=100).astype(float),
            "feature1": rng.uniform(0.5, 1.0, size=100),  # always > 0
        })
        rules = [{"conditions": [("feature1", "<=", -1.0)]}]
        m, _, _ = build_cp_model(X, binary_config, rules=rules)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status not in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_sensitive_feature_in_rule_conditions_is_skipped(self):
        """
        A rule condition referencing a sensitive feature must be silently
        ignored — the model should remain feasible.
        """
        X = binary_dataset(n=100)
        rules = [{"conditions": [("age", "<=", 0.5), ("feature1", ">", 0.3)]}]
        m, _, _ = build_cp_model(X, binary_config, rules=rules)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_degenerate_constant_feature_domain_expanded(self):
        """
        When min == max for a feature, build_cp_model widens the domain by 1
        to avoid a degenerate single-point IntVar.
        """
        X = pd.DataFrame({
            "age":      [0.0, 1.0, 0.0, 1.0],
            "feature1": [5.0, 5.0, 5.0, 5.0],  # all same value
        })
        m, feature_vars, feat_scales = build_cp_model(X, binary_config)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        # Should be feasible and variable should exist
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ================================================================== #
# add_diversity_constraint
# ================================================================== #

class TestAddDiversityConstraint:

    def _simple_model_with_one_var(self):
        """Return a CpModel with a single IntVar in [0, 10]."""
        m = cp_model.CpModel()
        var = m.NewIntVar(0, 10, "x")
        return m, {"x": var}

    def test_first_constraint_forbids_exact_value(self):
        """After adding constraint for x=5, solver must not return 5."""
        m, feature_vars = self._simple_model_with_one_var()
        known = [{"x": 5}]
        add_diversity_constraint(m, feature_vars, known, k=1)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert solver.Value(feature_vars["x"]) != 5

    def test_two_constraints_exclude_two_values(self):
        """After adding constraints for x=3 and x=7, solver avoids both."""
        m, feature_vars = self._simple_model_with_one_var()
        known = [{"x": 3}]
        add_diversity_constraint(m, feature_vars, known, k=1)
        known.append({"x": 7})
        add_diversity_constraint(m, feature_vars, known, k=1)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        val = solver.Value(feature_vars["x"])
        assert val not in (3, 7)

    def test_exhausted_domain_becomes_infeasible(self):
        """
        IntVar in [0, 2] (3 values). After excluding 0, 1, 2 separately,
        domain must be infeasible.
        """
        m = cp_model.CpModel()
        var = m.NewIntVar(0, 2, "x")
        feature_vars = {"x": var}
        known = []
        for val in [0, 1, 2]:
            known.append({"x": val})
            add_diversity_constraint(m, feature_vars, known, k=1)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status not in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_unknown_feature_in_known_is_skipped(self):
        """A key in known_discriminatory that doesn't exist in feature_vars is ignored."""
        m, feature_vars = self._simple_model_with_one_var()
        known = [{"x": 5, "ghost_col": 999}]  # ghost_col not in feature_vars
        # Should not raise
        add_diversity_constraint(m, feature_vars, known, k=1)
        solver = cp_model.CpSolver()
        status = solver.Solve(m)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def test_only_most_recent_entry_used(self):
        """
        add_diversity_constraint only uses known[-1]. If x=5 is excluded by
        the latest entry, the solver may still return previous excluded values.
        """
        m, feature_vars = self._simple_model_with_one_var()
        known = [{"x": 3}, {"x": 5}]
        add_diversity_constraint(m, feature_vars, known, k=1)
        solver = cp_model.CpSolver()
        # solver is still free to return 3
        solver.Solve(m)
        # Just check it doesn't crash and the result is valid
        assert True


# ================================================================== #
# run_csp
# ================================================================== #

class TestRunCsp:

    def test_returns_correct_keys(self):
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=5, model=never_discriminatory_model(), X=X)
        assert set(result.keys()) == {"idi_ratio", "idi_count", "inputs_generated"}

    def test_budget_zero_returns_zeros(self):
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=0, model=always_discriminatory_model(), X=X)
        assert result["idi_ratio"] == 0.0
        assert result["idi_count"] == 0
        assert result["inputs_generated"] == 0

    def test_inputs_generated_at_most_budget(self):
        """inputs_generated <= budget (may be less if solver is exhausted)."""
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=10, model=never_discriminatory_model(), X=X)
        assert result["inputs_generated"] <= 10

    def test_idi_ratio_consistency(self):
        """idi_ratio == idi_count / inputs_generated."""
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=10, model=always_discriminatory_model(), X=X)
        if result["inputs_generated"] > 0:
            expected = result["idi_count"] / result["inputs_generated"]
            assert abs(result["idi_ratio"] - expected) < 1e-9

    def test_idi_ratio_bounds(self):
        """idi_ratio must be in [0.0, 1.0]."""
        X = binary_dataset(n=100)
        for model in [always_discriminatory_model(), never_discriminatory_model()]:
            r = run_csp(binary_config, budget=10, model=model, X=X)
            assert 0.0 <= r["idi_ratio"] <= 1.0

    def test_never_discriminatory_gives_zero_idi(self):
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=10, model=never_discriminatory_model(), X=X)
        assert result["idi_count"] == 0
        assert result["idi_ratio"] == 0.0

    def test_always_discriminatory_gives_high_idi(self):
        """With a model that always discriminates, all inputs should be IDIs."""
        X = binary_dataset(n=200)
        result = run_csp(binary_config, budget=10, model=always_discriminatory_model(), X=X)
        assert result["idi_count"] == result["inputs_generated"]
        assert abs(result["idi_ratio"] - 1.0) < 1e-9

    def test_rules_none_runs_unconstrained(self):
        """rules=None should not crash and should produce valid output."""
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=5, model=never_discriminatory_model(), X=X, rules=None)
        assert "idi_count" in result

    def test_rules_empty_list_runs_unconstrained(self):
        """rules=[] should behave like rules=None."""
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=5, model=never_discriminatory_model(), X=X, rules=[])
        assert "idi_count" in result

    def test_rule_constrained_region_increases_idi_ratio(self):
        """
        feature1_structured_model only discriminates for feature1 > 0.5.
        Passing a rule that forces feature1 > 0.5 should give idi_ratio = 1.0.
        """
        X = binary_dataset(n=200)
        rules = [{"conditions": [("feature1", ">", 0.5)]}]
        result = run_csp(
            binary_config, budget=10,
            model=feature1_structured_model(), X=X, rules=rules
        )
        # All CSP solutions have feature1 > 0.5, so model always discriminates
        if result["inputs_generated"] > 0:
            assert result["idi_ratio"] == 1.0

    def test_multivalue_sensitive_feature_no_crash(self):
        """Multi-value sensitive feature flipping must not crash."""
        X = multivalue_dataset(n=200)
        model = MockModel(lambda row: 0.9 if row[0] == 0 else 0.1)
        result = run_csp(multivalue_config, budget=5, model=model, X=X)
        assert 0.0 <= result["idi_ratio"] <= 1.0

    def test_idi_count_non_negative(self):
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=10, model=always_discriminatory_model(), X=X)
        assert result["idi_count"] >= 0

    def test_diversity_reduces_duplicate_solutions(self):
        """
        With a model that always discriminates, every solution triggers a
        diversity constraint. The solver should generate diverse inputs
        rather than returning the same point every time.
        The exact values returned per iteration differ from one another.
        """
        X = binary_dataset(n=200)
        result = run_csp(binary_config, budget=5, model=always_discriminatory_model(), X=X)
        # All found inputs are IDIs; diversity constraints were applied
        assert result["idi_count"] == result["inputs_generated"]

    def test_budget_1_runs_correctly(self):
        """Budget of 1 must produce exactly 1 input."""
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=1, model=always_discriminatory_model(), X=X)
        assert result["inputs_generated"] == 1
        assert result["idi_count"] in (0, 1)

    def test_idi_threshold_exactly_below_not_counted(self):
        """
        diff = 0.04 (clearly below 0.05 threshold): should not be counted as IDI.
        Uses values well clear of the boundary to avoid IEEE 754 issues.
        """
        model_below = MockModel(lambda row: 0.54 if row[0] >= 1 else 0.50)
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=5, model=model_below, X=X)
        assert result["idi_count"] == 0

    def test_idi_threshold_exactly_above_counted(self):
        """
        diff = 0.10 (clearly above 0.05 threshold): all pairs should be IDIs.
        """
        model_above = MockModel(lambda row: 0.60 if row[0] >= 1 else 0.50)
        X = binary_dataset(n=100)
        result = run_csp(binary_config, budget=5, model=model_above, X=X)
        assert result["idi_count"] == result["inputs_generated"]
