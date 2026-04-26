"""
test_evaluate.py — Tests for evaluate.py: run_trials, compare, save_results.

Run from the dt-csp-fairness-tool/ directory:
    python -m pytest tests/test_evaluate.py -v

All tests use lightweight mock callables and synthetic data — no real datasets
or .h5 model files are required.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compare, run_trials, save_results


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _fixed_ratio_fn(ratio):
    """Return a method_fn that always reports a fixed IDI ratio."""
    def fn(config, budget, model, X, **kwargs):
        return {"idi_ratio": ratio, "idi_count": 0, "inputs_generated": budget}
    return fn


def _counting_fn(store):
    """Return a method_fn that appends its call count to store and returns ratio=0."""
    def fn(config, budget, model, X, **kwargs):
        store.append(1)
        return {"idi_ratio": 0.0, "idi_count": 0, "inputs_generated": budget}
    return fn


def _ratio_sequence_fn(ratios):
    """Return a method_fn that cycles through a list of ratios, one per call."""
    calls = [0]
    def fn(config, budget, model, X, **kwargs):
        r = ratios[calls[0] % len(ratios)]
        calls[0] += 1
        return {"idi_ratio": r, "idi_count": 0, "inputs_generated": budget}
    return fn


DUMMY_CONFIG = {"sensitive": ["age"]}
DUMMY_BUDGET = 10
DUMMY_MODEL  = None   # method_fn mocks don't use model
DUMMY_X      = pd.DataFrame({"age": [0.0, 1.0], "f1": [0.2, 0.8]})


# ================================================================== #
# run_trials
# ================================================================== #

class TestRunTrials:

    def test_returns_list_of_floats(self):
        ratios = run_trials(
            _fixed_ratio_fn(0.3), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=5,
        )
        assert isinstance(ratios, list)
        assert all(isinstance(r, float) for r in ratios)

    def test_length_equals_n_runs(self):
        ratios = run_trials(
            _fixed_ratio_fn(0.5), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=7,
        )
        assert len(ratios) == 7

    def test_method_called_exactly_n_runs_times(self):
        store = []
        run_trials(
            _counting_fn(store), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=4,
        )
        assert len(store) == 4

    def test_correct_ratio_values_collected(self):
        ratios = run_trials(
            _fixed_ratio_fn(0.75), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=3,
        )
        assert ratios == [0.75, 0.75, 0.75]

    def test_varying_ratios_collected_in_order(self):
        sequence = [0.1, 0.2, 0.3, 0.4]
        ratios = run_trials(
            _ratio_sequence_fn(sequence), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=4,
        )
        assert ratios == sequence

    def test_n_runs_1_works(self):
        ratios = run_trials(
            _fixed_ratio_fn(0.42), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=1,
        )
        assert ratios == [0.42]

    def test_kwargs_forwarded_to_method_fn(self):
        """Extra kwargs (e.g. seed_ratio) must be passed through to method_fn."""
        received = {}
        def fn(config, budget, model, X, **kwargs):
            received.update(kwargs)
            return {"idi_ratio": 0.0, "idi_count": 0, "inputs_generated": budget}

        run_trials(
            fn, DUMMY_CONFIG, DUMMY_BUDGET, DUMMY_MODEL, DUMMY_X,
            n_runs=1, seed_ratio=0.25,
        )
        assert received.get("seed_ratio") == 0.25

    def test_no_save_path_does_not_write_files(self, tmp_path):
        """When save_path is None, no files should be written."""
        before = set(tmp_path.iterdir())
        run_trials(
            _fixed_ratio_fn(0.1), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=3, save_path=None,
        )
        assert set(tmp_path.iterdir()) == before

    def test_save_path_creates_csv_after_first_run(self, tmp_path):
        path = str(tmp_path / "out.csv")
        run_trials(
            _fixed_ratio_fn(0.2), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=3, save_path=path,
        )
        assert os.path.exists(path)

    def test_save_path_csv_has_correct_row_count(self, tmp_path):
        """CSV must have one row per completed run."""
        path = str(tmp_path / "out.csv")
        run_trials(
            _fixed_ratio_fn(0.3), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=5, save_path=path,
        )
        df = pd.read_csv(path)
        assert len(df) == 5

    def test_save_path_csv_has_run_and_idi_ratio_columns(self, tmp_path):
        path = str(tmp_path / "out.csv")
        run_trials(
            _fixed_ratio_fn(0.4), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=2, save_path=path,
        )
        df = pd.read_csv(path)
        assert "run" in df.columns
        assert "idi_ratio" in df.columns

    def test_incremental_save_after_each_run(self, tmp_path, monkeypatch):
        """
        After run k, the CSV must contain exactly k rows.
        We verify by inspecting the file after every call to save_results.
        """
        path = str(tmp_path / "incremental.csv")
        row_counts = []

        original_save = save_results

        def capturing_save(results, p):
            original_save(results, p)
            if p == path:
                row_counts.append(len(pd.read_csv(p)))

        monkeypatch.setattr("src.evaluate.save_results", capturing_save)

        run_trials(
            _fixed_ratio_fn(0.1), DUMMY_CONFIG, DUMMY_BUDGET,
            DUMMY_MODEL, DUMMY_X, n_runs=4, save_path=path,
        )
        assert row_counts == [1, 2, 3, 4]


# ================================================================== #
# compare
# ================================================================== #

class TestCompare:

    def test_returns_correct_keys(self):
        result = compare([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
        expected_keys = {
            "baseline_mean", "baseline_std",
            "method_mean", "method_std",
            "p_value", "significant",
        }
        assert set(result.keys()) == expected_keys

    def test_baseline_mean_correct(self):
        result = compare([0.0, 0.2, 0.4], [0.5, 0.5, 0.5])
        assert abs(result["baseline_mean"] - 0.2) < 1e-9

    def test_method_mean_correct(self):
        result = compare([0.1, 0.1, 0.1], [0.4, 0.6, 0.5])
        assert abs(result["method_mean"] - 0.5) < 1e-9

    def test_baseline_std_correct(self):
        result = compare([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        assert result["baseline_std"] == 0.0

    def test_significant_true_when_methods_clearly_differ(self):
        """
        Baseline always 0.0, method always 1.0 → maximum possible difference,
        Wilcoxon should give p < 0.05.
        """
        baseline = [0.0] * 20
        method   = [1.0] * 20
        result = compare(baseline, method)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_significant_false_when_methods_identical(self):
        """All paired differences are zero → p_value=1.0, not significant."""
        ratios = [0.3] * 10
        result = compare(ratios, ratios)
        assert result["significant"] is False
        assert result["p_value"] == 1.0

    def test_p_value_is_none_when_n_runs_is_1(self):
        """Wilcoxon is invalid with a single observation — p_value must be None."""
        result = compare([0.5], [0.6])
        assert result["p_value"] is None
        assert result["significant"] is False

    def test_p_value_is_none_when_lists_empty(self):
        result = compare([], [])
        assert result["p_value"] is None

    def test_all_zero_baseline_and_method(self):
        """Both methods find zero IDIs every run — should not crash."""
        result = compare([0.0] * 10, [0.0] * 10)
        assert result["baseline_mean"] == 0.0
        assert result["method_mean"] == 0.0
        assert result["p_value"] == 1.0

    def test_significant_false_when_random_noise_no_real_difference(self):
        """
        When both methods draw from the same distribution, Wilcoxon should
        not find significance. Use fixed values that are clearly the same.
        """
        same = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
        result = compare(same, same)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_p_value_is_float_when_valid(self):
        baseline = [0.0] * 10
        method   = [0.5] * 10
        result = compare(baseline, method)
        assert isinstance(result["p_value"], float)

    def test_significant_is_bool(self):
        result = compare([0.1] * 5, [0.2] * 5)
        assert isinstance(result["significant"], bool)


# ================================================================== #
# save_results
# ================================================================== #

class TestSaveResults:

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "out.csv")
        save_results([{"run": 1, "idi_ratio": 0.5}], path)
        assert os.path.exists(path)

    def test_csv_has_correct_columns(self, tmp_path):
        path = str(tmp_path / "out.csv")
        save_results([{"run": 1, "value": 42}], path)
        df = pd.read_csv(path)
        assert list(df.columns) == ["run", "value"]

    def test_csv_has_correct_row_count(self, tmp_path):
        path = str(tmp_path / "out.csv")
        data = [{"run": i, "idi_ratio": i * 0.1} for i in range(5)]
        save_results(data, path)
        df = pd.read_csv(path)
        assert len(df) == 5

    def test_csv_values_are_correct(self, tmp_path):
        path = str(tmp_path / "out.csv")
        save_results([{"run": 1, "idi_ratio": 0.123}], path)
        df = pd.read_csv(path)
        assert abs(df["idi_ratio"].iloc[0] - 0.123) < 1e-6

    def test_creates_parent_directory_if_missing(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "out.csv")
        save_results([{"x": 1}], path)
        assert os.path.exists(path)

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "out.csv")
        save_results([{"run": 1, "idi_ratio": 0.1}], path)
        save_results([{"run": 1, "idi_ratio": 0.9}, {"run": 2, "idi_ratio": 0.8}], path)
        df = pd.read_csv(path)
        assert len(df) == 2
        assert abs(df["idi_ratio"].iloc[0] - 0.9) < 1e-6

    def test_empty_results_creates_file(self, tmp_path):
        # An empty list produces an empty file — just verify it is created without error
        path = str(tmp_path / "empty.csv")
        save_results([], path)
        assert os.path.exists(path)
