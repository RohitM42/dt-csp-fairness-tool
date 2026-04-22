"""
evaluate.py — Evaluation utilities.

Runs multiple trials of any fairness-testing method, compares IDI ratios
against a baseline using the Wilcoxon signed-rank test, and saves results.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats


def run_trials(method_fn, config, budget, model, X, n_runs=20, save_path=None, **kwargs):
    """
    Run a fairness testing method n_runs times and return IDI ratios.

    Results are saved incrementally after each run when save_path is given,
    so partial results survive an interrupted experiment.

    Args:
        method_fn:  callable — run_baseline, run_dt, run_csp, or run_hybrid
        config:     dataset config dict from config.py
        budget:     total inputs generated per run
        model:      loaded Keras model
        X:          dataset as pandas DataFrame (target column already dropped)
        n_runs:     number of independent trials
        save_path:  optional CSV path; file is written/updated after every run
        **kwargs:   extra arguments forwarded to method_fn (e.g. seed_ratio)

    Returns:
        list of idi_ratio floats, length n_runs
    """
    ratios = []
    for i in range(n_runs):
        result = method_fn(config=config, budget=budget, model=model, X=X, **kwargs)
        ratios.append(result["idi_ratio"])
        print(f"  run {i + 1}/{n_runs}: idi_ratio={result['idi_ratio']:.4f}")
        if save_path is not None:
            save_results(
                [{"run": j + 1, "idi_ratio": r} for j, r in enumerate(ratios)],
                save_path,
            )
    return ratios


def compare(baseline_ratios, method_ratios):
    """
    Compare two lists of IDI ratios using the Wilcoxon signed-rank test.

    The test is paired by run index (run i of baseline vs run i of method).
    Requires at least 2 runs; returns p_value=None when this is not met.
    Handles the all-zero-difference edge case (e.g. both methods always
    find zero IDIs) gracefully — returns p_value=1.0, significant=False.

    Args:
        baseline_ratios: list of floats from run_trials(run_baseline, ...)
        method_ratios:   list of floats from run_trials(method_fn, ...)

    Returns:
        dict with keys:
            baseline_mean, baseline_std,
            method_mean,   method_std,
            p_value (float or None), significant (bool)
    """
    b = np.array(baseline_ratios, dtype=float)
    m = np.array(method_ratios, dtype=float)

    result = {
        "baseline_mean": float(np.mean(b)),
        "baseline_std":  float(np.std(b)),
        "method_mean":   float(np.mean(m)),
        "method_std":    float(np.std(m)),
        "p_value":       None,
        "significant":   False,
    }

    if len(b) >= 2 and len(b) == len(m):
        try:
            _, p = stats.wilcoxon(b, m, zero_method="wilcox", alternative="two-sided")
            result["p_value"] = float(p)
            result["significant"] = bool(p < 0.05)
        except ValueError:
            # All paired differences are zero — methods are statistically identical
            result["p_value"] = 1.0
            result["significant"] = False

    return result


def save_results(results, path):
    """
    Save a list of result dicts to CSV.

    The parent directory is created automatically if it does not exist.

    Args:
        results: list of dicts (each dict becomes one CSV row)
        path:    file path for the output CSV
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
