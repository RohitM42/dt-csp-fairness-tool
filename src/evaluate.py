"""
evaluate.py — Evaluation utilities.

Runs multiple trials of baseline and hybrid, compares IDI ratios,
runs Wilcoxon signed-rank test, and saves results.
"""


def run_trials(method_fn, config, budget, model, X, n_runs=20, **kwargs):
    """
    Run a fairness testing method n_runs times and return IDI ratios.

    Args:
        method_fn: callable (run_baseline or run_hybrid)
        config, budget, model, X: passed through to method_fn
        n_runs: number of independent trials
        **kwargs: extra args forwarded to method_fn

    Returns:
        list of IDI ratio floats, length n_runs
    """
    # TODO: loop n_runs, call method_fn, collect idi_ratio each run
    raise NotImplementedError


def compare(baseline_ratios, hybrid_ratios):
    """
    Compare two lists of IDI ratios using Wilcoxon signed-rank test.

    Returns:
        dict with keys: baseline_mean, baseline_std, hybrid_mean,
                        hybrid_std, p_value, significant
    """
    # TODO: scipy.stats.wilcoxon
    raise NotImplementedError


def save_results(results, path):
    """Save per-run results to CSV."""
    # TODO: pandas DataFrame -> CSV
    raise NotImplementedError
