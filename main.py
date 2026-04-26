"""
main.py — Entry point for dt-csp-fairness-tool.

Runs one or more fairness-testing methods against the random-search baseline,
saves per-run IDI ratios to results/, and prints a Wilcoxon comparison summary.

Usage examples:
    # Hybrid vs baseline on a single dataset
    python main.py --dataset kdd --method hybrid

    # All three methods vs baseline on KDD (baseline run once, shared)
    python main.py --dataset kdd --method all

    # DT-only vs baseline across every dataset
    python main.py --dataset all --method dt

    # Full experiment — all methods, all datasets
    python main.py --dataset all --method all --budget 1000 --runs 20
"""

import argparse
import os
import time

import keras
import pandas as pd

from src.baseline import run_baseline
from src.config import DATASET_CONFIGS, DEFAULT_BUDGET, DEFAULT_SEED_RATIO
from src.csp import run_csp
from src.dt import run_dt
from src.evaluate import compare, run_trials, save_results
from src.hybrid import run_hybrid


# Registry: method_name -> (callable, display_label, extra_kwargs_needing_seed_ratio)
_METHOD_REGISTRY = {
    "dt":     (run_dt,     "DT-only",  True),
    "csp":    (run_csp,    "CSP-only", False),
    "hybrid": (run_hybrid, "Hybrid",   True),
}


def _method_kwargs(method_name, seed_ratio):
    """Return keyword arguments required by a given method callable."""
    if _METHOD_REGISTRY[method_name][2]:
        return {"seed_ratio": seed_ratio}
    return {}


def run_experiment(dataset_name, method_names, budget, n_runs, seed_ratio, results_dir="results"):
    """
    Load the dataset and model once, then run baseline + each requested
    method for n_runs trials, compare with Wilcoxon, and print a summary.

    Returns a list of summary dicts (one per method) for the overall CSV.
    """
    config = DATASET_CONFIGS[dataset_name]

    print(f"\n{'=' * 60}")
    print(f"Dataset : {dataset_name.upper()}")
    print(f"Budget  : {budget}   Runs: {n_runs}")
    print(f"Methods : {', '.join(method_names)}")
    print(f"Results : {results_dir}/")
    print(f"{'=' * 60}")

    df = pd.read_csv(config["data_path"])
    X = df.drop(columns=[config["target"]])
    model = keras.models.load_model(config["model_path"])

    os.makedirs(results_dir, exist_ok=True)

    # Baseline — run once and share across all method comparisons for this dataset
    baseline_path = f"{results_dir}/{dataset_name}_baseline.csv"
    print(f"\n[Baseline] {n_runs} trials  ->  {baseline_path}")
    t0 = time.time()
    baseline_ratios = run_trials(
        run_baseline, config, budget, model, X,
        n_runs=n_runs, save_path=baseline_path,
    )
    baseline_time = time.time() - t0
    b_mean = sum(baseline_ratios) / len(baseline_ratios)
    print(f"  mean IDI ratio: {b_mean:.4f}  total time: {baseline_time:.1f}s  ({baseline_time/n_runs:.1f}s/run)")

    summaries = []
    for method_name in method_names:
        fn, label, _ = _METHOD_REGISTRY[method_name]
        kwargs = _method_kwargs(method_name, seed_ratio)
        method_path = f"{results_dir}/{dataset_name}_{method_name}.csv"

        print(f"\n[{label}] {n_runs} trials  ->  {method_path}")
        t0 = time.time()
        method_ratios = run_trials(
            fn, config, budget, model, X,
            n_runs=n_runs, save_path=method_path, **kwargs,
        )
        method_time = time.time() - t0

        stats = compare(baseline_ratios, method_ratios)

        p_str = f"{stats['p_value']:.4f}" if stats["p_value"] is not None else "N/A"
        sig_str = "  *significant*" if stats["significant"] else ""
        print(
            f"\n  Baseline : mean={stats['baseline_mean']:.4f}  "
            f"std={stats['baseline_std']:.4f}  time={baseline_time:.1f}s  ({baseline_time/n_runs:.1f}s/run)"
        )
        print(
            f"  {label:<10}: mean={stats['method_mean']:.4f}  "
            f"std={stats['method_std']:.4f}  time={method_time:.1f}s  ({method_time/n_runs:.1f}s/run)"
        )
        print(f"  Wilcoxon p-value: {p_str}{sig_str}")

        summaries.append({
            "dataset":             dataset_name,
            "method":              method_name,
            "baseline_mean":       stats["baseline_mean"],
            "baseline_std":        stats["baseline_std"],
            "baseline_total_time_s": round(baseline_time, 2),
            "baseline_mean_time_s":  round(baseline_time / n_runs, 2),
            "method_mean":         stats["method_mean"],
            "method_std":          stats["method_std"],
            "method_total_time_s": round(method_time, 2),
            "method_mean_time_s":  round(method_time / n_runs, 2),
            "p_value":             stats["p_value"],
            "significant":         stats["significant"],
        })

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Run fairness-testing methods and compare against baseline."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset to evaluate: single name, comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--method", default="hybrid",
        choices=["dt", "csp", "hybrid", "all"],
        help=(
            "Method to compare against baseline: "
            "dt (DT-only), csp (CSP-only), hybrid (CSP+DT), "
            "all (run all three, baseline shared). Default: hybrid"
        ),
    )
    parser.add_argument(
        "--budget", type=int, default=DEFAULT_BUDGET,
        help=f"Inputs generated per run (default: {DEFAULT_BUDGET})",
    )
    parser.add_argument(
        "--runs", type=int, default=20,
        help="Number of independent trials per method (default: 20)",
    )
    parser.add_argument(
        "--seed-ratio", type=float, default=DEFAULT_SEED_RATIO,
        dest="seed_ratio",
        help=(
            f"Fraction of budget used for Phase 1 seed sampling "
            f"in DT and Hybrid (default: {DEFAULT_SEED_RATIO})"
        ),
    )
    parser.add_argument(
        "--results-dir", default="results",
        dest="results_dir",
        help="Directory to save result CSVs (default: results)",
    )
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [d.strip() for d in args.dataset.split(",")]
    invalid = [d for d in datasets if d not in DATASET_CONFIGS]
    if invalid:
        parser.error(f"Unknown dataset(s): {', '.join(invalid)}. Valid: {', '.join(DATASET_CONFIGS.keys())}")
    methods = ["dt", "csp", "hybrid"] if args.method == "all" else [args.method]

    all_summaries = []
    summary_path = f"{args.results_dir}/summary.csv"
    for dataset in datasets:
        summaries = run_experiment(
            dataset, methods, args.budget, args.runs, args.seed_ratio, args.results_dir
        )
        all_summaries.extend(summaries)
        if all_summaries:
            save_results(all_summaries, summary_path)
            print(f"\nSummary updated ({len(all_summaries)} rows) -> {summary_path}")


if __name__ == "__main__":
    main()
