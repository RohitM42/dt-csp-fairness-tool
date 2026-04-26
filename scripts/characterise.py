"""
characterise.py — Group fairness characterisation of pre-trained DNN models.

Computes Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD)
for each sensitive feature across all configured datasets. Intended as a one-time
characterisation pass to profile model-level bias before running IDI search.

Outputs a summary table to terminal and saves characterisation.csv at repo root.

Usage:
    python scripts/characterise.py                    # all datasets
    python scripts/characterise.py --dataset adult    # single dataset
"""

import argparse
import csv
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # suppress TF/Keras verbosity

from src.config import DATASET_CONFIGS


def load_model(path):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def get_predictions(model, X):
    """Return a flat array of scalar predictions (probability of positive class)."""
    raw = model.predict(X.values, verbose=0)
    if raw.ndim > 1 and raw.shape[1] > 1:
        # multi-class: use column 1 as positive class probability
        return raw[:, 1].flatten()
    return raw.flatten()


def compute_dpd(X, preds, feature):
    """
    Demographic Parity Difference.
    Max difference in mean positive prediction rate across sensitive attribute groups.
    Does not require ground truth labels.
    """
    groups = sorted(X[feature].unique())
    if len(groups) < 2:
        return float("nan")
    rates = {g: preds[X[feature] == g].mean() for g in groups}
    return max(rates.values()) - min(rates.values())


def compute_eod(X, y, preds, feature, threshold=0.5):
    """
    Equalized Odds Difference.
    Max of |TPR difference| and |FPR difference| across sensitive attribute groups.
    Requires binary ground truth labels. Returns (tpr_diff, fpr_diff, eod).
    Returns (nan, nan, nan) if target is not binary.
    """
    unique_y = np.unique(y)
    if len(unique_y) != 2:
        return float("nan"), float("nan"), float("nan")

    pos_label = unique_y[1]
    bin_preds = (preds >= threshold).astype(int)
    y_bin = (y == pos_label).astype(int)

    groups = sorted(X[feature].unique())
    if len(groups) < 2:
        return float("nan"), float("nan"), float("nan")

    tprs, fprs = {}, {}
    for g in groups:
        mask = X[feature] == g
        yg, pg = y_bin[mask], bin_preds[mask]
        pos_mask, neg_mask = yg == 1, yg == 0
        tprs[g] = pg[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
        fprs[g] = pg[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

    tpr_diff = max(tprs.values()) - min(tprs.values())
    fpr_diff = max(fprs.values()) - min(fprs.values())
    eod = max(tpr_diff, fpr_diff)
    return tpr_diff, fpr_diff, eod


def fmt(val):
    return f"{val:.4f}" if not (isinstance(val, float) and np.isnan(val)) else "  n/a"


def characterise(name, config):
    print(f"\n{'='*60}")
    print(f"  {name.upper()}")
    print(f"{'='*60}")

    df = pd.read_csv(config["data_path"])
    target = config["target"]
    sensitive = config["sensitive"]

    y = df[target].values
    X = df.drop(columns=[target])

    print(f"  Rows: {len(X):,}   Features: {len(X.columns)}   Sensitive: {sensitive}")

    model = load_model(config["model_path"])
    preds = get_predictions(model, X)

    print(f"\n  {'Sensitive feature':<24} {'DPD':>8} {'TPR diff':>10} {'FPR diff':>10} {'EOD':>8}")
    print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    rows = []
    for feat in sensitive:
        dpd = compute_dpd(X, preds, feat)
        tpr_diff, fpr_diff, eod = compute_eod(X, y, preds, feat)

        print(f"  {feat:<24} {fmt(dpd):>8} {fmt(tpr_diff):>10} {fmt(fpr_diff):>10} {fmt(eod):>8}")
        rows.append({
            "dataset": name,
            "sensitive_feature": feat,
            "n_rows": len(X),
            "n_features": len(X.columns),
            "dpd": round(dpd, 4) if not np.isnan(dpd) else None,
            "eod_tpr_diff": round(tpr_diff, 4) if not np.isnan(tpr_diff) else None,
            "eod_fpr_diff": round(fpr_diff, 4) if not np.isnan(fpr_diff) else None,
            "eod": round(eod, 4) if not np.isnan(eod) else None,
        })

    return rows


def save_csv(all_rows, path="characterisation.csv"):
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Group fairness characterisation")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    datasets = DATASET_CONFIGS if args.dataset == "all" else {args.dataset: DATASET_CONFIGS[args.dataset]}

    all_rows = []
    for name, config in datasets.items():
        rows = characterise(name, config)
        all_rows.extend(rows)

    save_csv(all_rows)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<20} {'Feature':<24} {'DPD':>8} {'EOD':>8}")
    print(f"  {'-'*20} {'-'*24} {'-'*8} {'-'*8}")
    for r in all_rows:
        dpd = f"{r['dpd']:.4f}" if r['dpd'] is not None else "  n/a"
        eod = f"{r['eod']:.4f}" if r['eod'] is not None else "  n/a"
        print(f"  {r['dataset']:<20} {r['sensitive_feature']:<24} {dpd:>8} {eod:>8}")


if __name__ == "__main__":
    main()
