"""
baseline.py — Random search baseline for fairness testing.

Generates random inputs, flips the primary sensitive feature,
queries the DNN, and records discriminatory instances.

Supports two generation modes (controlled by PERTURBATION_MODE):
  "none"   — sample real data rows, flip sensitive feature only
  "random" — sample real row for sensitive feature, randomise all
             non-sensitive features uniformly within [min, max]
Note: "none" is the correct mode. "random" collapses on large search spaces
(e.g. KDD: IDI 0.283 -> 0.001) because model discrimination is concentrated
on the real data distribution, not spread across the full feature space.
"""

import numpy as np


DISCRIMINATION_THRESHOLD = 0.05

# "none"   : no perturbation, use real data rows as-is
# "random" : full random sampling within [min, max] per non-sensitive feature
PERTURBATION_MODE = "none"


def run_baseline(config, budget, model, X):
    """
    Run random search baseline.

    Args:
        config:  dataset config dict from config.py
        budget:  total number of inputs to generate (S)
        model:   loaded Keras model
        X:       full dataset as pandas DataFrame (target column already dropped)

    Returns:
        dict with keys:
            idi_ratio        — float (I / S)
            idi_count        — int
            inputs_generated — int
    """
    sensitive = config["sensitive"]
    primary_sensitive = sensitive[0]
    non_sensitive = [c for c in X.columns if c not in sensitive]

    sensitive_unique_vals = {col: X[col].unique() for col in sensitive}
    domains = {col: (X[col].min(), X[col].max()) for col in non_sensitive}

    seen_discriminatory = set()

    for _ in range(budget):
        sample_a, sample_b = _generate_pair(
            X, primary_sensitive, non_sensitive,
            sensitive_unique_vals, domains
        )
        if sample_a is None:
            continue

        pred_a = model.predict(sample_a.reshape(1, -1), verbose=0)[0][0]
        pred_b = model.predict(sample_b.reshape(1, -1), verbose=0)[0][0]

        if abs(pred_a - pred_b) > DISCRIMINATION_THRESHOLD:
            seen_discriminatory.add(tuple(sample_a))

    idi_count = len(seen_discriminatory)
    return {
        "idi_ratio": idi_count / budget,
        "idi_count": idi_count,
        "inputs_generated": budget,
    }


def _generate_pair(X, primary_sensitive, non_sensitive, sensitive_unique_vals, domains):
    """
    Generate a pair of inputs differing only on the primary sensitive feature.

    Mode "none":   takes a real data row, flips sensitive feature.
    Mode "random": takes a real data row for sensitive feature value,
                   randomises all non-sensitive features within [min, max].
    """
    row = X.iloc[np.random.randint(len(X))].copy().astype(float)

    current_val = row[primary_sensitive]
    other_vals = [
        v for v in sensitive_unique_vals[primary_sensitive] if v != current_val
    ]
    if not other_vals:
        return None, None

    if PERTURBATION_MODE == "random":
        for col in non_sensitive:
            mn, mx = domains[col]
            row[col] = np.random.uniform(mn, mx) if mn != mx else mn

    sample_a = row.values.copy()
    sample_b = sample_a.copy()
    col_idx = list(X.columns).index(primary_sensitive)
    sample_b[col_idx] = float(np.random.choice(other_vals))

    return sample_a, sample_b
