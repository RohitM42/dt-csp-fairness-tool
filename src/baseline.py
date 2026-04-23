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
    non_sensitive = [c for c in X.columns if c not in sensitive]

    sensitive_unique_vals = {col: X[col].unique() for col in sensitive}
    domains = {col: (X[col].min(), X[col].max()) for col in non_sensitive}

    seen_discriminatory = set()

    for _ in range(budget):
        sample_a, sample_b = _generate_pair(
            X, sensitive, non_sensitive,
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


def _generate_pair(X, sensitive, non_sensitive, sensitive_unique_vals, domains):
    """
    Generate a pair of inputs differing only on the sensitive features.
    All sensitive features are flipped simultaneously, matching the approach
    used by DT, CSP, and Hybrid, and consistent with lab4_solution.py.

    Mode "none":   takes a real data row, flips all sensitive features.
    Mode "random": takes a real data row, randomises all non-sensitive
                   features within [min, max], then flips all sensitive features.
    """
    row = X.iloc[np.random.randint(len(X))].copy().astype(float)

    if PERTURBATION_MODE == "random":
        for col in non_sensitive:
            mn, mx = domains[col]
            row[col] = np.random.uniform(mn, mx) if mn != mx else mn

    sample_a = row.values.copy()
    sample_b = sample_a.copy()
    col_indices = list(X.columns)

    any_flipped = False
    for col in sensitive:
        current_val = row[col]
        other_vals = [v for v in sensitive_unique_vals[col] if v != current_val]
        if not other_vals:
            continue
        idx = col_indices.index(col)
        sample_b[idx] = float(np.random.choice(other_vals))
        any_flipped = True

    if not any_flipped:
        return None, None

    return sample_a, sample_b
