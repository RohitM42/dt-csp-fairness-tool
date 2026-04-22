"""
hybrid.py — CSP + Decision Tree hybrid (Phase 1 + Phase 2 combined).

Orchestrates dt.py and csp.py:
  Phase 1: random seed sampling, DT training, and rule extraction (dt.py)
  Phase 2: CP-SAT generation constrained by DT rules (csp.py)
"""

import numpy as np

from src.dt import _flip_sensitive, train_dt, extract_rules
from src.csp import run_csp


def run_hybrid(config, budget, model, X, seed_ratio=0.15):
    """
    Run CSP + DT hybrid.

    Phase 1 spends seed_ratio * budget on random seed sampling to label
    inputs as discriminatory or not, then trains a DecisionTreeClassifier
    to identify high-discrimination feature regions.

    Phase 2 uses CP-SAT (run_csp) constrained to those DT regions to
    generate the remaining budget inputs more efficiently.

    Args:
        config: dataset config dict from config.py
        budget: total number of unique inputs to generate (S)
        model: loaded Keras model
        X: dataset as pandas DataFrame
        seed_ratio: fraction of budget used for Phase 1 seed sampling

    Returns:
        dict with keys:
            - idi_ratio: float (total IDIs / total inputs generated)
            - idi_count: int (Phase 1 + Phase 2 combined)
            - inputs_generated: int (Phase 1 + Phase 2 combined)
            - dt_rules: list of extracted decision tree rules (for reporting)
    """
    if budget <= 0:
        return {"idi_ratio": 0.0, "idi_count": 0, "inputs_generated": 0, "dt_rules": []}

    sensitive_cols = config["sensitive"]
    feature_names = X.columns.tolist()
    seed_n = max(1, int(seed_ratio * budget))
    remaining_n = budget - seed_n

    # --- Phase 1: random seed sampling ---
    seed_inputs, seed_labels = [], []
    phase1_idi_count = 0

    for _ in range(seed_n):
        sample_a = X.iloc[np.random.randint(len(X))].copy()
        sample_b = _flip_sensitive(sample_a, X, sensitive_cols)

        pred_a = model.predict(sample_a.values.reshape(1, -1), verbose=0)[0][0]
        pred_b = model.predict(sample_b.values.reshape(1, -1), verbose=0)[0][0]
        label = int(abs(pred_a - pred_b) > 0.05)

        seed_inputs.append(sample_a.values)
        seed_labels.append(label)
        phase1_idi_count += label

    seed_inputs = np.array(seed_inputs)
    seed_labels = np.array(seed_labels)

    dt = train_dt(seed_inputs, seed_labels, feature_names)
    rules = extract_rules(dt, feature_names)

    # --- Phase 2: CSP constrained to DT-identified regions ---
    csp_result = run_csp(config, remaining_n, model, X, rules=rules)

    total_idi = phase1_idi_count + csp_result["idi_count"]
    total_inputs = seed_n + csp_result["inputs_generated"]

    return {
        "idi_ratio": total_idi / total_inputs if total_inputs > 0 else 0.0,
        "idi_count": total_idi,
        "inputs_generated": total_inputs,
        "dt_rules": rules,
    }
