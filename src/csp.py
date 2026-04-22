"""
csp.py — CP-SAT phase for fairness testing (Phase 2).

Uses Google OR-Tools CP-SAT to systematically and diversely generate
inputs within constrained feature regions.
Can be run standalone as a CSP-only method (for ablation comparison).
"""

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# Scaling factor applied to continuous (non-integer) feature values so they
# fit inside CP-SAT IntVar domains with 3 decimal places of precision.
SCALE = 1000


def _feature_scale(series):
    """Return 1 for integer-valued columns, SCALE for continuous ones."""
    vals = series.dropna()
    if len(vals) > 0 and (vals == vals.round()).all():
        return 1
    return SCALE


def _flip_col(val, unique_vals):
    """Return a different value for a single sensitive feature."""
    if len(unique_vals) == 2:
        return float(1 - int(val))
    return float(np.random.choice([v for v in unique_vals if v != val]))


def run_csp(config, budget, model, X, rules=None):
    """
    CSP-only method: CP-SAT generation with optional DT rule constraints.
    If rules=None or [], runs unconstrained (pure CSP, any feasible region).
    Used standalone for ablation comparison against the hybrid.

    Args:
        config: dataset config dict from config.py
        budget: total number of unique inputs to generate (S)
        model: loaded Keras model
        X: dataset as pandas DataFrame
        rules: list of rule dicts from dt.extract_rules(), or None

    Returns:
        dict with keys:
            - idi_ratio: float (I / inputs_generated)
            - idi_count: int
            - inputs_generated: int  (may be < budget if solver exhausts the space)
    """
    if budget <= 0:
        return {"idi_ratio": 0.0, "idi_count": 0, "inputs_generated": 0}

    sensitive_cols = config["sensitive"]
    non_sensitive_cols = [c for c in X.columns if c not in sensitive_cols]

    cp_m, feature_vars, feat_scales = build_cp_model(X, config, rules)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1.0

    idi_count = 0
    inputs_generated = 0
    known_discriminatory = []

    while inputs_generated < budget:
        # Randomise objective to encourage the solver to explore different regions
        if non_sensitive_cols:
            coeffs = np.random.randint(-10, 11, len(non_sensitive_cols))
            terms = [int(coeffs[i]) * feature_vars[col]
                     for i, col in enumerate(non_sensitive_cols)]
            cp_m.Minimize(cp_model.LinearExpr.Sum(terms))

        status = solver.Solve(cp_m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break  # model infeasible or solver timed out

        # Decode solution: non-sensitive features from solver, sensitive from dataset
        row_a = pd.Series(index=X.columns, dtype=float)
        for col in non_sensitive_cols:
            row_a[col] = solver.Value(feature_vars[col]) / feat_scales[col]
        for col in sensitive_cols:
            row_a[col] = float(X[col].iloc[np.random.randint(len(X))])

        # Build comparison pair by flipping each sensitive feature
        row_b = row_a.copy()
        for col in sensitive_cols:
            row_b[col] = _flip_col(row_a[col], X[col].unique())

        pred_a = model.predict(row_a.values.reshape(1, -1), verbose=0)[0][0]
        pred_b = model.predict(row_b.values.reshape(1, -1), verbose=0)[0][0]
        is_idi = int(abs(pred_a - pred_b) > 0.05)
        idi_count += is_idi
        inputs_generated += 1

        if is_idi:
            d_scaled = {col: solver.Value(feature_vars[col]) for col in non_sensitive_cols}
            known_discriminatory.append(d_scaled)
            add_diversity_constraint(cp_m, feature_vars, known_discriminatory, k=1)

    return {
        "idi_ratio": idi_count / inputs_generated if inputs_generated > 0 else 0.0,
        "idi_count": idi_count,
        "inputs_generated": inputs_generated,
    }


def build_cp_model(X, config, rules=None):
    """
    Build and return a CP-SAT model with feature domains and optional rule constraints.

    Non-sensitive features become IntVars with domain [min*scale, max*scale].
    DT rules (if provided) are encoded as a disjunction: the solution must
    satisfy at least one rule's conditions.

    Args:
        X: dataset as pandas DataFrame (used to derive feature domains)
        config: dataset config dict
        rules: optional list of rule dicts from extract_rules()

    Returns:
        (CpModel, dict[feature_name -> IntVar], dict[feature_name -> int scale])
    """
    sensitive_cols = config["sensitive"]
    non_sensitive_cols = [c for c in X.columns if c not in sensitive_cols]

    m = cp_model.CpModel()
    feature_vars = {}
    feat_scales = {}

    for col in non_sensitive_cols:
        scale = _feature_scale(X[col])
        lo = int(X[col].min() * scale)
        hi = int(X[col].max() * scale)
        if lo == hi:
            hi = lo + 1  # prevent degenerate single-point domain
        feature_vars[col] = m.NewIntVar(lo, hi, col)
        feat_scales[col] = scale

    # Encode rules as a disjunction: at least one rule's region must be satisfied
    if rules:
        rule_bools = []
        for i, rule in enumerate(rules):
            b_rule = m.NewBoolVar(f"rule_{i}")
            rule_bools.append(b_rule)
            for feat, op, thresh in rule["conditions"]:
                if feat not in feature_vars:
                    continue  # skip sensitive features and unrecognised columns
                var = feature_vars[feat]
                scale = feat_scales[feat]
                thresh_scaled = round(thresh * scale)
                if op == "<=":
                    m.Add(var <= thresh_scaled).OnlyEnforceIf(b_rule)
                else:  # ">"
                    m.Add(var >= thresh_scaled + 1).OnlyEnforceIf(b_rule)
        if rule_bools:
            m.AddBoolOr(rule_bools)

    return m, feature_vars, feat_scales


def add_diversity_constraint(model, feature_vars, known_discriminatory, k):
    """
    Add a constraint ensuring the next solution differs from the most recently
    found discriminatory input by at least k features.

    Each call adds one constraint (for the last entry in known_discriminatory).
    Accumulated over time, these prevent the solver from revisiting any known
    discriminatory region.

    Args:
        model: CpModel (modified in place)
        feature_vars: dict[feature_name -> IntVar]
        known_discriminatory: list of {feature_name: scaled_int_value} dicts
        k: minimum number of features that must differ (typically 1)
    """
    d = known_discriminatory[-1]
    counter = len(known_discriminatory)
    diff_bools = []

    for feat, d_val in d.items():
        if feat not in feature_vars:
            continue
        var = feature_vars[feat]
        b = model.NewBoolVar(f"diff_{feat}_{counter}")
        # b=False → var must equal d_val
        model.Add(var == d_val).OnlyEnforceIf(b.Not())
        # b=True  → var must differ from d_val
        model.Add(var != d_val).OnlyEnforceIf(b)
        diff_bools.append(b)

    if diff_bools:
        model.Add(sum(diff_bools) >= k)
