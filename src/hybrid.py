"""
hybrid.py — CSP + Decision Tree hybrid (Phase 1 + Phase 2 combined).

Orchestrates dt.py and csp.py:
  Phase 1: DT training and rule extraction (dt.py)
  Phase 2: CP-SAT generation constrained by DT rules (csp.py)
"""

from src.dt import train_dt, extract_rules
from src.csp import run_csp


def run_hybrid(config, budget, model, X, seed_ratio=0.15):
    """
    Run CSP + DT hybrid.

    Args:
        config: dataset config dict from config.py
        budget: total number of unique inputs to generate (S)
        model: loaded Keras model
        X: dataset as pandas DataFrame
        seed_ratio: fraction of budget used for Phase 1 seed sampling

    Returns:
        dict with keys:
            - idi_ratio: float (I / S)
            - idi_count: int
            - inputs_generated: int
            - dt_rules: list of extracted decision tree rules (for reporting)
    """
    # --- Phase 1 ---
    # TODO: spend seed_ratio * budget on random sampling (baseline logic)
    # TODO: label each input
    # TODO: call train_dt() -> fitted tree
    # TODO: call extract_rules() -> rules list

    # --- Phase 2 ---
    # TODO: call run_csp(config, remaining_budget, model, X, rules=rules)
    # TODO: combine Phase 1 + Phase 2 IDI counts for final ratio

    raise NotImplementedError
