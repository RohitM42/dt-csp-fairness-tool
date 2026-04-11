"""
csp.py — CP-SAT phase for fairness testing (Phase 2).

Uses Google OR-Tools CP-SAT to systematically and diversely generate
inputs within constrained feature regions.
Can be run standalone as a CSP-only method (for ablation comparison).
"""


def run_csp(config, budget, model, X, rules=None):
    """
    CSP-only method: CP-SAT generation with optional rules.
    If rules=None, runs with no DT constraints (pure CSP, random regions).
    Used standalone for ablation comparison against hybrid.

    Args:
        config: dataset config dict from config.py
        budget: total number of unique inputs to generate (S)
        model: loaded Keras model
        X: dataset as pandas DataFrame
        rules: list of rule dicts from dt.extract_rules(), or None

    Returns:
        dict with keys:
            - idi_ratio: float (I / S)
            - idi_count: int
            - inputs_generated: int
    """
    # TODO: initialise CP-SAT model (cp_model.CpModel)
    # TODO: define IntVar per non-sensitive feature with domain [min, max]
    # TODO: if rules provided, add constraints encoding DT rules
    # TODO: add inter-feature constraints (dataset-specific, see config)
    # TODO: loop: solve -> generate input pair -> query DNN -> record IDI
    # TODO: after each IDI found: add diversity constraint
    # TODO: continue until budget exhausted
    raise NotImplementedError


def build_cp_model(X, config, rules=None):
    """
    Build and return a CP-SAT model with feature domains and optional rule constraints.

    Args:
        X: dataset as pandas DataFrame (used to derive feature domains)
        config: dataset config dict
        rules: optional list of rule dicts from extract_rules()

    Returns:
        (cp_model.CpModel, dict of feature -> IntVar)
    """
    # TODO: create model, define IntVars, add rule constraints
    raise NotImplementedError


def add_diversity_constraint(cp_model, feature_vars, known_discriminatory, k):
    """
    Add constraint: new input must differ from each known discriminatory
    input by at least k features.
    """
    # TODO: implement diversity constraint
    raise NotImplementedError
