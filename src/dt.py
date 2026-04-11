"""
dt.py — Decision Tree phase for fairness testing (Phase 1).

Trains a DecisionTreeClassifier on random seed samples to identify
which feature regions are likely to contain discrimination.
Can be run standalone as a DT-only method (for ablation comparison).
"""


def run_dt(config, budget, model, X, seed_ratio=0.15):
    """
    DT-only method: random seed sampling + DT-guided generation.
    Used standalone for ablation comparison against hybrid.

    Args:
        config: dataset config dict from config.py
        budget: total number of unique inputs to generate (S)
        model: loaded Keras model
        X: dataset as pandas DataFrame
        seed_ratio: fraction of budget used for random seed sampling

    Returns:
        dict with keys:
            - idi_ratio: float (I / S)
            - idi_count: int
            - inputs_generated: int
            - dt_rules: list of extracted decision tree rules
    """
    # TODO: spend seed_ratio * budget on random sampling (same logic as baseline)
    # TODO: label each input (1 = discriminatory, 0 = not)
    # TODO: train DecisionTreeClassifier(max_depth=4) on labelled results
    # TODO: use DT-predicted high-discrimination regions to bias remaining sampling
    # TODO: (no CP-SAT — just DT-guided random sampling for remaining budget)
    raise NotImplementedError


def train_dt(seed_inputs, seed_labels, feature_names, max_depth=4):
    """
    Train a DecisionTreeClassifier on seed samples.

    Args:
        seed_inputs: array of input vectors
        seed_labels: array of 0/1 labels (1 = discriminatory)
        feature_names: list of feature names
        max_depth: max depth of decision tree

    Returns:
        fitted DecisionTreeClassifier
    """
    # TODO: fit DecisionTreeClassifier
    raise NotImplementedError


def extract_rules(tree, feature_names):
    """
    Extract decision rules from a fitted DecisionTreeClassifier.

    Returns:
        list of rule dicts, each with feature conditions leading to
        a positive (discriminatory) leaf
    """
    # TODO: walk tree structure, extract conditions on path to positive leaves
    raise NotImplementedError
