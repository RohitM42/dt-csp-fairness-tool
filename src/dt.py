"""
dt.py — Decision Tree phase for fairness testing (Phase 1).

Trains a DecisionTreeClassifier on random seed samples to identify
which feature regions are likely to contain discrimination.
Can be run standalone as a DT-only method (for ablation comparison).
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree


def _flip_sensitive(sample, X, sensitive_cols):
    """Return a copy of sample with each sensitive feature flipped to a different value."""
    flipped = sample.copy()
    for col in sensitive_cols:
        unique_vals = X[col].unique()
        if len(unique_vals) == 2:
            flipped[col] = 1 - int(sample[col])
        else:
            flipped[col] = np.random.choice([v for v in unique_vals if v != sample[col]])
    return flipped


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
    sensitive_cols = config["sensitive"]
    seed_n = max(1, int(seed_ratio * budget))
    remaining_n = budget - seed_n
    feature_names = X.columns.tolist()

    # --- random seed sampling ---
    seed_inputs, seed_labels = [], []
    seen_discriminatory = set()

    for _ in range(seed_n):
        sample_a = X.iloc[np.random.randint(len(X))].copy()
        sample_b = _flip_sensitive(sample_a, X, sensitive_cols)

        pred_a = model.predict(sample_a.values.reshape(1, -1), verbose=0)[0][0]
        pred_b = model.predict(sample_b.values.reshape(1, -1), verbose=0)[0][0]
        label = int(abs(pred_a - pred_b) > 0.05)

        seed_inputs.append(sample_a.values)
        seed_labels.append(label)
        if label:
            seen_discriminatory.add(tuple(sample_a.values))

    seed_inputs = np.array(seed_inputs)
    seed_labels = np.array(seed_labels)

    dt = train_dt(seed_inputs, seed_labels, feature_names)
    rules = extract_rules(dt, feature_names)

    # --- DT-guided sampling for remaining budget ---
    # Oversample candidates, filter to DT-predicted positive regions to bias search
    remaining = remaining_n
    while remaining > 0:
        n_candidates = remaining * 5
        candidates = X.sample(n=n_candidates, replace=(n_candidates > len(X)))
        preds = dt.predict(candidates.values)
        positive_mask = preds == 1
        pool = candidates[positive_mask] if positive_mask.any() else candidates

        for _, row in pool.iterrows():
            if remaining <= 0:
                break
            sample_a = row.copy()
            sample_b = _flip_sensitive(sample_a, X, sensitive_cols)

            pred_a = model.predict(sample_a.values.reshape(1, -1), verbose=0)[0][0]
            pred_b = model.predict(sample_b.values.reshape(1, -1), verbose=0)[0][0]
            if abs(pred_a - pred_b) > 0.05:
                seen_discriminatory.add(tuple(sample_a.values))
            remaining -= 1

    idi_count = len(seen_discriminatory)
    return {
        "idi_ratio": idi_count / budget if budget > 0 else 0.0,
        "idi_count": idi_count,
        "inputs_generated": budget,
        "dt_rules": rules,
    }


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
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(seed_inputs, seed_labels)
    return dt


def extract_rules(tree, feature_names):
    """
    Extract decision rules from a fitted DecisionTreeClassifier.

    Returns:
        list of rule dicts, each with feature conditions leading to
        a positive (discriminatory) leaf. Each rule has:
            - conditions: list of (feature_name, operator, threshold) tuples
            - n_samples: number of training samples reaching this leaf
            - n_positive: number of discriminatory samples at this leaf
    """
    tree_ = tree.tree_
    classes = tree.classes_
    # index of class 1 in classes_ (may be absent if training data was all one class)
    pos_idx = next((i for i, c in enumerate(classes) if c == 1), None)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else None
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            # leaf — map argmax index through classes_ to get the actual predicted class
            value = tree_.value[node][0]
            predicted_class = classes[int(np.argmax(value))]
            if predicted_class == 1:
                rules.append({
                    "conditions": list(path),
                    "n_samples": int(tree_.n_node_samples[node]),
                    "n_positive": int(value[pos_idx]) if pos_idx is not None else 0,
                })
            return
        feat = feature_name[node]
        threshold = float(tree_.threshold[node])
        recurse(tree_.children_left[node], path + [(feat, "<=", threshold)])
        recurse(tree_.children_right[node], path + [(feat, ">", threshold)])

    recurse(0, [])
    return rules