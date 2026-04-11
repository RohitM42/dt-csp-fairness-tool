"""
baseline.py — Random search baseline for fairness testing.

Replicates the random search logic from lab4_solution.py.
Generates random inputs, flips sensitive feature, queries DNN,
records discriminatory instances.
"""


def run_baseline(config, budget, model, X):
    """
    Run random search baseline.

    Args:
        config: dataset config dict from config.py
        budget: total number of unique inputs to generate (S)
        model: loaded Keras model
        X: dataset as pandas DataFrame

    Returns:
        dict with keys:
            - idi_ratio: float (I / S)
            - idi_count: int
            - inputs_generated: int
    """
    # TODO: implement random sampling loop
    # TODO: for each sample, flip sensitive[0] (binary) or random other value (multi)
    # TODO: query model on both, check if predictions differ
    # TODO: track unique discriminatory inputs
    raise NotImplementedError
