"""
test_baseline.py — Verify baseline random search across datasets.

Run from repo root:
    python tests/test_baseline.py kdd
    python tests/test_baseline.py all
"""

import sys
import os
import pandas as pd
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import DATASET_CONFIGS, DEFAULT_BUDGET
from src.baseline import run_baseline


def test_dataset(name):
    config = DATASET_CONFIGS[name]

    df = pd.read_csv(config["data_path"])
    X = df.drop(columns=[config["target"]])

    model = keras.models.load_model(config["model_path"])

    from src.baseline import PERTURBATION_MODE
    print(f"\n[{name.upper()}] Running baseline (budget={DEFAULT_BUDGET}, mode={PERTURBATION_MODE})...")
    result = run_baseline(config, DEFAULT_BUDGET, model, X)
    print(f"  IDI ratio:  {result['idi_ratio']:.4f}")
    print(f"  IDI count:  {result['idi_count']}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "kdd"

    if target == "all":
        for name in ["kdd", "adult", "compas", "german", "dutch"]:
            test_dataset(name)
    else:
        test_dataset(target)
