# Requirements

## Python Version

Python **3.12.10** (tested). Python 3.9 or higher should be compatible.

## Dependencies

Exact versions used to produce the reported results:

| Package | Tested version | Minimum | Purpose |
|---------|---------------|---------|---------|
| tensorflow | 2.21.0 | >=2.10 | Loading and querying pre-trained DNN models (.h5) |
| keras | 3.14.0 | >=3.0 | Bundled with TensorFlow; model loading via `keras.models.load_model` |
| ortools | 9.15.6755 | >=9.0 | CP-SAT constraint solver (CSP and Hybrid Phase 2) |
| scikit-learn | 1.8.0 | >=1.0 | DecisionTreeClassifier (DT and Hybrid Phase 1) |
| pandas | 3.0.2 | >=1.5 | Dataset loading and manipulation |
| numpy | 2.4.4 | >=1.23 | Numerical operations |
| scipy | 1.17.1 | >=1.9 | Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) |
| matplotlib | 3.10.8 | >=3.6 | Optional — results visualisation |
| pytest | 9.0.3 | >=7.0 | Test suite (135 tests) |

## Installation

```bash
pip install -r requirements.txt
```

All packages are available on PyPI. No conda or special channels required.

## Hardware

No GPU required. All experiments run on CPU only. Approximate wall-clock runtime per dataset for a 20-run DT experiment (budget=1000, standard laptop, Intel CPU):

| Dataset | Baseline (20 runs) | DT (20 runs) | Total (both) |
|---------|--------------------|-------------|--------------|
| adult   | ~44 min | ~43 min | ~87 min |
| compas  | ~42 min | ~42 min | ~84 min |
| dutch   | ~39 min | ~39 min | ~78 min |
| german  | ~47 min | ~40 min | ~87 min |
| kdd     | ~40 min | ~44 min | ~84 min |

CSP and Hybrid methods are slower due to CP-SAT solver overhead (~197s/run observed on German). Running `--method all` across all 5 datasets will take several hours and is best left as an overnight run.

---

> Export this file to `requirements.pdf` and place at repo root before submission.
