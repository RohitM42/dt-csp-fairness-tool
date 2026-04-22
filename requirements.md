# Requirements

## Python Version

Python 3.9 or higher recommended.

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| tensorflow | >=2.10 | Loading and querying pre-trained DNN models (.h5) |
| ortools | >=9.0 | CP-SAT constraint solver (Phase 2) |
| scikit-learn | >=1.0 | DecisionTreeClassifier (Phase 1) |
| pandas | >=1.5 | Data loading and manipulation |
| numpy | >=1.23 | Numerical operations |
| scipy | >=1.9 | Wilcoxon signed-rank test |
| matplotlib | >=3.6 | Results figures |
| pytest | >=7.0 | Test suite |

## Installation

```bash
pip install -r requirements.txt
```

## Hardware

No GPU required. All experiments run on CPU. Estimated runtime per dataset: TODO (fill after experiments).

---

> Export this file to `requirements.pdf` and place at repo root before submission.
