# csp-fairness-tool

A Decision Tree + CSP hybrid approach to AI fairness testing, developed as part of the
Intelligent Software Engineering (ISE) module.

## Overview

This tool finds **Individual Discriminatory Instances (IDIs)** in pre-trained neural network
models — pairs of inputs identical except for a sensitive feature (e.g., gender, race, age)
where the model predicts a meaningfully different outcome (threshold: > 0.05 difference).

Three methods are evaluated against a random-search baseline:

- **DT-only:** trains a DecisionTreeClassifier on randomly sampled pairs to identify
  high-discrimination feature regions, then biases further sampling toward those regions
- **CSP-only:** uses CP-SAT (Google OR-Tools) to generate diverse inputs across the feature
  domain with diversity constraints preventing revisiting known discriminatory regions
- **Hybrid:** combines DT rule extraction (Phase 1) with CSP generation constrained to
  those rules (Phase 2)

All methods — including the baseline — flip **all sensitive features simultaneously** when
creating a comparison pair, consistent with `lab4_solution.py`.

## Datasets

KDD, Adult, COMPAS, German, Dutch

## Usage

```bash
# Full experiment — all datasets, all methods, baseline shared per dataset
python main.py --dataset all --method all --budget 1000 --runs 20

# Save to a custom directory (to avoid overwriting a previous run)
python main.py --dataset all --method all --budget 1000 --runs 20 --results-dir results_v2

# Single dataset, all methods
python main.py --dataset kdd --method all

# Single dataset, single method
python main.py --dataset kdd --method dt
```

See `manual.md` for full usage, arguments, and configuration details.

## Replication

See `replication.md` for exact reproduction steps and preliminary results
(export to `replication.pdf` for submission).

## Tests

135 tests across all modules. No real datasets or model files required.

```bash
python -m pytest tests/ -v
```

| Module             | Tests |
|--------------------|-------|
| `test_dt.py`       | 33    |
| `test_csp.py`      | 47    |
| `test_hybrid.py`   | 24    |
| `test_evaluate.py` | 31    |

## Dependencies

See `requirements.txt` and `requirements.md` (export to `requirements.pdf` for submission).
