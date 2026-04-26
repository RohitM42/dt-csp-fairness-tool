# Manual

## Overview

`dt-csp-fairness-tool` finds Individual Discriminatory Instances (IDIs) in pre-trained neural
network models. An IDI is a pair of inputs identical except for a sensitive feature (e.g.,
gender, race, age) where the model's output differs by more than 0.05 — indicating the model
treats people differently based on a protected characteristic.

Three methods are implemented and compared against a random-search baseline. The baseline
samples real data rows and flips **all** sensitive features simultaneously (e.g. sex, race,
and age together), consistent with the reference implementation in `lab4_solution.py`. All
three methods use the same simultaneous flip definition so comparisons are like-for-like.

| Method  | Description |
|---------|-------------|
| `dt`    | **DT-only** — random seed sampling to label pairs, trains a DecisionTreeClassifier to identify high-discrimination feature regions, then samples more inputs from those regions |
| `csp`   | **CSP-only** — uses CP-SAT (Google OR-Tools) to generate inputs across the feature domain with diversity constraints to avoid revisiting known discriminatory regions |
| `hybrid`| **Hybrid** — Phase 1 (DT seed sampling + rule extraction) followed by Phase 2 (CSP constrained to DT-identified regions) |

**Primary finding:** DT is the strongest method, consistently and substantially beating the
baseline. CSP and Hybrid are included as ablation comparisons and provide valuable evaluation
content around the trade-offs of constraint-based vs data-driven search.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure `dataset/` and `DNN/` folders contain the required files (see below)

> **Windows note:** Commands below use `python`. On some Windows setups the Python Launcher
> is invoked as `py` instead (e.g. `py main.py ...`). Use whichever works in your environment.
> On Linux/macOS `python3` may be required depending on your PATH.

## Required Files

**Datasets** (`dataset/`):

Primary (required for main experiment):
- `processed_kdd.csv`
- `processed_adult.csv`
- `processed_compas.csv`
- `processed_german.csv`
- `processed_dutch.csv`
- `processed_credit_with_numerical.csv`

Ablation only (required for `--dataset all`):
- `processed_communities_crime.csv`
- `processed_law_school.csv`

**Models** (`DNN/`):

Primary:
- `model_processed_kdd_cleaned.h5`
- `model_processed_adult.h5`
- `model_processed_compas.h5`
- `model_processed_greman_cleaned.h5`  ← note: typo in source filename, keep as-is
- `model_processed_dutch.h5`
- `model_processed_credit.h5`

Ablation only:
- `model_processed_communities_crime.h5`
- `model_processed_law_school_cleaned.h5`

## Running the Tool

```bash
# Full experiment — all 8 datasets, all methods (recommended; expect several hours)
python main.py --dataset all --method all --budget 1000 --runs 20

# Full experiment saving to a separate folder (e.g. to preserve a previous run)
python main.py --dataset all --method all --budget 1000 --runs 20 --results-dir results_v2

# All three methods vs baseline on a single dataset (baseline shared, run once)
python main.py --dataset kdd --method all --budget 1000 --runs 20

# Multiple specific datasets (comma-separated, spaces optional)
python main.py --dataset kdd,adult,compas --method all --budget 1000 --runs 20

# Single method vs baseline
python main.py --dataset kdd --method dt     --budget 1000 --runs 20
python main.py --dataset kdd --method csp    --budget 1000 --runs 20
python main.py --dataset kdd --method hybrid --budget 1000 --runs 20

# Quick sanity check — 3 runs on one dataset to verify changes before a full run
python main.py --dataset adult --method all --runs 3 --results-dir results_test
```

> **Note:** `--dataset all` runs all 8 datasets (KDD, Adult, COMPAS, Dutch, Credit, German, Law School, Communities & Crime). Law School and Communities & Crime are ablation-only datasets; to run only the 6 primary datasets, either specify them individually or use a comma-separated list.

**Arguments:**

| Argument        | Default    | Description |
|-----------------|------------|-------------|
| `--dataset`     | required   | Single dataset name, comma-separated list (e.g. `kdd,adult,compas`), or `all` (all 8 datasets) |
| `--method`      | `hybrid`   | `dt`, `csp`, `hybrid`, or `all` |
| `--budget`      | `1000`     | Total inputs generated per run |
| `--runs`        | `20`       | Independent trials per method (min 20 for meaningful Wilcoxon p-values) |
| `--seed-ratio`  | `0.15`     | Fraction of budget used for Phase 1 seed sampling (DT and Hybrid only) |
| `--results-dir` | `results`  | Directory to write output CSVs — use a different path to avoid overwriting a previous run |

## Output

Results are saved incrementally — each run is written immediately so partial results survive
an interrupted experiment. Output goes to `--results-dir` (default: `results/`):

- `<results-dir>/<dataset>_baseline.csv` — per-run IDI ratios for random search
- `<results-dir>/<dataset>_dt.csv`       — per-run IDI ratios for DT-only
- `<results-dir>/<dataset>_csp.csv`      — per-run IDI ratios for CSP-only
- `<results-dir>/<dataset>_hybrid.csv`   — per-run IDI ratios for Hybrid
- `<results-dir>/summary.csv`            — Wilcoxon comparison summary (means, std, p-values,
                                           timing); updated after each dataset completes when
                                           running `--dataset all`

**IDI ratio** = unique discriminatory inputs found / total inputs generated. Higher is better —
a higher ratio means the method found more discrimination per unit of budget.

## Configuration

Key parameters in `src/config.py`:

| Parameter       | Default | Description |
|-----------------|---------|-------------|
| `DEFAULT_BUDGET`     | `1000` | Total inputs per run — imported by `main.py` as the `--budget` default |
| `DEFAULT_SEED_RATIO` | `0.15` | Phase 1 seed fraction — imported by `main.py` as the `--seed-ratio` default |
| `DT_MAX_DEPTH`       | `4`    | Reference value documenting the `max_depth` default in `dt.train_dt()` — not imported at runtime; change the default in `src/dt.py` directly to override |
| `DIVERSITY_K`        | `1`    | Reference value documenting the `k` default in `csp.add_diversity_constraint()` — not imported at runtime; change the argument in `src/csp.py` directly to override |

## Group Fairness Characterisation

A standalone script is provided to profile each model's group-level bias before running IDI
search. It computes two metrics per sensitive feature:

- **Demographic Parity Difference (DPD)** — max difference in positive prediction rate across
  sensitive attribute groups. Does not require ground truth labels.
- **Equalized Odds Difference (EOD)** — max of TPR difference and FPR difference across groups.
  Requires the ground truth target column (already present in all dataset CSVs).

```bash
# Characterise all configured datasets
python scripts/characterise.py

# Single dataset
python scripts/characterise.py --dataset adult
```

Results are printed as a table per dataset and saved to `characterisation.csv` at the repo root.
This script is intended as a one-time characterisation pass; re-running it produces the same
output (no randomness involved).

## Testing

The test suite covers all modules with 147 tests total. No real datasets or `.h5` model
files are required — tests use `MockModel` fixtures and synthetic `pandas` DataFrames.

```bash
# Run all tests
python -m pytest tests/ -v

# Run per module
python -m pytest tests/test_dt.py       -v   # 33 tests
python -m pytest tests/test_csp.py      -v   # 47 tests
python -m pytest tests/test_hybrid.py   -v   # 24 tests
python -m pytest tests/test_evaluate.py -v   # 31 tests
python -m pytest tests/test_baseline.py -v   # 12 tests
```

scipy and numpy RuntimeWarnings (e.g. Wilcoxon on all-zero differences) are suppressed
via `conftest.py` and do not indicate failures.

---
