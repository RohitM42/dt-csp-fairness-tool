# Manual

## Overview

`csp-fairness-tool` finds Individual Discriminatory Instances (IDIs) in pre-trained neural network models using a CSP + Decision Tree hybrid approach.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure `dataset/` and `DNN/` folders contain the required files (see below)

## Required Files

**Datasets** (`dataset/`):
- `processed_kdd.csv`
- `processed_adult.csv`
- `processed_compas.csv`
- `processed_german.csv`
- `processed_dutch.csv`

**Models** (`DNN/`):
- `model_processed_kdd_cleaned.h5`
- `model_processed_adult.h5`
- `model_processed_compas.h5`
- `model_processed_greman_cleaned.h5`
- `model_processed_dutch.h5`

## Running the Tool

```bash
# Run on a single dataset
python main.py --dataset kdd --budget 1000 --runs 20

# Available datasets: kdd, adult, compas, german, dutch
```

## Output

Results are saved to `results/` as CSV files, one per dataset:
- `results/<dataset>_baseline.csv` — IDI ratios per run for random search
- `results/<dataset>_hybrid.csv` — IDI ratios per run for CSP+DT hybrid

## Configuration

Key parameters in `src/config.py`:
- `DEFAULT_BUDGET` — total inputs generated per run (default: 1000)
- `DEFAULT_SEED_RATIO` — fraction used for Phase 1 seed sampling (default: 0.15)
- `DT_MAX_DEPTH` — decision tree depth (default: 4)
- `DIVERSITY_K` — minimum feature differences for diversity constraint (default: 1)

---

> Export this file to `manual.pdf` and place at repo root before submission.
