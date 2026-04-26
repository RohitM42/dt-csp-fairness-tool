# results_full

**Status: Authoritative — all figures in the report are sourced from this folder.**

## What was tested

All methods (DT, CSP, Hybrid) vs baseline across all 8 datasets:

Primary (6): KDD, Adult, COMPAS, Dutch, Credit, German

Ablation (2): Law School, Communities & Crime

20 runs per method, budget=1000.

## Baseline methodology

The baseline flips **all sensitive features simultaneously** when generating a comparison pair,
consistent with the reference implementation (`lab4_solution.py`) and all three comparison
methods. This corrects the single-feature flip used in `results_v1/`.

## Key output

- `summary.csv` — Wilcoxon p-values, IDI means, std, and timing for all method-dataset pairs.
  This is the primary source for all tables and statistics in the report.

## Prerequisites

- Python 3.9+, all dependencies installed (`pip install -r requirements.txt`)
- Dataset CSVs in `dataset/` and model `.h5` files in `DNN/` (see `manual.pdf`)
- Reproduce by running: `python main.py --dataset all --method all --budget 1000 --runs 20`
