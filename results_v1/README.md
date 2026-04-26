# results_v1

**Status: Superseded — retained for reference only. Do not use for report comparisons.**

## What was tested

All three methods (DT, CSP, Hybrid) vs baseline across 5 datasets:
- KDD, Adult, COMPAS, Dutch, German

20 runs per method, budget=1000.

## Key limitation

The baseline in this run flips **only the primary sensitive feature** (`sensitive[0]`) when
generating a comparison pair. This is inconsistent with the reference implementation
(`lab4_solution.py`) and with all three comparison methods, which flip all sensitive features
simultaneously. This makes method-vs-baseline comparisons in this folder unreliable.

## Superseded by

`results_full/` — all 8 datasets, all methods, corrected all-sensitive-feature baseline.
