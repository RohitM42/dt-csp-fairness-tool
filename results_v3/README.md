# results_v3

**Status: Superseded — retained for reference only. Do not use for report comparisons.**

## What was tested

The 3 datasets not covered in `results_v1/` or `results_v2/`:
- Credit (all methods: DT, CSP, Hybrid, baseline)
- Communities & Crime (DT + baseline only)
- Law School (DT + baseline only)

20 runs per method, budget=1000. Baseline uses corrected all-sensitive-feature flip.

## Notes

- CSP and Hybrid were only run for Credit in this batch.
- Communities & Crime and Law School have DT + baseline only — CSP and Hybrid results
  for these datasets are in `results_full/`.

## Superseded by

`results_full/` — all 8 datasets, all methods, corrected all-sensitive-feature baseline.
