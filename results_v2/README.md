# results_v2

**Status: Superseded — retained for reference only. Do not use for report comparisons.**

## What was tested

DT-only vs baseline across 5 datasets:
- KDD, Adult, COMPAS, Dutch, German

20 runs per method, budget=1000.

## Notes

- The baseline here correctly flips **all sensitive features simultaneously**, fixing the
  inconsistency in `results_v1/`.
- Only DT was run in this batch — CSP and Hybrid results are absent.
- Coverage is limited to the original 5 datasets; Credit, Law School, and Communities & Crime
  were not tested here.

## Superseded by

`results_full/` — all 8 datasets, all methods, corrected all-sensitive-feature baseline.
