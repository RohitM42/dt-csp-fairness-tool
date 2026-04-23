# Replication

Steps to exactly reproduce the results reported in the paper.

## Environment

- OS: Windows 11 Home (cross-platform compatible — Linux/macOS should work without changes)
- Python: 3.12.10
- Key dependencies: TensorFlow/Keras, OR-Tools CP-SAT, scikit-learn, scipy, pandas (see `requirements.txt`)
- Hardware: Standard laptop (Intel CPU); no GPU required — all model inference runs on CPU

## Random Seed

No fixed global seed is used. Each run independently samples from the dataset and the solver
uses random objectives each iteration. Results are averaged over `--runs` independent trials
(default 20) and compared with a Wilcoxon signed-rank test. Trial-to-trial variance is low
(see std columns in `results/summary.csv`).

## Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/csp-fairness-tool.git
   cd csp-fairness-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the required dataset CSVs in `dataset/` and model files in `DNN/`
   (see `manual.md` for exact filenames).

4. Run the full experiment — all datasets, all methods, baseline shared per dataset:
   ```bash
   python main.py --dataset all --method all --budget 1000 --runs 20
   ```

   This is the recommended single command. The baseline is run once per dataset and shared
   across all three method comparisons. Results are saved incrementally after each run and
   the summary CSV is updated after each dataset completes, so an interrupted run preserves
   all completed datasets.

   **To save to a custom directory (e.g. to preserve a previous run):**
   ```bash
   python main.py --dataset all --method all --budget 1000 --runs 20 --results-dir results_v2
   ```

   **Alternatively, run one dataset at a time:**
   ```bash
   python main.py --dataset kdd    --method all --budget 1000 --runs 20
   python main.py --dataset adult  --method all --budget 1000 --runs 20
   python main.py --dataset compas --method all --budget 1000 --runs 20
   python main.py --dataset german --method all --budget 1000 --runs 20
   python main.py --dataset dutch  --method all --budget 1000 --runs 20
   ```

   **For targeted re-runs of a single method:**
   ```bash
   python main.py --dataset kdd --method dt --budget 1000 --runs 20
   ```

5. Results are written to `results/` (or `--results-dir` if specified):
   - `<results-dir>/<dataset>_baseline.csv` — per-run IDI ratios for random search
   - `<results-dir>/<dataset>_dt.csv`       — per-run IDI ratios for DT-only
   - `<results-dir>/<dataset>_csp.csv`      — per-run IDI ratios for CSP-only
   - `<results-dir>/<dataset>_hybrid.csv`   — per-run IDI ratios for Hybrid
   - `<results-dir>/summary.csv`            — Wilcoxon p-values, IDI means, and timing per method
                                              (updated after each dataset completes)

## IDI Counting — Methodology Note

All methods count **unique discriminatory inputs** only. A discriminatory pair `(a, b)` is
counted once regardless of how many times input `a` is sampled. This is enforced via a
`seen_discriminatory` set in the baseline and DT. CSP uses CP-SAT diversity constraints to
avoid revisiting discriminatory regions rather than a deduplication set.

## Sensitive Feature Flip — Methodology Note

All methods (baseline, DT, CSP, Hybrid) flip **all sensitive features simultaneously** when
creating a comparison pair. For example, on KDD (sensitive: sex, race, age), both sex, race,
and age are changed in `sample_b`. This is consistent with the reference implementation in
`lab4_solution.py` and tests for intersectional discrimination across all protected attributes
at once. The baseline previously flipped only the primary sensitive feature (sensitive[0]);
this was corrected to match the reference and the three comparison methods.

## Expected Results

Full 20-run results (budget=1000) from `results_v2/` — all-sensitive-feature flip baseline.
p-values from Wilcoxon signed-rank test (paired, two-sided).

> **Note:** v1 results (in `results/`) used a single-attribute baseline flip (primary
> sensitive only). v2 results below use all-sensitive flip for baseline, consistent with
> `lab4_solution.py` and the three comparison methods. Baseline IDI ratios are higher in
> v2 as a result. DT/CSP/Hybrid values are unchanged.

| Dataset | Baseline (mean) | DT (mean) | CSP (mean) | Hybrid (mean) |
|---------|-----------------|-----------|------------|---------------|
| KDD     | pending         | 0.676     | 0.041      | 0.124         |
| Adult   | pending         | 0.560     | 0.212      | 0.268         |
| COMPAS  | pending         | 0.661     | 0.427      | 0.455         |
| German  | pending         | 0.363     | 0.235      | 0.288         |
| Dutch   | pending         | 0.151     | 0.029      | 0.085         |

DT/CSP/Hybrid values are from v1 runs and are unaffected by the baseline fix.
Baseline values are pending the v2 run (`--results-dir results_v2`).

## Key Findings (Preliminary)

- **DT consistently and substantially beats the baseline** across all tested datasets,
  confirming DT-guided region focusing is an effective fairness-testing strategy.
- **CSP beats baseline on German** (0.230 vs 0.027) but underperforms on KDD (0.047 vs 0.287).
  Root cause: CSP generates inputs uniformly across the full feature domain rather than
  the real data distribution. Performance improves significantly as feature dimensionality
  decreases (German has fewer features than KDD).
- **Hybrid sits between CSP and DT** in all cases. Phase 1 seed sampling is real-data based
  (high IDI rate) while Phase 2 is CSP-based (lower IDI rate), explaining the intermediate result.
- **CSP's 1-second per-solve timeout** means effective budget degrades as diversity constraints
  accumulate across a run. This is a design limitation noted in the evaluation.

---

> Export this file to `replication.pdf` and place at repo root before submission.
