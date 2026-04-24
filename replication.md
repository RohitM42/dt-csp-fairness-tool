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

## Group Fairness Characterisation

To reproduce the group-level bias characterisation (DPD and EOD per sensitive feature):

```bash
python characterise.py
```

Output is printed to the terminal and saved to `characterisation.csv` at the repo root.
This is deterministic — no randomness is involved and results are identical on every run.

Expected output file: `characterisation.csv` (18 rows, one per dataset–sensitive feature pair).
Key values to verify:
- `dutch, age`: DPD=0.4793, EOD=0.6364
- `compas, Race`: DPD=0.1278, EOD=0.5000
- `communities_crime, Black`: DPD=0.0079, EOD=0.0000
- `german, AgeInYears`: DPD=0.4282, EOD=1.0000

---

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

Full 20-run results (budget=1000) from `results_v2/`. All methods use all-sensitive-feature
flip. p-values from Wilcoxon signed-rank test (paired, two-sided), Bonferroni-corrected
threshold α=0.01 (5 comparisons).

> **Note:** v1 results (in `results/`) used a single-attribute baseline flip (primary
> sensitive only). v2 results use all-sensitive flip for baseline, consistent with
> `lab4_solution.py` and the three comparison methods. Baseline IDI ratios are substantially
> higher in v2. DT/CSP/Hybrid values are unchanged between v1 and v2.
>
> CSP and Hybrid results below are from v1 (20-run) experiments — the code was not modified
> so results are directly comparable to the v2 baseline.
>

### DT vs Baseline (v2, primary results — `results_v2/`)

| Dataset | Baseline mean ± std | DT mean ± std | p-value | Significant |
|---------|--------------------|--------------:|---------|-------------|
| adult   | 0.4619 ± 0.0191 | 0.5682 ± 0.0257 | 8.84e-05 | Yes |
| compas  | 0.5882 ± 0.0167 | 0.6587 ± 0.0200 | 8.84e-05 | Yes |
| dutch   | 0.0201 ± 0.0042 | 0.1595 ± 0.0458 | 1.03e-04 | Yes |
| german  | 0.3910 ± 0.0091 | 0.3482 ± 0.0352 | 1.30e-04 | Yes (DT lower) |
| kdd     | 0.5598 ± 0.0163 | 0.6482 ± 0.0341 | 8.83e-05 | Yes |

### CSP and Hybrid vs Baseline (v1 method results, v2 corrected baseline)

| Dataset | Baseline (v2) | CSP mean ± std | Hybrid mean ± std |
|---------|--------------|---------------|------------------|
| adult   | 0.4619 | 0.212 ± 0.015 | 0.268 ± 0.025 |
| compas  | 0.5882 | 0.427 ± 0.017 | 0.455 ± 0.034 |
| dutch   | 0.0201 | 0.029 ± 0.006 | 0.085 ± 0.030 |
| german  | 0.3910 | 0.235 ± 0.013 | 0.288 ± 0.019 |
| kdd     | 0.5598 | 0.041 ± 0.007 | 0.124 ± 0.018 |

## Key Findings

- **DT outperforms the corrected baseline on 4 of 5 datasets**, with statistically
  significant improvements on all 4 (Wilcoxon, p < 0.001 in each case). Improvements
  range from +12% (COMPAS) to +694% proportionally (Dutch). DT achieves this at
  comparable runtime to the baseline — the DT training overhead is negligible.

- **German is the only dataset where DT underperforms the baseline** (0.348 vs 0.391,
  −10.9%). This is attributed to three factors: the small dataset size (1,000 rows) causing
  heavy resampling in the guided phase; a relatively uniform discrimination density that
  leaves the DT with no concentrated subspace to identify; and a multi-valued sensitive
  feature (PersonStatusSex, 0–3) creating a mismatch between DT threshold splits and the
  random flip operation.

- **CSP underperforms the corrected baseline on 4 of 5 datasets.** The root cause is a
  domain/distribution mismatch: CP-SAT generates inputs uniformly within `[min, max]`
  feature bounds, while the DNN's discrimination is concentrated on the real data
  distribution. On KDD (19 features, 284k rows), CSP achieves only 0.041 vs baseline 0.560.
  Performance is better on smaller, lower-dimensional datasets (COMPAS: 0.427 vs 0.588;
  Dutch: 0.029 vs 0.020).

- **Hybrid underperforms the corrected baseline on 4 of 5 datasets**, beating it only on
  Dutch. Hybrid's Phase 2 uses CSP constrained to DT-identified rule regions — which is
  better than unconstrained CSP but still generates synthetic inputs that sit off the real
  data distribution. DT-only outperforms Hybrid because DT Phase 2 continues sampling
  real data rows, not synthetic ones.

- **DT is the recommended method.** It is black-box (no model internals required),
  interpretable (extracted decision rules describe which feature combinations drive
  discrimination), and achieves the highest IDI ratios at no additional computational cost
  over random search.

---

> Export this file to `replication.pdf` and place at repo root before submission.
