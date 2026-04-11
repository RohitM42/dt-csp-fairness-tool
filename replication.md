# Replication

Steps to exactly reproduce the results reported in the paper.

## Environment

- OS: TODO
- Python: TODO
- Hardware: TODO

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

3. Run experiments for all datasets:
   ```bash
   python main.py --dataset kdd    --budget 1000 --runs 20
   python main.py --dataset adult  --budget 1000 --runs 20
   python main.py --dataset compas --budget 1000 --runs 20
   python main.py --dataset german --budget 1000 --runs 20
   python main.py --dataset dutch  --budget 1000 --runs 20
   ```

4. Results are written to `results/`. These should match the raw data files committed to the repo.

## Random Seed

TODO: note whether a fixed seed is used or results are averaged over random runs.

## Expected Results

| Dataset | Baseline IDI (mean) | Hybrid IDI (mean) | p-value |
|---|---|---|---|
| KDD | TODO | TODO | TODO |
| Adult | TODO | TODO | TODO |
| COMPAS | TODO | TODO | TODO |
| German | TODO | TODO | TODO |
| Dutch | TODO | TODO | TODO |

---

> Export this file to `replication.pdf` and place at repo root before submission.
