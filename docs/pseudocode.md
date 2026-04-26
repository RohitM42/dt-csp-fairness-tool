# DT-Guided Fairness Testing: Pseudocode

Algorithm for the decision tree-guided search method (primary method).

```
Input:  X       -- dataset
        f       -- black-box model (predict() access only)
        S       -- query budget (default 1000)
        r       -- seed ratio (default 0.15)
        d       -- max DT depth (default 4)

S_seed   = floor(r * S)
S_guided = S - S_seed
seen     = {} (empty hash set for deduplication)

--- Phase 1: Seed sampling ---

for i = 1 to S_seed:
    x  = random row sampled from X
    x' = flip all sensitive features of x
    l  = 1 if |f(x) - f(x')| > 0.05 else 0
    record (x, l) for DT training
    if l == 1: add x to seen

DT = train DecisionTreeClassifier(max_depth=d) on labelled seed samples

--- Phase 2: Guided sampling ---

while S_guided > 0:
    C = sample (5 * S_guided) rows from X
    P = { x in C : DT(x) == 1 }
    if P is empty: P = C

    for each x in P while S_guided > 0:
        x' = flip all sensitive features of x
        if |f(x) - f(x')| > 0.05: add x to seen
        S_guided = S_guided - 1

return |seen| / S
```

## Notes

- The candidate pool size `5 * S_guided` scales with the remaining budget, so it shrinks naturally as the budget depletes.
- Sensitive features are all flipped simultaneously (intersectional counterfactual).
- The hash set `seen` spans both phases; IDIs found during seeding count toward the final ratio.
- The DT is trained once after the seed phase and is not updated online.
- If no DT-positive candidates exist in a given iteration, all sampled rows are used as fallback.
