# Phase 5 Part 2 — B1 Drift Check

_Generated: 2026-04-20T12:51:27.543Z_

Inputs: `docs/PHASE5_PART1_RESULTS.jsonl` (Part 1 B1 baseline) vs `docs/PHASE5_PART2_RESULTS.jsonl` (Part 2 drift runs).

**Aggregate verdict: `PASS`**

Rule (pinned): per workload, outlier = Part-2-B1 jankRate ∉ [μ ± 2σ] of Part-1-B1.
`PASS` = 0 outliers AND |mean shift| ≤ 1.0pp. `WARNING` = 1 outlier. `STOP` = ≥2 outliers OR |mean shift| > 1.0pp.

| Workload | n₁ | n₂ | Part 1 μ±σ | ±2σ band | Part 2 μ | Mean shift | Outliers | Status |
|---|---:|---:|---|---|---|---:|---:|---|
| constant | 10 | 2 | 0.00% ± 0.00% | [-0.50%, 0.50%] | 0.00% | +0.00pp | 0/2 | PASS |
| sawtooth | 10 | 2 | 1.66% ± 0.00% | [1.16%, 2.16%] | 1.67% | +0.01pp | 0/2 | PASS |
| burst | 10 | 2 | 5.56% ± 0.00% | [5.06%, 6.06%] | 5.56% | -0.00pp | 0/2 | PASS |
| scroll | 10 | 2 | 3.38% ± 0.02% | [2.88%, 3.88%] | 3.41% | +0.03pp | 0/2 | PASS |

_No drift detected. Proceed to `analyze.js --compare` for the three comparison tables._
