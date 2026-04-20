# Phase 5 Part 2 — Pretrained vs Scratch Comparison

_Generated: 2026-04-20T12:51:45.877Z_

Inputs: `K:\GIT\Tempo-js\docs\PHASE5_PART1_RESULTS.jsonl` + `K:\GIT\Tempo-js\docs\PHASE5_PART2_RESULTS.jsonl`.

Gate (pinned): `p < 0.05` AND `|d| ≥ 0.5` on jankRate per workload.
Verdict column: `A<B` / `A>B` indicates direction of the effect (not a value judgement — interpret per section).

## (a) Scratch (Part 1) vs Pretrained + Online (Part 2)

_Same online learning, different starting point. Isolates the **init-quality** contribution._

| Workload | n(A) | n(B) | A jank [95% CI] | B jank [95% CI] | U | p | d | Verdict |
|---|---:|---:|---|---|---:|---:|---:|---|
| constant | 10 | 10 | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 45 | 0.3681 | -0.4472 | NO-GO |
| sawtooth | 10 | 10 | 6.62% [6.46%, 6.78%] | 4.59% [4.52%, 4.67%] | 0 | 1.83e-4 | 9.4281 | GO (A>B) |
| burst | 10 | 10 | 5.45% [5.44%, 5.47%] | 5.51% [5.48%, 5.54%] | 13 | 0.0058 | -1.6064 | GO (A<B) |
| scroll | 10 | 10 | 7.08% [6.97%, 7.19%] | 6.78% [6.62%, 6.94%] | 22 | 0.0376 | 1.1943 | GO (A>B) |

## (b) Pretrained + Online vs Pretrained + Frozen (Part 2)

_Same starting point, online learning on vs off. Isolates the **online-learning marginal value**._

| Workload | n(A) | n(B) | A jank [95% CI] | B jank [95% CI] | U | p | d | Verdict |
|---|---:|---:|---|---|---:|---:|---:|---|
| constant | 10 | 10 | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 49.5 | 1.0000 | -0.0007 | NO-GO |
| sawtooth | 10 | 10 | 4.59% [4.52%, 4.67%] | 11.68% [11.67%, 11.69%] | 0 | 1.82e-4 | -80.2267 | GO (A<B) |
| burst | 10 | 10 | 5.51% [5.48%, 5.54%] | 5.52% [5.50%, 5.54%] | 46 | 0.7913 | -0.1724 | NO-GO |
| scroll | 10 | 10 | 6.78% [6.63%, 6.97%] | 14.61% [14.53%, 14.70%] | 0 | 1.82e-4 | -34.0306 | GO (A<B) |

## (c) B1 (hand-crafted frozen prior) vs Pretrained + Frozen (data-learned frozen prior)

_The blog-post headline match. Both are frozen priors — one designed by a human on EMA thresholds, one learned by SGD from 334k frames._

| Workload | n(A) | n(B) | A jank [95% CI] | B jank [95% CI] | U | p | d | Verdict |
|---|---:|---:|---|---|---:|---:|---:|---|
| constant | 12 | 10 | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 54 | 0.3153 | -0.4714 | NO-GO |
| sawtooth | 12 | 10 | 1.66% [1.66%, 1.67%] | 11.68% [11.67%, 11.69%] | 0 | 8.15e-5 | -649.3136 | GO (A<B) |
| burst | 12 | 10 | 5.56% [5.56%, 5.56%] | 5.52% [5.50%, 5.54%] | 8 | 6.68e-4 | 1.4149 | GO (A>B) |
| scroll | 12 | 10 | 3.38% [3.37%, 3.40%] | 14.61% [14.53%, 14.69%] | 0 | 8.50e-5 | -114.3681 | GO (A<B) |
