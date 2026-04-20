# Methodology

This document summarises the measurement and statistical protocol used to
produce the numbers in [RESULTS.md](RESULTS.md). For the raw records, see
[PHASE5_PART1_RESULTS.jsonl](PHASE5_PART1_RESULTS.jsonl) and
[PHASE5_PART2_RESULTS.jsonl](PHASE5_PART2_RESULTS.jsonl).

## Frame model

- **Budget:** 16.67 ms (60 Hz).
- **Jank tolerance:** 1.0 ms. A frame is counted as jank only when `dt >
  17.67 ms`. The tolerance exists to absorb rAF vsync jitter (~0.1–0.3 ms
  of float noise even on a quiet page); a strict `dt > 16.67` check
  classified ~57 % of a constant 5 ms workload as jank during live
  verification. 1.0 ms is the smallest margin that reproducibly dropped
  that false-positive floor below 10 %. Applied uniformly in
  `FrameMetrics`, `RollingFrameMetrics`, `SequentialLoop.step`, and
  `FeatureExtractor`'s `miss_rate_32` feature.

## Workloads

Four synthetic generators model representative browser loads
(`src/harness/workloads.js`):

| Workload | Structure | Base load |
|---|---|---|
| `constant` | Flat 5 ms every frame | Sub-budget — sanity baseline only. |
| `sawtooth` | Ramps 0 → 20 ms over 60 frames, resets | Ramping workload. |
| `burst` | 3 ms base, 5-frame 30 ms spikes every 90 frames | Spiky. |
| `scroll` | Sinusoidal, amplitude × scroll velocity, `k = 0.3` | Continuous with secondary scroll activity. |

Budget-exceeding workloads (`sawtooth`, `burst`, `scroll`) are the primary
comparators; `constant` is a sanity check that the scheduler does not
degrade sub-budget frames.

## Schedulers

| Name | Description | Source |
|---|---|---|
| `B0_AlwaysFull` | Always returns `"full"` — reference for the workload's ground-truth miss pattern. | `src/core/schedulers.js` |
| `B1_EmaThreshold` | EMA of normalised `dt`; thresholds at 0.8 for `reduce`, 1.2 for `degrade`. | `src/core/schedulers.js` |
| `PredictorScheduler` | 353-parameter MLP (`src/core/predictor.js`) fed the 12-dim feature vector from `FeatureExtractor`; thresholds `p_miss` at 0.1 / 0.3. | `src/core/schedulers.js` |

Decisions map to a work-cost multiplier applied to the base load:
`full = 1.0`, `reduce = 0.7`, `degrade = 0.35`.

## Harness

Phase 4 ships a live page (`benchmark/index.html`). Phase 5 drives that
same page through headless Chrome using `scripts/benchmark.js`
(Puppeteer + `vite preview`). Sequential mode runs one active scheduler
per run and lets the two shadow schedulers observe every frame without
affecting execution; confusion-matrix counts come from the shadow pass.

Relevant Chrome flags (`CHROME_FLAGS` in `scripts/benchmark.js`):

- `--disable-background-timer-throttling`
- `--disable-renderer-backgrounding`
- `--disable-backgrounding-occluded-windows`

Each run: 60 s measurement + 10 s cooldown. The first 30 frames are
dropped as JIT warm-up. Per-run shadow frames are streamed to
`shadow.jsonl` for post-hoc feature reconstruction.

## Run plan

### Part 1

4 workloads × 3 schedulers × 10 reps = 120 runs, shuffled with seeded
Fisher–Yates (seed = 42) so that execution order is decoupled from the
`(workload, scheduler)` pair and thermal / JIT attribution can be
recovered via the `executionPosition` column.

### Part 2

40 `Predictor + pretrained + online` + 40 `Predictor + pretrained +
frozen` + 8 `B1` drift-check runs = 88 runs, same shuffle seed. URL
query `?init=pretrained&freeze=true|false` routes each run to the
correct condition at page load.

## Statistics

All tests are non-parametric and reference a deterministic RNG (seed
20260419) for bootstrap reproducibility:

- **Mann–Whitney U** (two-sided). Asymptotic normal approximation with
  continuity correction and tie-corrected variance. We accept the
  asymptotic gap versus scipy's exact test at small n — for n₁ = n₂ =
  10 the gap shifts two-sided p by ~50 % at the tails (e.g., 1e-4 exact
  vs 2e-4 asymptotic); the Go / No-Go gate (p < 0.05) is insensitive
  to that gap.
- **Cohen's d** with pooled SD (Cohen 1988). Sign: `(mean(A) − mean(B)) /
  σ_pooled`. Degenerate case (σ_pooled = 0 and means differ) returns
  signed `Infinity` rather than NaN.
- **95 % percentile bootstrap CI** of the mean. 1000 resamples with
  replacement; seeded RNG (`mulberry32(BOOT_SEED)`) for reproducibility.

## Go / No-Go gate

**GO requires `p < 0.05` AND `|d| ≥ 0.5`** on `jankRate` vs B1. Effect-size
direction is reported separately from the GO / NO-GO verdict: a GO may
mean either "Predictor lower than B1" or "Predictor higher than B1"
(the latter is recorded as such in RESULTS.md).

### Failure policy

If `status = "error"` exceeds 5 % of runs, `scripts/analyze.js` suppresses
the verdict section entirely and surfaces a WARNING. A half-broken run
does not publish a verdict.

## Part 2 drift check

Part 2's B1 sub-plan exists to detect environmental drift between Part 1
and Part 2 runs. Rule (pinned in `scripts/drift-check.js`):

Per workload, define the outlier band `[μ − h, μ + h]` where
`h = max(2σ, 0.005)` (the 0.5 pp floor keeps the band usable when B1's
deterministic baseline collapses σ to ~0). Then:

- **PASS** → 0 outliers AND `|μ_part2 − μ_part1| ≤ 1.0 pp`.
- **WARNING** → 1 outlier AND shift ≤ 1.0 pp.
- **STOP** → ≥ 2 outliers OR shift > 1.0 pp.

Aggregate verdict is the worst per-workload status. STOP halts the Part 2
comparison pipeline for human review.

## Scripts reference

| Script | Purpose |
|---|---|
| `scripts/benchmark.js --mode=part1` | Part 1 sweep (120 runs). |
| `scripts/benchmark.js --mode=part2` | Part 2 sweep (88 runs). |
| `scripts/generate-pretrained.js` | Train the 353 pretrained weights from Part 1's B0 shadow log. |
| `scripts/drift-check.js` | Part 2 harness health gate (run before the comparison). |
| `scripts/analyze.js` | Part 1 results → RESULTS.md sections. |
| `scripts/analyze.js --compare=p1.jsonl,p2.jsonl` | Part 2's three comparison tables. |
| `scripts/measure-overhead.js` | Part 0 predictor-overhead probe. |
| `scripts/measure-floor.js` | Part 0 headless-harness floor probe. |
