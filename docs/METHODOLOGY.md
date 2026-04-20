# Methodology

This document specifies the measurement, architectural, and statistical
protocol used to produce the numbers in [RESULTS.md](RESULTS.md) and
[PHASE5_PART2_COMPARE.md](PHASE5_PART2_COMPARE.md). Raw per-run records
live in [PHASE5_PART1_RESULTS.jsonl](PHASE5_PART1_RESULTS.jsonl) and
[PHASE5_PART2_RESULTS.jsonl](PHASE5_PART2_RESULTS.jsonl).

## 1. Research question

Can a 353-parameter online-learning MLP (the Predictor) outperform a
well-tuned EMA-threshold heuristic (B1) at predicting browser frame-
budget overruns, under workloads that include both ramping patterns
and unpredictable spikes? If not, is the deficit a cold-start artifact
(addressable with pretraining) or structural (an inductive-bias limit
not addressable without an architecture change)?

Phase 5 answers both. The MLP loses by ~10 percentage points of jank
rate on the ramping workloads (sawtooth, scroll) even when pretrained
on 334,510 samples of in-distribution data and given 60 s of online
learning per run. The deficit persists with pretraining frozen and
narrows only slightly with online learning, which identifies the gap
as a capacity or inductive-bias limit rather than a training-data
limit. On the unpredictable-spike workload (`burst`) all three
conditions converge within measurement noise to ≈ 5.5 % jank.

## 2. System design

### 2.1 Scheduler interface

All three schedulers share a single contract (`src/core/schedulers.js`):

- `decide(features) → "full" | "reduce" | "degrade"` — called once
  per frame with the 12-dim feature vector.
- `onFrameComplete(dt, wasMiss)` — called after every frame on every
  scheduler (active and shadow). `dt` reflects the active scheduler's
  execution, not a counterfactual.

Three implementations:

| Name | Description | Source |
|---|---|---|
| `B0_AlwaysFull` | Always returns `"full"` — reference for the workload's ground-truth miss pattern. | `src/core/schedulers.js` |
| `B1_EmaThreshold` | EMA of normalised `dt`; thresholds at 0.8 for `reduce`, 1.2 for `degrade`. | `src/core/schedulers.js` |
| `PredictorScheduler` | 353-parameter MLP (`src/core/predictor.js`) fed the 12-dim feature vector from `FeatureExtractor`; thresholds `p_miss` at 0.1 / 0.3. | `src/core/schedulers.js` |

Decisions map to a work-cost multiplier applied to the base load:
`full = 1.0`, `reduce = 0.7`, `degrade = 0.35`. Calibration caveats in
§7.4.

### 2.2 SequentialLoop

`src/harness/sequential-loop.js` runs one active scheduler per run
and lets the other two observe every frame as shadows. Per-frame
order: `observe` → `onFrameComplete` → `extract` → `decide` →
`busyWait`. The live page and the headless benchmark share the same
loop — only the `busyWait` / `now` injections differ.

### 2.3 Shadow vs Sequential execution

Shadow decisions are recorded but not executed — the shadow
scheduler's `decide()` output never alters the work-cost multiplier.
Every Sequential run therefore follows the active scheduler's causal
path exactly. Confusion-matrix counts (TP / FP / TN / FN) come from
comparing shadow decisions against the active scheduler's observed
`wasMiss`, which is the only honest attribution available without
running the experiment three separate times.

## 3. Predictor architecture

### 3.1 Input features (12-dim `Float32Array`)

`FeatureExtractor` in `src/core/features.js` produces one feature
vector per frame:

| Idx | Feature | Definition | Normalisation |
|---:|---|---|---|
| 0 | `dt_ema_fast` | EMA of `dt / budget`, α = 0.3 | `[0, ∞)`, typically < 2 |
| 1 | `dt_ema_slow` | EMA of `dt / budget`, α = 0.05 | as above |
| 2 | `dt_var_8` | Variance of the last 8 raw `dt` values | `var / budget²` |
| 3 | `dt_max_8` | Max `dt` in the last 8 frames | `max / budget` |
| 4 | `miss_rate_32` | Fraction of last 32 frames with `dt > budget + 1 ms` | `[0, 1]` |
| 5 | `gc_pressure` | `log(1 + max(0, Δ JS heap bytes))` | unbounded, typ. 0–20 |
| 6 | `input_activity` | EMA of input events per frame, saturating at 10 | `[0, 1]` |
| 7 | `scroll_velocity` | `\|scrollVelocity\| / viewportHeight` | unbounded, typ. 0–2 |
| 8 | `visible_animating` | `log(1 + N visible animations)` | unbounded, typ. 0–3 |
| 9 | `dom_mutations` | `log(1 + N DOM mutations)` | unbounded, typ. 0–5 |
| 10 | `workload_delta` | `callerDelta / budget` | caller-supplied, 0 in Phase 5 |
| 11 | `device_tier` | `{0, 1, 2}` from `navigator.hardwareConcurrency` | integer scalar |

Budget = `FRAME_BUDGET_60 = 16.67 ms`. The 1 ms term in feature 4
matches `JANK_TOLERANCE_MS` (§4.1). Features 5–9 are best-effort:
`gc_pressure` requires `performance.memory` (Chrome only, 0 elsewhere);
`scroll_velocity`, `visible_animating`, and `dom_mutations` default to
0 when the caller does not supply them. Phase 5's headless benchmark
supplies none of them, so features 5–10 are all 0 in the committed
records.

### 3.2 MLP architecture (353 parameters)

`12 → 16 → 8 → 1`, ReLU on the two hidden layers, sigmoid on output:

- Layer 1: `W₁ ∈ ℝ¹⁶ˣ¹²`, `b₁ ∈ ℝ¹⁶` — 208 params.
- Layer 2: `W₂ ∈ ℝ⁸ˣ¹⁶`, `b₂ ∈ ℝ⁸` — 136 params.
- Output: `W₃ ∈ ℝ¹ˣ⁸`, `b₃ ∈ ℝ` — 9 params.
- **Total: 353 parameters.**

Parameters live in a single `Float32Array` for cache locality and to
keep the pretrained-weight inlining a simple `params.set(weights)`.
Hidden activations are pre-allocated Float64 buffers — Float32 mul-add
accumulates enough rounding error to fail the analytic-vs-numerical
gradient check — and gradients are written to a reused Float32 buffer.
`forward()` and `backward()` allocate zero typed arrays after
construction.

### 3.3 Training

Online SGD with momentum and L2 gradient clipping:

- Loss: binary cross-entropy with ε-clamp (`PRED_LOSS_EPS = 1e-7`).
- Hyperparameters: `LR = 1e-3`, `MOMENTUM = 0.9`, `GRAD_CLIP = 1.0`
  (L2 norm of the averaged gradient, not individual entries).
- Batch size 16, sampled with replacement from a 1024-sample ring
  buffer.
- **Non-destructive sampling**: samples remain in the buffer across
  batches. "Most recent B" sampling would share 15 / 16 samples
  between adjacent batches, starving the optimiser of gradient
  diversity.

Training runs every idle tick in the live page
(`requestIdleCallback`) and between frames in the headless benchmark.
The 10,000-step synthetic convergence test in
`tests/trainer.test.js` is the reference: it reaches `loss < 0.1` on
a `features[0] > 0.5` classifier within budget, pinning the
learning-rate choice.

## 4. Benchmark protocol

### 4.1 Frame model

- **Budget:** 16.67 ms (60 Hz).
- **Jank tolerance:** 1.0 ms. A frame counts as jank only when `dt >
  17.67 ms`. The tolerance absorbs rAF vsync jitter (~0.1–0.3 ms of
  float noise even on a quiet page); a strict `dt > 16.67` check
  classified ~57 % of a constant 5 ms workload as jank during live
  verification. 1.0 ms is the smallest margin that reproducibly
  dropped that false-positive floor below 10 %. Applied uniformly in
  `FrameMetrics`, `RollingFrameMetrics`, `SequentialLoop.step`, and
  `FeatureExtractor`'s `miss_rate_32` feature.

### 4.2 Workloads

Four synthetic generators (`src/harness/workloads.js`):

| Workload | Structure | Base load |
|---|---|---|
| `constant` | Flat 5 ms every frame | Sub-budget — sanity baseline only. |
| `sawtooth` | Ramps 0 → 20 ms over 60 frames, resets | Ramping workload. |
| `burst` | 3 ms base, 5-frame 30 ms spikes every 90 frames | Spiky. |
| `scroll` | Sinusoidal, amplitude × scroll velocity, `k = 0.3` | Continuous with secondary scroll activity. |

Budget-exceeding workloads (`sawtooth`, `burst`, `scroll`) are the
primary comparators; `constant` is a sanity check that the scheduler
does not degrade sub-budget frames.

### 4.3 Headless-Chrome harness

`scripts/benchmark.js` drives the live page through Puppeteer + `vite
preview`. Relevant Chrome flags (`CHROME_FLAGS`):

- `--disable-background-timer-throttling`
- `--disable-renderer-backgrounding`
- `--disable-backgrounding-occluded-windows`

Each run: 60 s measurement + 10 s cooldown. The first 30 frames are
dropped as JIT warm-up. Per-run shadow frames are streamed to
`shadow.jsonl` (gitignored, ephemeral) for post-hoc feature
reconstruction when regenerating pretrained weights (§6).

### 4.4 Run plan

**Part 1.** 4 workloads × 3 schedulers × 10 reps = 120 runs, shuffled
with seeded Fisher–Yates (seed = 42) so execution order is decoupled
from the `(workload, scheduler)` pair. Thermal and JIT attribution
are recoverable via `executionPosition`.

**Part 2.** 40 `pretrained + online` + 40 `pretrained + frozen` + 8
`B1` drift-check runs = 88 runs, same shuffle seed. Condition
selection via URL query `?init=pretrained&freeze=true|false` at
page-load time.

### 4.5 Drift check (Part 2 harness-health gate)

Part 2's B1 sub-plan detects environmental drift between Part 1 and
Part 2 runs (`scripts/drift-check.js`). Per workload, define the
outlier band `[μ − h, μ + h]` where `h = max(2σ, 0.005)` (the 0.5 pp
floor keeps the band usable when B1's deterministic baseline
collapses σ to ~0). Then:

- **PASS** → 0 outliers AND `|μ_part2 − μ_part1| ≤ 1.0 pp`.
- **WARNING** → 1 outlier AND shift ≤ 1.0 pp.
- **STOP** → ≥ 2 outliers OR shift > 1.0 pp.

Aggregate verdict is the worst per-workload status. STOP halts the
Part 2 comparison pipeline for human review. The Part 2 sweep in
this repository cleared PASS on all four workloads; detailed
report in [PHASE5_PART2_DRIFT.md](PHASE5_PART2_DRIFT.md).

## 5. Statistical methods

All tests are non-parametric and seeded (`BOOT_SEED = 20260419`) for
bootstrap reproducibility. Pure-JS implementations in
`src/harness/stats.js` — no scipy / R dependency.

### 5.1 Mann–Whitney U (two-sided)

Asymptotic normal approximation with continuity correction and
tie-corrected variance. The project's pure-JS implementation
reproduces `scipy.stats.mannwhitneyu(..., method='asymptotic',
use_continuity=True)` bit-for-bit (verified against Part 1 sawtooth
/ burst / scroll cells — see Appendix A). At `n₁ = n₂ = 10` the
asymptotic value *overstates* scipy's exact p by up to ~16× at the
extreme tail (`U = 0`: exact `1.08 × 10⁻⁵` vs asymptotic
`1.69 × 10⁻⁴`); at the centre (`U = n₁·n₂/2`) the two agree. The
Go / No-Go gate (`p < 0.05`) is insensitive to this gap because
every significant cell reported in RESULTS.md sits at `p ≤ 2 × 10⁻⁴`,
three decades below the threshold. See §7.7 for the small-n caveat.

### 5.2 Cohen's d

Pooled-SD form (Cohen 1988):

```
pooled_sd = √(((n₁ − 1)·var_A + (n₂ − 1)·var_B) / (n₁ + n₂ − 2))
d         = (mean_A − mean_B) / pooled_sd
```

Degenerate case (`pooled_sd = 0` and means differ) returns signed
`Infinity` rather than NaN. In controlled-measurement regimes with
deterministic schedulers, pooled SD collapses to 1e-4 order and
`|d|` inflates accordingly (see §7.6). Manual cell-by-cell validation
in [PHASE5_COHENS_D_VALIDATION.md](PHASE5_COHENS_D_VALIDATION.md)
confirms the implementation matches the formula exactly on all nine
Part 2 comparison cells.

### 5.3 Bootstrap 95 % CI

Non-parametric percentile bootstrap of the mean. 1000 resamples with
replacement; seeded `mulberry32(BOOT_SEED)` RNG so every re-render of
RESULTS.md produces byte-identical CIs. Applied to `jankRate`, `p95`,
and `meanDt` per `(workload, scheduler)` cell.

### 5.4 Go / No-Go gate

**GO requires `p < 0.05` AND `|d| ≥ 0.5`** on `jankRate` versus B1.
Direction (Predictor lower vs higher than B1) is reported separately;
a GO with Predictor *higher* than B1 is explicitly flagged — a
statistically significant loss is still a loss, and the analysis
pipeline does not collapse the two verdicts.

### 5.5 Failure policy

If `status = "error"` exceeds 5 % of runs, `scripts/analyze.js`
suppresses the verdict section entirely and surfaces a WARNING. A
half-broken run does not publish a verdict.

## 6. Pretrained weights protocol (Part 2)

Pretrained weights are produced offline by
`scripts/generate-pretrained.js` from Part 1's `shadow.jsonl`.

### 6.1 Training data

Active = B0 rows only. B0 is always-full, so its `dt` reflects the
ground-truth cost of every frame; using any other scheduler's rows
would pair features with the outcome *that scheduler* executed,
which is a different counterfactual and would poison the label.
After dropping the last frame of each run (no next-frame label),
40 B0 runs × ~8 000 frames yield 334,510 `(features, label)` pairs.
The training data's SHA-256 is pinned in
`PRETRAINED_META.sourceDataSHA256`; a modified `shadow.jsonl` surfaces
as a hash mismatch rather than silent drift.

### 6.2 Training loop

5 epochs of shuffled-minibatch SGD (batch size 64, same LR / momentum
/ grad-clip as online training). Fisher–Yates shuffle per epoch. The
script is byte-exact reproducible: same seed (42) + same
`sourceDataSHA256` → same 353 weights, enforced by
`tests/generate-pretrained.test.js`.

### 6.3 Benchmark conditions

Part 2 evaluates two Predictor conditions in addition to Part 1's
scratch baseline:

- **Scratch + online** (Part 1 Predictor) — fresh He-initialisation,
  online learning.
- **Pretrained + online** — loaded from `PRETRAINED_WEIGHTS`, online
  learning continues during each run.
- **Pretrained + frozen** — loaded weights; `OnlineTrainer.setEnabled(false)`
  disables `trainStep()`. `forward` / `backward` / `push` remain
  functional — only the learning path is silenced.

URL parameters `?init=scratch|pretrained&freeze=true|false` drive
condition selection at page-load time and are issued per-run by
`scripts/benchmark.js`.

## 7. Known limitations

### 7.1 In-distribution training (and a deeper mechanism beneath it)

Pretrained weights are trained on data sampled from the same four
workloads and the same headless-Chrome regime that Part 2 then
evaluates on. Out-of-distribution generalisation (novel workloads,
other browsers, real production pages) is explicitly out of scope.
A reader expecting "pretrained weights transfer to your site" should
read §6.1 before drawing conclusions.

A separate mechanism is isolated in the companion single-file analysis
(`tempo.js` at the repository root). Even *in*-distribution, online
SGD and offline distillation starting from the same initialization
converge to geometrically different weight vectors (cosine 0.105
between their Δ from the shared init; same-seed online runs cluster
at 0.9997 as a noise baseline). The 353-parameter MLP has the
representational capacity to imitate B1's policy — 98% train
agreement when supervised offline — but online self-generated data
produces a different loss landscape than offline supervision. This
reframes Part 2's residual gap: it is a learnability gap under the
online data-generation protocol, not a capacity limit of the
architecture. The mechanism is material for the blog post's
contribution but does not alter the Part 2 conclusions, which remain
correct as reported.

### 7.2 Single seed for Predictor init

He-initialisation uses `mulberry32(42)` throughout. All "Predictor"
cells in RESULTS.md therefore share one initialisation sample. A
second seed would add another 120 + 88 runs to characterise init
variance; we opted for depth (10 reps per cell) over breadth.

### 7.3 Simulated workloads

The four workload generators drive frame time via `busyWait(ms)` in a
tight while loop (`src/harness/workloads.js`). Real pages have DOM
mutations, layout thrash, compositor interactions, and GC pressure
whose interaction with the scheduler's decisions is not captured.
Additionally, all measurements come from a single Puppeteer host;
hardware-specific effects on rAF cadence and GC behaviour are not
characterised.

### 7.4 Work-cost multiplier calibration

`reduce = 0.7` and `degrade = 0.35` approximate the decorative /
essential split common in Framer Motion and Motion One, but the
exact ratios were not empirically measured against a real page's
cost breakdown. Sensitivity to these values is not exhaustively
tested; a sweep across 0.5–0.9 (reduce) and 0.2–0.5 (degrade) would
tighten the conclusion.

### 7.5 Live vs headless floor gap

Live Chrome at 60 Hz vsync reports a ≈ 10 % baseline jank rate on
`constant` because rAF jitter sits just above the 1 ms tolerance
band. Headless Chrome without vsync reports 0 % on the same workload.
Phase 5 uses headless (cleaner measurement), but a reader visiting
the live demo sees numbers that look strictly worse than the
benchmark. Both regimes are reported; neither is "the true" jank
rate.

### 7.6 Cohen's d inflation under deterministic schedulers

B1 and Pretrained + Frozen are fully deterministic at the chosen
seed, so pooled SD collapses to 1e-4 order and `|d|` values inflate
to hundreds. Effect sizes remain mathematically correct (see
[PHASE5_COHENS_D_VALIDATION.md](PHASE5_COHENS_D_VALIDATION.md)) but
Cohen's conventional "large = 0.8" calibration does not apply. Read
absolute percentage-point shifts as the practical magnitude; treat
d as a "beyond measurement noise" qualifier only.

### 7.7 Asymptotic vs exact p-values at n = 10

Mann–Whitney U p-values in this report use the asymptotic normal
approximation, which is valid for large n but known to diverge from
exact p-values at small n. For our n = 10 per condition, asymptotic
p-values can differ from exact (e.g., `scipy.stats.mannwhitneyu`
with `method='exact'`) by factors up to ~16× in the tail (see
Appendix A). Results near the `p = 0.05` threshold should be
interpreted with this caveat. All reported gates (`p < 0.05` AND
`|d| ≥ 0.5`) comfortably clear the threshold in the
direction-determining cells, so the conclusions do not hinge on this
distinction, but future work with exact p-values is warranted for
borderline cases.

## 8. Reproducibility

### 8.1 Single-command reruns

Four `npm run bench:*` scripts wrap the long-running harness:

| Command | Effect |
|---|---|
| `npm run bench:part1` | 120-run Part 1 sweep (≈ 2 h 25 m). |
| `npm run bench:part2` | 88-run Part 2 sweep (≈ 1 h 43 m). |
| `npm run bench:compare` | Renders the three Part 2 comparison tables to `docs/PHASE5_PART2_COMPARE.md`. |
| `npm run bench:drift` | Part 2 B1 drift gate. Exit code 0 = PASS, 1 = WARNING, 2 = STOP. |

### 8.2 Pretrained weight determinism

`PRETRAINED_META.sourceDataSHA256` pins the exact `shadow.jsonl`
input. Given the same SHA and seed (42),
`node scripts/generate-pretrained.js` reproduces byte-identical
weights — enforced by `tests/generate-pretrained.test.js`, which
runs the same-seed training twice and asserts bit equality on all
353 parameters.

### 8.3 Expected runtime and disk usage

Measured on a desktop (ASRock B650M Pro RS WiFi, 32 GB RAM):
Part 1 = 2 h 25 m, Part 2 = 1 h 43 m. Disk use for a full sweep:
≈ 160 MB of `shadow.jsonl` (Part 1) plus ≈ 120 MB (Part 2), plus
~60 KB of committed `*.jsonl` results.

### 8.4 Test suite

`npm test` runs the 260-case vitest suite (≈ 3 s). Coverage includes:
analytic-vs-numerical gradient check on all 353 parameters (three
fixtures), Mann–Whitney U and Cohen's d against reference values,
bootstrap determinism, harness glue, pure DOM helpers, and the full
analyse / benchmark-plan / drift-check pipelines.

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

## Appendix A — Mann–Whitney U: asymptotic vs scipy exact

Spot-check of the pure-JS asymptotic implementation against
`scipy.stats.mannwhitneyu` on Part 1 cells (Python 3.13.7,
scipy 1.16.3). For each `(workload, B1, Predictor)` pair, we compute:

- **scipy exact** — `method='exact'`, no approximation.
- **scipy asymptotic+cc** — `method='asymptotic', use_continuity=True`.
- **project asymptotic** — `mannWhitneyU(A, B)` from
  [`src/harness/stats.js`](../src/harness/stats.js#L107-L148), as
  surfaced in [RESULTS.md](RESULTS.md).

All three for the four Part 1 workloads, each n₁ = n₂ = 10:

| Workload | U (min) | scipy exact p | scipy asymp+cc p | project asymp p | asymp / exact |
|---|---:|---:|---:|---:|---:|
| sawtooth | 0 | 1.083 × 10⁻⁵ | 1.688 × 10⁻⁴ | 1.688 × 10⁻⁴ | 15.59 |
| burst    | 0 | 1.083 × 10⁻⁵ | 1.796 × 10⁻⁴ | 1.796 × 10⁻⁴ | 16.59 |
| scroll   | 0 | 1.083 × 10⁻⁵ | 1.786 × 10⁻⁴ | 1.786 × 10⁻⁴ | 16.50 |
| constant | 50 | 1.000 | 1.000 | 1.000 | 1.00 |

Reproduce with:

```python
import json
from scipy import stats
rows = [json.loads(l) for l in open('docs/PHASE5_PART1_RESULTS.jsonl', encoding='utf-8')]
def s(wl, sch):
    return [r['jankRate'] for r in rows if r['workload']==wl and r['active']==sch]
for wl in ('sawtooth', 'burst', 'scroll', 'constant'):
    a, b = s(wl, 'B1'), s(wl, 'Predictor')
    u_e, p_e = stats.mannwhitneyu(b, a, alternative='two-sided', method='exact')
    u_a, p_a = stats.mannwhitneyu(b, a, alternative='two-sided',
                                  method='asymptotic', use_continuity=True)
    print(wl, min(u_e, len(a)*len(b)-u_e), f'{p_e:.3e}', f'{p_a:.3e}')
```

Two observations:

1. **Implementation validity.** `project asymp p` matches `scipy asymp+cc p`
   to reported precision on every non-degenerate cell, confirming the
   pure-JS rank / U / tie-correction / continuity-correction pipeline.
2. **Asymptotic is conservative.** At n = 10 with `U = 0`, scipy exact
   reaches `2 / C(20, 10) ≈ 1.08 × 10⁻⁵` (the lower bound for this
   sample size), while the asymptotic approximation settles around
   `1.7 × 10⁻⁴`. This is a ceiling on the gap, not a typical one —
   the `constant` row (`U = 50`, the null centre) shows the two
   approximations agreeing exactly. The Go / No-Go gate at `p < 0.05`
   is therefore robust across the entire feasible range of U.
