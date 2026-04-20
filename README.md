# Tempo

A 353-parameter online-learning MLP that predicts browser frame-budget
overruns and drives an adaptive frame-work scheduler. Research artifact,
not an npm library.

## TL;DR

On four synthetic browser workloads benchmarked in headless Chrome over
208 sixty-second runs, a hand-crafted 3-parameter EMA baseline (`B1`)
beats the 353-parameter MLP on both ramping workloads by 7–10 percentage
points of jank rate, even when the MLP is initialised from weights
pretrained on 334,510 samples of the same distribution. On the
unpredictable-spike workload (`burst`) all schedulers converge to the
same ~5.5 % floor, which is where the learned model finds its niche.

The central finding is structural, not a cold-start artifact: the gap to
B1 persists under ideal pretraining + continued online learning
([RESULTS.md](docs/RESULTS.md) Part 2 section (c)). The Predictor's
inductive bias is wrong for recent-trend threshold crossing; it is at
parity only on the workload where that bias is useless.

## Architecture

| Component | Detail |
|---|---|
| Feature extractor | `src/core/features.js` — 12-dim `Float32Array` per frame (dt EMAs, variance, max, miss-rate, GC pressure, scroll velocity, input activity, DOM mutations, visible animations, workload-delta, device tier). |
| Predictor | `src/core/predictor.js` — 12 → 16 → 8 → 1 MLP with ReLU hidden activations and sigmoid output. Float32 parameters, Float64 hidden activations. Zero-allocation `forward()` / `backward()` after construction. |
| Trainer | `src/core/trainer.js` — SGD with momentum, L2 gradient clip, ring buffer of recent samples, sample-with-replacement minibatches. |
| Schedulers | `src/core/schedulers.js` — `B0_AlwaysFull` (reference), `B1_EmaThreshold` (hand-crafted), `PredictorScheduler` (MLP-driven). |
| Harness | `src/harness/` — `SequentialLoop` runs one active scheduler per run while the other two observe as shadows. `FrameMetrics` + `RollingFrameMetrics` aggregate per-run and per-window statistics. `pretrained.js` inlines the trained weights for offline initialisation. |
| Stats | `src/harness/stats.js` — pure-JS Mann–Whitney U (asymptotic, continuity-corrected, tie-adjusted), Cohen's d (pooled SD), seeded 1000-resample percentile bootstrap. No scipy / R dependency. |

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full measurement
and statistical protocol.

## Live page

`benchmark/index.html` is the sole web entry. Visit it to watch the
three schedulers race on a chosen workload in real time, inspect a
parameter heatmap, and run a comparison sequence over all three.

Supported: Chrome 47+, Firefox 55+, Edge 79+. Safari is explicitly
unsupported; the page relies on `requestIdleCallback` for training
cadence and shows a compatibility banner instead of silently
misbehaving under a `setTimeout` polyfill.

## Quick start

```bash
npm install
npm run dev        # live benchmark at http://localhost:5173/
npm test           # vitest suite (260 tests)
npm run build      # production build to dist/
npm run preview    # preview the production build locally
```

### URL query parameters

`benchmark/app.js` reads two flags from `location.search`:

- `?init=pretrained` initialises the Predictor from
  `src/harness/pretrained.js` instead of He-initialised random weights.
- `&freeze=true` disables `OnlineTrainer.trainStep()` for the session.
  `Predictor.forward()` still runs (the scheduler keeps scoring
  `p_miss`); only the learning path is silenced.

Defaults (`?init=scratch`, no freeze) reproduce the Phase 4 / Part 1
behaviour.

## Results summary

Full tables and interpretation in [docs/RESULTS.md](docs/RESULTS.md).
Machine-readable records:

- [docs/PHASE5_PART1_RESULTS.jsonl](docs/PHASE5_PART1_RESULTS.jsonl) —
  120 runs (4 workloads × 3 schedulers × 10 reps).
- [docs/PHASE5_PART2_RESULTS.jsonl](docs/PHASE5_PART2_RESULTS.jsonl) —
  88 runs (40 `pretrained+online` + 40 `pretrained+frozen` + 8 B1
  drift-check).
- [docs/PHASE5_PART2_COMPARE.md](docs/PHASE5_PART2_COMPARE.md) — three
  pre-registered comparison tables from `scripts/analyze.js --compare`.
- [docs/PHASE5_PART2_DRIFT.md](docs/PHASE5_PART2_DRIFT.md) — harness
  health report (drift from `scripts/drift-check.js`).

Part 1 jank rate means (95 % percentile bootstrap CIs in RESULTS.md):

| Workload | B0 | B1 | Predictor |
|---|---:|---:|---:|
| constant | 0.00 % | 0.00 % | 0.00 % |
| sawtooth | 11.69 % | **1.66 %** | 6.62 % |
| burst | 5.55 % | 5.56 % | **5.45 %** |
| scroll | 14.68 % | **3.38 %** | 7.08 % |

Part 2 (Predictor conditions, same seeds):

| Workload | Scratch + online | Pretrained + online | Pretrained + frozen |
|---|---:|---:|---:|
| constant | 0.00 % | 0.00 % | 0.00 % |
| sawtooth | 6.62 % | 4.59 % | 11.68 % |
| burst | 5.45 % | 5.51 % | 5.52 % |
| scroll | 7.08 % | 6.78 % | 14.61 % |

## Reproducing the benchmark

### Part 1 — 120-run sweep (~2 h 25 m)

```bash
npm run build
node scripts/benchmark.js --mode=part1 --reps=10
node scripts/analyze.js
```

### Part 2 — 88-run pretrained sweep (~1 h 43 m)

```bash
npm run build
node scripts/benchmark.js --mode=part2 --reps=10
node scripts/drift-check.js --part1=docs/PHASE5_PART1_RESULTS.jsonl --part2=docs/PHASE5_PART2_RESULTS.jsonl
node scripts/analyze.js --compare=docs/PHASE5_PART1_RESULTS.jsonl,docs/PHASE5_PART2_RESULTS.jsonl
```

### Regenerating the pretrained weights

```bash
# Requires Part 1's shadow.jsonl (generated during --mode=part1; gitignored).
node scripts/generate-pretrained.js --seed=42 --epochs=5 --batch=64
```

Determinism is enforced by `tests/generate-pretrained.test.js`: given
the same seed and the same shadow-log SHA-256
(`PRETRAINED_META.sourceDataSHA256` pins the current value), the script
produces byte-identical weights.

### Part 0 harness probes

```bash
node scripts/measure-overhead.js      # predictor-infra per-frame cost
node scripts/measure-floor.js         # headless ambient jank floor
```

## Project layout

```
benchmark/       Live page (index.html, app.js, charts.js, live-controls.js, style.css)
src/core/        Feature extractor, Predictor, OnlineTrainer, schedulers, constants
src/harness/     SequentialLoop, metrics, workloads, work-cost, pretrained weights, stats
scripts/         benchmark / analyze / drift-check / generate-pretrained / Part 0 probes
tests/           Vitest suites — 260 tests covering numerics (gradcheck, Mann–Whitney
                 against scipy reference values, bootstrap determinism), harness glue,
                 pure-function DOM helpers, and end-to-end integration smoke
docs/            Methodology, results, raw JSONL, comparison and drift reports
```

## Methodology in one paragraph

Headless Chrome (Puppeteer + `vite preview`) with background-throttling
flags disabled. Each run: 60 s measurement + 10 s cooldown, first
30 frames dropped as JIT warm-up. Four synthetic workloads (`constant`,
`sawtooth`, `burst`, `scroll`); one active scheduler per run with the
other two observed as shadows for confusion-matrix accounting. Jank
threshold is budget + 1.0 ms vsync-jitter tolerance. 120-run sweep
(Part 1) shuffled with a seeded Fisher–Yates (seed = 42); Part 2's 88
runs use the same shuffle seed with condition-specific URL queries.
Go / No-Go gate: two-sided Mann–Whitney U `p < 0.05` **and** Cohen's
`|d| ≥ 0.5` on jank rate versus B1, with 95 % percentile bootstrap CIs
from 1000 seeded resamples (BOOT_SEED = 20260419). Failure policy
suppresses the verdict if `status = "error"` exceeds 5 % of runs.

## Limitations

- **In-distribution only.** Pretrained weights are trained on Part 1's
  B0-active shadow log; Part 2 evaluation samples the same four-workload
  distribution. Out-of-distribution generalisation is explicitly out of
  scope.
- **Feature replay gap.** Training data re-extracts the 12-dim feature
  vector from shadow-log `dt` only. Scroll velocity, input events, DOM
  mutations and GC pressure are zero during training because the shadow
  log doesn't record them. Live and headless evaluation also run with
  these signals near zero, so the gap is small — but it exists.
- **Device tier pinned.** Pretrained weights assume
  `DEVICE_TIER_DEFAULT = 1`. `navigator.hardwareConcurrency` varies by
  host; reproducing the exact weights across machines requires
  `hardwareConcurrency: undefined` in the `FeatureExtractor`
  constructor.
- **Asymptotic Mann–Whitney U.** At `n = 10` per cell the asymptotic
  normal approximation diverges from scipy's exact test by ~50 % at
  the tails (e.g., 1e-4 exact vs 2e-4 asymptotic). The `p < 0.05`
  gate is insensitive to that gap; a cross-check against scipy for a
  disputed cell would re-open that assumption.
- **Safari unsupported.** See the Live page section above.

## License

[MIT](LICENSE). Copyright © 2026 Haksung Lee.
