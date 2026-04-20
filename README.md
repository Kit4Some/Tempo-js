# Tempo

Can a 353-parameter online-learning MLP outperform a well-tuned EMA heuristic at browser frame scheduling? Three ablations across 208 benchmark runs locate the answer: mostly no, with a tie on burst — the one workload where EMA's smoothing can't keep up.

**Status.** Phase 5 complete — Part 1 (scheduler comparison, 120 runs) and Part 2 (pretrained vs scratch, 88 runs) both committed. Blog post forthcoming (Phase 6b). This is a research artifact, not a maintained library; PRs are not reviewed.

## TL;DR

On predictable ramps (sawtooth, scroll-correlated), a well-tuned EMA heuristic beats a 353-parameter MLP by ~10 percentage points in jank rate — even when the MLP is pretrained on over 300,000 samples of in-distribution data and given 60 seconds of online learning per run.

On unpredictable bursts, the MLP holds its own (within noise).

Three ablations in the Phase 5 benchmark ruled out the obvious suspects: not cold-start, not data quantity, not online learning cadence.

A companion single-file analysis, [`tempo.js`](./tempo.js) (comment-stripped twin: [`tempo_onlycode.js`](./tempo_onlycode.js)), isolates the actual mechanism. The MLP has the representational capacity to imitate the EMA heuristic — 98% train agreement when supervised offline on B1's decisions. But online SGD and offline distillation, starting from identical initialization, converge to nearly-orthogonal weight vectors (cosine 0.105; same-seed online runs cluster at 0.9997 for reference). The two optimizers are solving geometrically different problems, not the same problem at different learning rates.

The residual gap is a learnability gap under online self-generated data, not a capacity limit of the 353-parameter architecture. Falsifiable repair directions (DAgger, distillation-anchored loss, grid-supervised pretraining) are listed in `tempo.js` §14.

Full Phase 5 numbers in §4 and [docs/RESULTS.md](docs/RESULTS.md). Mechanism in [`tempo.js`](./tempo.js) (run: `node tempo.js`).

## Key results

Jank-rate means per `(workload, scheduler)` cell, 60 s per run, n = 10 reps (n = 12 for B1 after combining Part 1 + Part 2 drift-check). 95 % percentile bootstrap CIs in [RESULTS.md](docs/RESULTS.md).

| Workload | B0 | B1 | Pred (Scr) | Pred (Pr+On) | Pred (Pr+Fr) | Best | Notes |
|---|---:|---:|---:|---:|---:|---|---|
| constant | 0.00 % | 0.00 % | 0.00 % | 0.00 % | 0.00 % | tied | sanity (below measurement floor) |
| sawtooth | 11.69 % | **1.66 %** | 6.62 % | 4.59 % | 11.68 % | **B1** | Frozen ≈ B0 |
| burst | 5.55 % | 5.56 % | **5.45 %** | 5.51 % | 5.52 % | Scratch (tied) | within noise |
| scroll | 14.68 % | **3.38 %** | 7.08 % | 6.78 % | 14.61 % | **B1** | Frozen ≈ B0 |

*`B0` = always-full reference (never reduces or degrades). `B1` = EMA-threshold heuristic.*
*`Pred (Scr)` = scratch initialization + online learning (Part 1 baseline).*
*`Pred (Pr+On)` = pretrained initialization + online learning.*
*`Pred (Pr+Fr)` = pretrained initialization, frozen weights (no online learning).*

Row-wise winner in **bold**.

Directional p-values and Cohen's d in [PHASE5_PART2_COMPARE.md](docs/PHASE5_PART2_COMPARE.md). B1 drift between Part 1 and Part 2 was ≤ 0.03 pp on every workload ([PHASE5_PART2_DRIFT.md](docs/PHASE5_PART2_DRIFT.md)).

*These headline numbers are the **symptom**. The companion analysis [`tempo.js`](./tempo.js) isolates the **mechanism** — why the MLP loses to a 3-threshold heuristic despite having 100× more parameters. Four experiments in one zero-dependency script: benchmark reproduction, policy distillation (capacity test), optimizer divergence (learnability test), decision-surface visualization.*

### On effect sizes

Cohen's d values in [docs/RESULTS.md](docs/RESULTS.md) are unusually large (|d| up to 649 in Table c). This is an artifact of minimal run-to-run variance in headless Chrome, not an error — pooled standard deviation drops to 0.015 pp for deterministic schedulers, inflating d. Interpret magnitudes as percentage-point differences in the tables above; treat d as a "beyond measurement noise" qualifier only. Full validation in [docs/PHASE5_COHENS_D_VALIDATION.md](docs/PHASE5_COHENS_D_VALIDATION.md).

<!-- Day 3 activation pending: Live demo section (GitHub Pages URL + live-vs-headless floor discussion) goes here once the repo flips to public. -->

## Design

Single-file-per-concern, zero runtime dependencies in the core. `devDependencies` only: `vite`, `vitest`, `puppeteer`.

**Core:**
- [src/core/predictor.js](src/core/predictor.js) — 353-parameter MLP (`12 → 16 → 8 → 1`), zero-allocation `forward` / `backward`.
- [src/core/trainer.js](src/core/trainer.js) — SGD with momentum + L2 grad clip, ring-buffer minibatches.
- [src/core/features.js](src/core/features.js) — 12-dim feature extractor.
- [src/core/schedulers.js](src/core/schedulers.js) — `B0_AlwaysFull`, `B1_EmaThreshold`, `PredictorScheduler`.
- [src/core/constants.js](src/core/constants.js) — single source of truth for tunable numerics.

**Harness:**
- [src/harness/sequential-loop.js](src/harness/sequential-loop.js) — one active + two shadow schedulers per frame.
- [src/harness/metrics.js](src/harness/metrics.js) — `FrameMetrics`, `RollingFrameMetrics`.
- [src/harness/stats.js](src/harness/stats.js) — pure-JS Mann–Whitney U, Cohen's d, seeded percentile bootstrap.
- [src/harness/pretrained.js](src/harness/pretrained.js) — inlined 353-element `Float32Array` with training provenance metadata.

**Scripts:**
- [scripts/benchmark.js](scripts/benchmark.js) — Puppeteer headless benchmark (`--mode=part1|part2`, `--resume`).
- [scripts/analyze.js](scripts/analyze.js) — Go/No-Go rendering + Part 2 `--compare`.
- [scripts/drift-check.js](scripts/drift-check.js) — Part 2 B1 drift gate.
- [scripts/generate-pretrained.js](scripts/generate-pretrained.js) — offline training from Part 1 shadow log.
- [scripts/measure-overhead.js](scripts/measure-overhead.js), [scripts/measure-floor.js](scripts/measure-floor.js) — Part 0 harness probes.

**Tests:** 260 vitest cases covering predictor numerics (analytic-vs-numeric gradcheck), Mann–Whitney U against scipy reference values, bootstrap determinism, harness glue, pure DOM helpers, and the full analyze / benchmark / drift pipeline.

## The single-file artifact

[`tempo.js`](./tempo.js) at the repository root is a ~600-line zero-dependency Node script that reproduces the Phase 5 benchmark and adds three mechanistic experiments the modular codebase doesn't cover:

- **Policy distillation** — can the MLP *represent* B1's policy under offline supervision? (98% train agreement → yes)
- **Optimizer divergence** — how different are the directions online SGD and distillation descend in weight space? (cosine 0.105 with baseline 0.9997 → geometrically orthogonal)
- **Decision-surface visualization** — ASCII grid of what each scheduler actually computes over (ema_fast, ema_slow)

Run: `node tempo.js` for the full sawtooth report. Other workloads: `node tempo.js burst` or `node tempo.js scroll` (benchmark only).

This file is the reference for the forthcoming blog post's mechanistic narrative. The Phase 5 production pipeline lives in `src/`, `scripts/`, and `docs/`; `tempo.js` distills *why* the Phase 5 numbers came out the way they did. Both reference the same constants, the same MLP architecture, and the same deterministic seed protocol.

For readers who want the logic alone: [`tempo_onlycode.js`](./tempo_onlycode.js) is a comment-stripped rendering of the same file (~480 lines vs ~820), with the `main()` report orchestrator removed. It reads as a pure library — import the experiment functions directly, no output, no narrative. Both files are byte-identical in their executable semantics; only the commentary and reporting shell differ.

## Development

```bash
npm install
npm run dev       # live benchmark at http://localhost:5173/Tempo-js/
npm test          # vitest suite (260 tests)
npm run build     # production build to dist/
npm run preview   # preview the production build at http://localhost:4173/Tempo-js/
```

The live page in `benchmark/index.html` reads two optional URL parameters:

- `?init=pretrained` loads the Predictor from [src/harness/pretrained.js](src/harness/pretrained.js) instead of random He-initialisation.
- `&freeze=true` disables `OnlineTrainer.trainStep()` for the session (the Predictor keeps scoring, only learning is silenced).

Default (`?init=scratch`, no freeze) reproduces the Phase 4 / Part 1 behaviour.

## Methodology and reproducibility

- **Protocol:** [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — frame model, workloads, schedulers, Chrome flags, run-plan shuffling, stats, Go/No-Go gate, Part 2 drift rule.
- **Raw results:** [docs/PHASE5_PART1_RESULTS.jsonl](docs/PHASE5_PART1_RESULTS.jsonl) (120 runs), [docs/PHASE5_PART2_RESULTS.jsonl](docs/PHASE5_PART2_RESULTS.jsonl) (88 runs).
- **Part 0 probes:** [docs/PHASE5_PART0_OVERHEAD.md](docs/PHASE5_PART0_OVERHEAD.md) (predictor infra cost), [docs/PHASE5_PART0_FLOOR.md](docs/PHASE5_PART0_FLOOR.md) (headless ambient floor).
- **Pretrained artifact:** [docs/PHASE5_PART2_WEIGHTS.json](docs/PHASE5_PART2_WEIGHTS.json) with `sourceDataSHA256` pinning the exact `shadow.jsonl` input.

### Reproducing the benchmark

```bash
# Part 1 (120 runs, ≈2 h 25 m)
npm run build
node scripts/benchmark.js --mode=part1 --reps=10
node scripts/analyze.js

# Part 2 (88 runs, ≈1 h 43 m)
node scripts/benchmark.js --mode=part2 --reps=10
node scripts/drift-check.js --part1=docs/PHASE5_PART1_RESULTS.jsonl --part2=docs/PHASE5_PART2_RESULTS.jsonl
node scripts/analyze.js --compare=docs/PHASE5_PART1_RESULTS.jsonl,docs/PHASE5_PART2_RESULTS.jsonl
```

### Regenerating the pretrained weights

```bash
# Requires Part 1's shadow.jsonl (generated during --mode=part1, gitignored).
node scripts/generate-pretrained.js --seed=42 --epochs=5 --batch=64
```

`tests/generate-pretrained.test.js` enforces determinism: given the same seed and the same `shadow.jsonl` SHA-256 (pinned in `PRETRAINED_META`), the script produces byte-identical weights.

## Browser support

Chrome 47+, Firefox 55+, Edge 79+. Safari is unsupported — the page relies on `requestIdleCallback` for training cadence and paint deferral; a `setTimeout` polyfill would distort training semantics, so the page shows a compatibility banner instead of silently misbehaving.

## License

[MIT](LICENSE). Copyright © 2026 Haksung Lee.

## Acknowledgements

Inspired by Karpathy's microGPT: "the full algorithmic content of what is needed." Everything else, as Karpathy notes, is just efficiency. The neural net here operates at a very different scale, but the zero-dependency, math-first principle is borrowed directly.

The work-cost multipliers (0.7 for reduce, 0.35 for degrade) are calibrated against the decorative/essential split pattern common in animation libraries (Framer Motion, Motion One). Sensitivity analysis for these values is a limitation noted in [METHODOLOGY.md](docs/METHODOLOGY.md).
