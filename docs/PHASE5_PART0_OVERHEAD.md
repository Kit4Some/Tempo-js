# Phase 5 Part 0 — Predictor Overhead

_Generated: 2026-04-18T12:18:37.640Z · Node v22.18.0 · win32 x64_

Batch-accumulation methodology: N = 1000 iterations per component, divided by N; 100-iteration warm-up discarded (JIT cold-start). No per-call `performance.now()` reads (Spectre 5 μs floor) and no `performance.measure` marks (Heisenberg). See [METHODOLOGY.md](METHODOLOGY.md) for the full protocol.

## Per-component cost

| Component | Total (ms) | Avg (μs) |
|---|---:|---:|
| predictor.forward | 3.043 | 3.04 |
| trainer.push | 0.102 | 0.10 |
| trainer.trainStep (batch 16) | 21.474 | 21.47 |

## Per-frame infra cost (upper bound)

Assumes trainStep runs once per frame. Actual cadence is rIC-gated (≤1/frame).

- Sum (forward + push + trainStep): **24.62 μs**
- Frame budget (FRAME_BUDGET_60): 16670 μs
- Percent of budget: **0.15%**

## Gate decision

- Threshold: 30% of frame budget
- Outcome: **PROCEED**
- Reason: Per-frame infra 0.15% ≤ 30% gate. Continue to scripts/measure-floor.js.
