// Phase 5 Part 0 — Predictor infrastructure overhead measurement.
//
// Answers: "does forward + push + trainStep per-frame cost leave enough
// budget for the scheduler's decision to actually save time?"
//
// Methodology:
//   Batch accumulation with N = 1000 iterations per component. Divide
//   total by N. Warm up with 100 iterations first to absorb JIT
//   cold-start skew. No performance.measure() marks (Heisenberg), no
//   per-call now() reads (Spectre 5μs resolution floor).
//
// Gates:
//   - If per-frame infra (upper bound) > 30% of FRAME_BUDGET_60 → ABORT.
//   - Else → PROCEED to scripts/measure-floor.js.
//
// Output:
//   - docs/PHASE5_PART0_OVERHEAD.json — machine-readable
//   - docs/PHASE5_PART0_OVERHEAD.md   — human-readable
//   - stdout summary
//   - exit code: 0 (PROCEED) or 1 (ABORT)

import { performance } from "node:perf_hooks";
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { Predictor } from "../src/core/predictor.js";
import { OnlineTrainer } from "../src/core/trainer.js";
import { FeatureExtractor } from "../src/core/features.js";
import { mulberry32 } from "../tests/helpers/rng.js";
import {
  BATCH_SIZE,
  FRAME_BUDGET_60,
  MLP_INPUT_DIM,
} from "../src/core/constants.js";

const SEED = 42;
const WARMUP_ITERATIONS = 100;
const MEASURE_ITERATIONS = 1000;
const ABORT_GATE_PERCENT = 30;

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");
const OUT_JSON = resolve(REPO_ROOT, "docs/PHASE5_PART0_OVERHEAD.json");
const OUT_MD = resolve(REPO_ROOT, "docs/PHASE5_PART0_OVERHEAD.md");

function buildHarness() {
  const rng = mulberry32(SEED);
  const predictor = new Predictor({ rng });
  const trainer = new OnlineTrainer(predictor, { rng });
  const extractor = new FeatureExtractor();
  // Prime the extractor with enough frames to produce a non-trivial feature
  // vector (variance window needs ≥2 frames for meaningful dt_var).
  for (let i = 0; i < 32; i++) {
    extractor.observe({ dt: 8 + (i % 16) });
  }
  return { predictor, trainer, extractor };
}

function measure(label, fn, iterations) {
  // Warm-up — discard timing. JIT optimizations converge within ~100 iters
  // for inner loops of this size.
  for (let i = 0; i < WARMUP_ITERATIONS; i++) fn(i);

  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) fn(i);
  const elapsedMs = performance.now() - t0;

  const avgUs = (elapsedMs / iterations) * 1000;
  return { label, iterations, totalMs: elapsedMs, avgUs };
}

function main() {
  const { predictor, trainer, extractor } = buildHarness();

  // Build a representative features vector. FeatureExtractor.extract() reuses
  // the same Float32Array each call — we snapshot once so timing isn't
  // conflated with observe/extract work (those measured separately if ever).
  const features = new Float32Array(MLP_INPUT_DIM);
  features.set(extractor.extract(0));

  // --- forward ------------------------------------------------------------
  const forward = measure(
    "predictor.forward",
    () => predictor.forward(features),
    MEASURE_ITERATIONS,
  );

  // --- push ---------------------------------------------------------------
  // Push alternates target 0/1 to reflect realistic miss/no-miss stream.
  const push = measure(
    "trainer.push",
    (i) => trainer.push(features, i & 1),
    MEASURE_ITERATIONS,
  );

  // --- trainStep ----------------------------------------------------------
  // Prerequisite: buffer must have samples. push() just ran, buffer is full
  // of one feature vector repeated — trainStep samples with replacement so
  // it has something to train on.
  const trainStep = measure(
    "trainer.trainStep",
    () => trainer.trainStep(BATCH_SIZE),
    MEASURE_ITERATIONS,
  );

  // --- Per-frame infra cost (upper bound) --------------------------------
  // Per Part 0 decision (c): assume trainStep runs once per frame. Real
  // cadence is rIC-gated (≤1/frame), so this is the upper bound.
  const perFrameUs = forward.avgUs + push.avgUs + trainStep.avgUs;
  const frameBudgetUs = FRAME_BUDGET_60 * 1000;
  const percentOfBudget = (perFrameUs / frameBudgetUs) * 100;

  const gate =
    percentOfBudget > ABORT_GATE_PERCENT ? "ABORT" : "PROCEED";
  const gateReason =
    gate === "ABORT"
      ? `Per-frame infra ${percentOfBudget.toFixed(2)}% > ${ABORT_GATE_PERCENT}% gate. Redesign Predictor cadence before Part 1.`
      : `Per-frame infra ${percentOfBudget.toFixed(2)}% ≤ ${ABORT_GATE_PERCENT}% gate. Continue to scripts/measure-floor.js.`;

  // --- Report -------------------------------------------------------------
  const result = {
    timestamp: new Date().toISOString(),
    seed: SEED,
    iterations: MEASURE_ITERATIONS,
    warmup: WARMUP_ITERATIONS,
    measurements: {
      forward: { total_ms: forward.totalMs, avg_us: forward.avgUs },
      push: { total_ms: push.totalMs, avg_us: push.avgUs },
      trainStep: {
        total_ms: trainStep.totalMs,
        avg_us: trainStep.avgUs,
        batch_size: BATCH_SIZE,
      },
    },
    per_frame_infra_us: {
      upper_bound: perFrameUs,
      percent_of_budget: percentOfBudget,
      frame_budget_us: frameBudgetUs,
    },
    gate,
    gate_reason: gateReason,
    node_version: process.version,
    platform: `${process.platform} ${process.arch}`,
  };

  mkdirSync(dirname(OUT_JSON), { recursive: true });
  writeFileSync(OUT_JSON, JSON.stringify(result, null, 2));

  const md = [
    "# Phase 5 Part 0 — Predictor Overhead",
    "",
    `_Generated: ${result.timestamp} · Node ${result.node_version} · ${result.platform}_`,
    "",
    `Batch-accumulation methodology — N=1000 iterations per component, divide by N after; 100 warmup iterations absorb JIT skew; no per-call now() reads (Spectre resolution floor) and no performance.measure() marks (Heisenberg).`,
    `Warm-up: ${WARMUP_ITERATIONS} iterations (discarded). Measured window: ${MEASURE_ITERATIONS} iterations.`,
    "",
    "## Per-component cost",
    "",
    "| Component | Total (ms) | Avg (μs) |",
    "|---|---:|---:|",
    `| predictor.forward | ${forward.totalMs.toFixed(3)} | ${forward.avgUs.toFixed(2)} |`,
    `| trainer.push | ${push.totalMs.toFixed(3)} | ${push.avgUs.toFixed(2)} |`,
    `| trainer.trainStep (batch ${BATCH_SIZE}) | ${trainStep.totalMs.toFixed(3)} | ${trainStep.avgUs.toFixed(2)} |`,
    "",
    "## Per-frame infra cost (upper bound)",
    "",
    `Assumes trainStep runs once per frame. Actual cadence is rIC-gated (≤1/frame).`,
    "",
    `- Sum (forward + push + trainStep): **${perFrameUs.toFixed(2)} μs**`,
    `- Frame budget (FRAME_BUDGET_60): ${frameBudgetUs.toFixed(0)} μs`,
    `- Percent of budget: **${percentOfBudget.toFixed(2)}%**`,
    "",
    "## Gate decision",
    "",
    `- Threshold: ${ABORT_GATE_PERCENT}% of frame budget`,
    `- Outcome: **${gate}**`,
    `- Reason: ${gateReason}`,
    "",
  ].join("\n");

  writeFileSync(OUT_MD, md);

  // Stdout summary.
  process.stdout.write(`
Phase 5 Part 0 — Overhead
  forward:   ${forward.avgUs.toFixed(2)} μs/call
  push:      ${push.avgUs.toFixed(2)} μs/call
  trainStep: ${trainStep.avgUs.toFixed(2)} μs/call (batch ${BATCH_SIZE})
  ─────────────────────────────────────
  per-frame infra (upper bound): ${perFrameUs.toFixed(2)} μs (${percentOfBudget.toFixed(2)}% of budget)
  gate: ${gate}
  → ${gateReason}

  wrote: ${OUT_JSON}
  wrote: ${OUT_MD}
`);

  process.exit(gate === "ABORT" ? 1 : 0);
}

main();
