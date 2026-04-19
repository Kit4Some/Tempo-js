// Phase 5 Part 2 — offline pretraining of the 353-parameter Predictor.
//
// Trains on Part 1's shadow log (active=B0 rows only). B0 is always-full so
// dt reflects the ground-truth cost of every frame — using any other
// scheduler's rows would pair features with the outcome that scheduler
// executed, which is a different counterfactual and would poison the label.
//
// Usage:
//   node scripts/generate-pretrained.js                       # defaults
//   node scripts/generate-pretrained.js --seed=42 --epochs=5  # explicit
//   node scripts/generate-pretrained.js --input=shadow.jsonl --output=docs/PHASE5_PART2_WEIGHTS.json
//
// Determinism: given --seed=N and the same input JSONL, two runs produce
// bit-identical weights (see tests/generate-pretrained.test.js). The
// sourceDataSHA256 field in PRETRAINED_META commits to the exact JSONL
// contents the weights were trained on.
//
// Limitations (documented in PHASE5_NOTES.md and the blog post):
//   - Replay reconstructs the 12-dim feature vector from dt only. Shadow log
//     does not record scrollVelocity / inputEvents / DOM / memory, so
//     features 6-9 (gc_pressure, input_activity, scroll_velocity, anims,
//     dom_mutations) are all 0 during training. Live Phase 4 & headless
//     Phase 5 protocols show these features are near-zero anyway (pure
//     computation + no scroll), so the training/evaluation gap on those
//     channels is small — but it IS a gap.
//   - device_tier is pinned to DEVICE_TIER_DEFAULT (1) so training is
//     reproducible across hosts. Live Puppeteer reads navigator.
//     hardwareConcurrency which varies by machine; pretrained weights from
//     this script assume tier 1.
//   - In-distribution learning only: data and evaluation both sample from
//     the same Part 1 distribution (4 workloads × headless no-vsync regime).
//     Out-of-distribution generalization is explicitly out of scope for
//     Part 2 — the blog post must state this to preempt reviewer objection.

import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import { writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

import { Predictor } from "../src/core/predictor.js";
import { FeatureExtractor } from "../src/core/features.js";
import {
  FRAME_BUDGET_60,
  GRAD_CLIP,
  LR,
  MLP_INPUT_DIM,
  MOMENTUM,
  PARAM_COUNT,
  PRED_LOSS_EPS,
} from "../src/core/constants.js";

// mulberry32 — same implementation used elsewhere in the project. Inlined
// (not imported from tests/helpers/) so this script has no test-tree deps.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Shadow log loading ---------------------------------------------------

/**
 * Stream-parse a shadow JSONL file and return a Map<runIndex, {workload, frames}>
 * containing ONLY the active=B0 rows. Frame order within each run is
 * preserved (JSONL append order = frame order from benchmark.js).
 *
 * @param {string} jsonlPath
 * @returns {Promise<Map<number, {workload: string, frames: Array<{dt: number, miss: boolean}>}>>}
 */
export async function loadB0Runs(jsonlPath) {
  const rl = createInterface({
    input: createReadStream(jsonlPath, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });
  const runs = new Map();
  for await (const line of rl) {
    if (!line) continue;
    const rec = JSON.parse(line);
    if (rec.active !== "B0") continue;
    let run = runs.get(rec.runIndex);
    if (!run) {
      run = { workload: rec.workload, frames: [] };
      runs.set(rec.runIndex, run);
    }
    run.frames.push({ dt: rec.dt, miss: !!rec.miss });
  }
  return runs;
}

// --- Replay: dt sequence → (features, label) pairs -----------------------

/**
 * Replay one run's dt sequence through a FeatureExtractor.
 *
 * Label pairing matches PredictorScheduler's live contract:
 *   decide() at step t captures features[t]; onFrameComplete() at step t+1
 *   pushes (_lastFeatures, miss[t+1]). So samples are
 *   (features[t], miss[t+1]) for t ∈ [0, n-1). The last frame has no next
 *   frame and is dropped.
 *
 * @param {Array<{dt: number, miss: boolean}>} frames — ordered
 * @param {number} [budgetMs=FRAME_BUDGET_60]
 * @returns {{ features: Float32Array, labels: Uint8Array }}
 */
export function replayRun(frames, budgetMs = FRAME_BUDGET_60) {
  // Pin FeatureExtractor config for cross-host reproducibility. See the
  // module-level "Limitations" comment for what this implies about the
  // training/evaluation distribution gap.
  const extractor = new FeatureExtractor(budgetMs, {
    hardwareConcurrency: undefined, // → DEVICE_TIER_DEFAULT (1)
    getMemoryUsed: null,
    viewportH: 1000,
  });

  const n = frames.length;
  if (n < 2) {
    return { features: new Float32Array(0), labels: new Uint8Array(0) };
  }

  const features = new Float32Array((n - 1) * MLP_INPUT_DIM);
  const labels = new Uint8Array(n - 1);

  let writeIdx = 0;
  let lastFeatures = null;
  for (let t = 0; t < n; t++) {
    extractor.observe({ dt: frames[t].dt });
    if (lastFeatures !== null) {
      features.set(lastFeatures, writeIdx * MLP_INPUT_DIM);
      labels[writeIdx] = frames[t].miss ? 1 : 0;
      writeIdx++;
    }
    // extract() reuses an internal buffer — snapshot so the next iteration's
    // overwrite does not silently rewrite our stored features[t].
    lastFeatures = new Float32Array(extractor.extract(0));
  }
  return { features, labels };
}

// --- Offline training -----------------------------------------------------

/**
 * Shuffled-minibatch SGD with momentum + L2 grad clip, starting from a
 * fresh He-initialized Predictor. Intentionally re-implemented here rather
 * than reusing OnlineTrainer: offline training wants shuffle-without-
 * replacement per epoch and unbounded sample storage, while OnlineTrainer's
 * ring buffer (size 1024) + sample-with-replacement semantics are tuned for
 * online learning.
 *
 * A single mulberry32 RNG seeded with `seed` drives BOTH He initialization
 * and per-epoch shuffling. Byte-identical weights across runs require
 * (seed, features, labels, epochs, batchSize) to all match exactly.
 *
 * @param {object} opts
 * @param {Float32Array} opts.features — length N * MLP_INPUT_DIM, flat row-major
 * @param {Uint8Array} opts.labels    — length N
 * @param {number} [opts.epochs=5]
 * @param {number} [opts.batchSize=64]
 * @param {number} [opts.seed=42]
 * @param {number} [opts.lr=LR]
 * @param {number} [opts.momentum=MOMENTUM]
 * @param {number} [opts.gradClip=GRAD_CLIP]
 * @returns {{ params: Float32Array, lossCurve: number[] }}
 */
export function trainOffline({
  features,
  labels,
  epochs = 5,
  batchSize = 64,
  seed = 42,
  lr = LR,
  momentum = MOMENTUM,
  gradClip = GRAD_CLIP,
}) {
  const N = labels.length;
  if (N === 0) throw new Error("trainOffline: empty dataset");
  if (features.length !== N * MLP_INPUT_DIM) {
    throw new Error(
      `trainOffline: features length ${features.length} does not match labels.length * MLP_INPUT_DIM (${N * MLP_INPUT_DIM})`,
    );
  }

  const rng = mulberry32(seed);
  const predictor = new Predictor({ rng });

  const velocity = new Float32Array(PARAM_COUNT);
  const accumGrads = new Float32Array(PARAM_COUNT);
  const indices = new Uint32Array(N);
  for (let i = 0; i < N; i++) indices[i] = i;

  const lossCurve = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    // Fisher-Yates shuffle (in-place).
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }

    let epochLossSum = 0;
    let batchCount = 0;

    for (let b = 0; b < N; b += batchSize) {
      const end = Math.min(b + batchSize, N);
      const B = end - b;
      accumGrads.fill(0);
      let batchLoss = 0;

      for (let s = b; s < end; s++) {
        const idx = indices[s];
        const offset = idx * MLP_INPUT_DIM;
        const x = features.subarray(offset, offset + MLP_INPUT_DIM);
        const target = labels[idx];

        const grads = predictor.backward(x, target);
        for (let j = 0; j < PARAM_COUNT; j++) accumGrads[j] += grads[j];

        const pm = predictor._out.p_miss;
        const pc =
          pm < PRED_LOSS_EPS
            ? PRED_LOSS_EPS
            : pm > 1 - PRED_LOSS_EPS
              ? 1 - PRED_LOSS_EPS
              : pm;
        batchLoss += -(target * Math.log(pc) + (1 - target) * Math.log(1 - pc));
      }

      const inv = 1 / B;
      for (let j = 0; j < PARAM_COUNT; j++) accumGrads[j] *= inv;

      let normSq = 0;
      for (let j = 0; j < PARAM_COUNT; j++) normSq += accumGrads[j] * accumGrads[j];
      const gradNorm = Math.sqrt(normSq);
      if (gradNorm > gradClip) {
        const scale = gradClip / gradNorm;
        for (let j = 0; j < PARAM_COUNT; j++) accumGrads[j] *= scale;
      }

      const params = predictor.params;
      for (let j = 0; j < PARAM_COUNT; j++) {
        velocity[j] = momentum * velocity[j] - lr * accumGrads[j];
        params[j] += velocity[j];
      }

      epochLossSum += batchLoss * inv;
      batchCount++;
    }
    lossCurve.push(epochLossSum / batchCount);
  }

  // Copy out — callers should not have a live handle into Predictor.params.
  return {
    params: new Float32Array(predictor.params),
    lossCurve,
  };
}

// --- Source-data hashing --------------------------------------------------

/**
 * SHA-256 the full JSONL. Saved into PRETRAINED_META.sourceDataSHA256 so
 * future runs of this script can assert that the training data has not
 * been altered since the weights were generated.
 */
export async function hashFile(path) {
  const hash = createHash("sha256");
  const stream = createReadStream(path);
  for await (const chunk of stream) hash.update(chunk);
  return hash.digest("hex");
}

// --- CLI ------------------------------------------------------------------

function parseArgs(argv) {
  const args = {
    seed: 42,
    epochs: 5,
    batchSize: 64,
    input: "shadow.jsonl",
    output: "docs/PHASE5_PART2_WEIGHTS.json",
  };
  for (const a of argv.slice(2)) {
    if (a.startsWith("--seed=")) args.seed = parseInt(a.split("=")[1], 10);
    else if (a.startsWith("--epochs=")) args.epochs = parseInt(a.split("=")[1], 10);
    else if (a.startsWith("--batch=")) args.batchSize = parseInt(a.split("=")[1], 10);
    else if (a.startsWith("--input=")) args.input = a.split("=")[1];
    else if (a.startsWith("--output=")) args.output = a.split("=")[1];
    else throw new Error(`unknown argument: ${a}`);
  }
  return args;
}

async function main() {
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const repoRoot = resolve(scriptDir, "..");
  const args = parseArgs(process.argv);
  const inputPath = resolve(repoRoot, args.input);
  const outputPath = resolve(repoRoot, args.output);

  const log = (m) => process.stderr.write(`[${new Date().toISOString()}] ${m}\n`);

  log(`Loading B0 runs from ${inputPath}`);
  const runs = await loadB0Runs(inputPath);
  const totalFrames = [...runs.values()].reduce((s, r) => s + r.frames.length, 0);
  log(`  ${runs.size} B0 runs, ${totalFrames} raw frames`);

  log(`Replaying through FeatureExtractor (feat dim=${MLP_INPUT_DIM})`);
  const perRun = [];
  let totalSamples = 0;
  for (const [, run] of runs) {
    const r = replayRun(run.frames);
    perRun.push(r);
    totalSamples += r.labels.length;
  }
  const features = new Float32Array(totalSamples * MLP_INPUT_DIM);
  const labels = new Uint8Array(totalSamples);
  let offset = 0;
  for (const r of perRun) {
    features.set(r.features, offset * MLP_INPUT_DIM);
    labels.set(r.labels, offset);
    offset += r.labels.length;
  }
  log(`  ${totalSamples} (features, label) pairs`);

  let pos = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] === 1) pos++;
  log(
    `  class balance: pos=${pos}, neg=${labels.length - pos}, pos_rate=${((pos / labels.length) * 100).toFixed(2)}%`,
  );

  log(`Training: seed=${args.seed}, epochs=${args.epochs}, batch=${args.batchSize}`);
  const t0 = Date.now();
  const { params, lossCurve } = trainOffline({
    features,
    labels,
    epochs: args.epochs,
    batchSize: args.batchSize,
    seed: args.seed,
  });
  const trainMs = Date.now() - t0;
  log(
    `  loss curve: ${lossCurve.map((l) => l.toFixed(6)).join(" → ")}  (${trainMs} ms)`,
  );

  log(`Hashing source data for provenance`);
  const sourceDataSHA256 = await hashFile(inputPath);

  const meta = {
    seed: args.seed,
    epochs: args.epochs,
    batchSize: args.batchSize,
    finalLoss: lossCurve[lossCurve.length - 1],
    lossCurve,
    trainingSamples: totalSamples,
    sourcePath: args.input,
    sourceDataSHA256,
    timestamp: new Date().toISOString(),
  };
  const payload = {
    weights: Array.from(params),
    meta,
  };
  await writeFile(outputPath, JSON.stringify(payload, null, 2) + "\n");
  log(`Wrote ${outputPath}`);
}

// Only execute main() when this file is the entry point. Tests import the
// helpers above and must not trigger a full training run on import.
const invokedAsScript =
  process.argv[1] && process.argv[1].endsWith("generate-pretrained.js");
if (invokedAsScript) {
  main().catch((e) => {
    process.stderr.write(`[generate-pretrained] fatal: ${e?.stack ?? e}\n`);
    process.exit(1);
  });
}
