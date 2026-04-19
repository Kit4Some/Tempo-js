import { describe, it, expect } from "vitest";
import {
  replayRun,
  trainOffline,
  loadB0Runs,
  mulberry32,
} from "../scripts/generate-pretrained.js";
import { MLP_INPUT_DIM, PARAM_COUNT } from "../src/core/constants.js";
import { writeFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

describe("replayRun — dt sequence → (features, label) pairs", () => {
  it("emits n-1 pairs for n frames (last frame has no next-frame label)", () => {
    const frames = [
      { dt: 10, miss: false },
      { dt: 12, miss: false },
      { dt: 20, miss: true },
      { dt: 30, miss: true },
    ];
    const { features, labels } = replayRun(frames);
    expect(labels.length).toBe(3);
    expect(features.length).toBe(3 * MLP_INPUT_DIM);
  });

  it("pairs features[t] with miss[t+1] (matches live PredictorScheduler contract)", () => {
    // PredictorScheduler copies features at decide() time and pairs them with
    // the NEXT frame's miss flag at onFrameComplete(). Replay must produce
    // the same (features[t], miss[t+1]) pairing, so labels = miss[1..n-1].
    const frames = [
      { dt: 10, miss: true }, // miss[0] unused — no features[-1]
      { dt: 11, miss: false },
      { dt: 12, miss: true },
      { dt: 13, miss: false },
    ];
    const { labels } = replayRun(frames);
    expect(Array.from(labels)).toEqual([0, 1, 0]);
  });

  it("yields no samples for empty or single-frame input", () => {
    expect(replayRun([]).labels.length).toBe(0);
    expect(replayRun([{ dt: 10, miss: false }]).labels.length).toBe(0);
  });

  it("produces finite features (no NaN propagation from first-frame state)", () => {
    const frames = [];
    for (let i = 0; i < 10; i++) frames.push({ dt: 15 + Math.sin(i), miss: false });
    const { features } = replayRun(frames);
    for (const v of features) expect(Number.isFinite(v)).toBe(true);
  });

  it("feature[0] (dt_ema_fast) tracks normalized dt into [0, 1]", () => {
    // dt=10ms against 16.67ms budget → normalized ≈ 0.6. EMA smoothing
    // keeps the value strictly in (0, 1) after the first observation.
    const frames = [];
    for (let i = 0; i < 5; i++) frames.push({ dt: 10, miss: false });
    const { features } = replayRun(frames);
    expect(features[0]).toBeGreaterThan(0);
    expect(features[0]).toBeLessThan(1);
  });

  it("is deterministic: same frames → byte-identical features", () => {
    const frames = [];
    for (let i = 0; i < 20; i++) {
      frames.push({ dt: 10 + ((i * 7) % 13), miss: i % 3 === 0 });
    }
    const a = replayRun(frames);
    const b = replayRun(frames);
    for (let i = 0; i < a.features.length; i++) {
      expect(a.features[i]).toBe(b.features[i]);
    }
    expect(Array.from(a.labels)).toEqual(Array.from(b.labels));
  });
});

describe("trainOffline — shuffled-minibatch SGD with momentum", () => {
  function makeFixture(n, rng) {
    const features = new Float32Array(n * MLP_INPUT_DIM);
    const labels = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const f0 = rng();
      for (let j = 0; j < MLP_INPUT_DIM; j++) {
        features[i * MLP_INPUT_DIM + j] = j === 0 ? f0 : 0;
      }
      labels[i] = f0 > 0.5 ? 1 : 0;
    }
    return { features, labels };
  }

  it("is bit-exact reproducible: same seed + same data → identical weights", () => {
    // This test is the load-bearing guarantee for PRETRAINED_META:
    //   "given seed=N and sourceDataSHA256=H, you get THIS param vector".
    // If this test ever drifts, the reproducibility claim in the blog post
    // has to be revisited.
    const fx = makeFixture(256, mulberry32(7));
    const a = trainOffline({ ...fx, epochs: 2, batchSize: 16, seed: 42 });
    const b = trainOffline({ ...fx, epochs: 2, batchSize: 16, seed: 42 });
    expect(a.params.length).toBe(PARAM_COUNT);
    for (let i = 0; i < PARAM_COUNT; i++) {
      expect(a.params[i]).toBe(b.params[i]);
    }
    expect(a.lossCurve).toEqual(b.lossCurve);
  });

  it("different seeds diverge (sanity — not stuck in a fixed point)", () => {
    const fx = makeFixture(256, mulberry32(7));
    const a = trainOffline({ ...fx, epochs: 1, batchSize: 16, seed: 1 });
    const b = trainOffline({ ...fx, epochs: 1, batchSize: 16, seed: 2 });
    let differing = 0;
    for (let i = 0; i < PARAM_COUNT; i++) {
      if (a.params[i] !== b.params[i]) differing++;
    }
    expect(differing).toBeGreaterThan(100);
  });

  it("loss decreases across epochs on a learnable signal", () => {
    const fx = makeFixture(512, mulberry32(7));
    const { lossCurve } = trainOffline({
      ...fx,
      epochs: 4,
      batchSize: 32,
      seed: 42,
    });
    expect(lossCurve.length).toBe(4);
    expect(lossCurve[3]).toBeLessThan(lossCurve[0]);
  });

  it("throws on empty dataset", () => {
    expect(() =>
      trainOffline({
        features: new Float32Array(0),
        labels: new Uint8Array(0),
        epochs: 1,
        batchSize: 16,
        seed: 42,
      }),
    ).toThrow();
  });

  it("throws when features / labels length disagree", () => {
    expect(() =>
      trainOffline({
        features: new Float32Array(10 * MLP_INPUT_DIM),
        labels: new Uint8Array(11),
        epochs: 1,
        batchSize: 4,
        seed: 42,
      }),
    ).toThrow();
  });
});

describe("loadB0Runs — JSONL filter + runIndex grouping", () => {
  it("keeps active=B0 only, groups by runIndex, preserves frame order", async () => {
    const tmpPath = resolve(tmpdir(), `tempo-loadB0-${Date.now()}.jsonl`);
    const lines = [
      { runIndex: 1, active: "B0", workload: "constant", frameIdx: 0, dt: 10, miss: false, decisions: {} },
      { runIndex: 1, active: "B0", workload: "constant", frameIdx: 1, dt: 11, miss: false, decisions: {} },
      { runIndex: 2, active: "B1", workload: "burst",    frameIdx: 0, dt: 12, miss: false, decisions: {} },
      { runIndex: 3, active: "B0", workload: "scroll",   frameIdx: 0, dt: 14, miss: true,  decisions: {} },
      { runIndex: 1, active: "B0", workload: "constant", frameIdx: 2, dt: 13, miss: true,  decisions: {} },
    ];
    writeFileSync(tmpPath, lines.map((l) => JSON.stringify(l)).join("\n"));
    try {
      const runs = await loadB0Runs(tmpPath);
      expect(runs.size).toBe(2);
      expect(runs.get(1).workload).toBe("constant");
      expect(runs.get(1).frames.map((f) => f.dt)).toEqual([10, 11, 13]);
      expect(runs.get(1).frames.map((f) => f.miss)).toEqual([false, false, true]);
      expect(runs.get(3).frames).toHaveLength(1);
      expect(runs.has(2)).toBe(false);
    } finally {
      unlinkSync(tmpPath);
    }
  });
});
