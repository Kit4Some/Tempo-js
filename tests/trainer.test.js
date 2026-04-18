import { describe, it, expect } from "vitest";
import { OnlineTrainer } from "../src/core/trainer.js";
import { Predictor } from "../src/core/predictor.js";
import { MLP_INPUT_DIM, PARAM_COUNT } from "../src/core/constants.js";
import { mulberry32 } from "./helpers/rng.js";

function featureVec(f0) {
  const v = new Float32Array(MLP_INPUT_DIM);
  v[0] = f0;
  return v;
}

describe("OnlineTrainer — buffer + shape", () => {
  it("trainStep returns null when buffer is empty", () => {
    const p = new Predictor({ rng: mulberry32(1) });
    const t = new OnlineTrainer(p);
    expect(t.trainStep()).toBeNull();
    expect(t.bufferCount()).toBe(0);
  });

  it("push copies features so later caller mutation does not leak into storage", () => {
    const p = new Predictor({ rng: mulberry32(1) });
    const t = new OnlineTrainer(p);
    const x = featureVec(0.5);
    t.push(x, 1);
    x[0] = 999; // caller mutates AFTER push — must not affect stored sample
    expect(t._features[0]).toBe(0.5);
  });

  it("push throws when features length is not MLP_INPUT_DIM", () => {
    const p = new Predictor({ rng: mulberry32(1) });
    const t = new OnlineTrainer(p);
    expect(() => t.push(new Float32Array(11), 0)).toThrow();
  });

  it("trainStep returns { loss, gradNorm } with finite numbers", () => {
    const p = new Predictor({ rng: mulberry32(1) });
    const t = new OnlineTrainer(p);
    t.push(featureVec(0.8), 1);
    const result = t.trainStep(1);
    expect(result).not.toBeNull();
    expect(Number.isFinite(result.loss)).toBe(true);
    expect(Number.isFinite(result.gradNorm)).toBe(true);
    expect(result.gradNorm).toBeGreaterThanOrEqual(0);
  });

  it("gradNorm is reported pre-clip (can exceed gradClip)", () => {
    const p = new Predictor({ rng: mulberry32(1) });
    // Force a very small gradClip; with a sparse input the pre-clip norm
    // should still be a finite number even if it gets clipped in the update.
    const t = new OnlineTrainer(p, { gradClip: 1e-9 });
    for (let i = 0; i < 8; i++) t.push(featureVec(i * 0.1), i > 4 ? 1 : 0);
    const result = t.trainStep(8);
    expect(result.gradNorm).toBeGreaterThan(1e-9);
  });

  it("gradient clipping keeps params update bounded", () => {
    const p = new Predictor({ rng: mulberry32(1) });
    const t = new OnlineTrainer(p, { gradClip: 0.1, lr: 1.0 });
    const before = new Float32Array(p.params); // snapshot
    t.push(featureVec(0.9), 1);
    t.trainStep(1);
    let maxDelta = 0;
    for (let i = 0; i < PARAM_COUNT; i++) {
      const d = Math.abs(p.params[i] - before[i]);
      if (d > maxDelta) maxDelta = d;
    }
    // With clipped grad (L2 ≤ 0.1) and lr=1.0, any single parameter's
    // change is bounded by 0.1 (since L2 of per-entry contributions ≤ 0.1).
    expect(maxDelta).toBeLessThanOrEqual(0.1 + 1e-6);
  });

  it("getAvgLoss returns 0 before any trainStep, then the rolling mean", () => {
    const p = new Predictor({ rng: mulberry32(1) });
    const t = new OnlineTrainer(p);
    expect(t.getAvgLoss()).toBe(0);
    t.push(featureVec(0.5), 1);
    t.trainStep(1);
    expect(t.getAvgLoss()).toBeGreaterThan(0);
  });
});

describe("OnlineTrainer — convergence on a synthetic classifier", () => {
  it(
    "converges to loss < 0.1 on features[0] > 0.5 in 10,000 steps (seed=42)",
    () => {
      const rng = mulberry32(42);
      const predictor = new Predictor({ rng });
      const trainer = new OnlineTrainer(predictor, { rng });
      const x = new Float32Array(MLP_INPUT_DIM);

      for (let step = 0; step < 10000; step++) {
        x.fill(0); // other features stay 0 so the net only learns features[0]
        x[0] = rng();
        const target = x[0] > 0.5 ? 1 : 0;
        trainer.push(x, target);
        trainer.trainStep(16);
      }

      const finalLoss = trainer.getAvgLoss();
      expect(finalLoss).toBeLessThan(0.1);
    },
    30000,
  );
});
