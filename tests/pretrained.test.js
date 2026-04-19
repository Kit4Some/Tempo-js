import { describe, it, expect } from "vitest";
import {
  PRETRAINED_WEIGHTS,
  PRETRAINED_META,
} from "../src/harness/pretrained.js";
import { Predictor } from "../src/core/predictor.js";
import { MLP_INPUT_DIM, PARAM_COUNT } from "../src/core/constants.js";

// These tests guard the inlined artifact's SHAPE — not its numeric contents.
// Re-running scripts/generate-pretrained.js legitimately changes every value,
// so asserting specific weights here would turn the test suite into a brake
// on intentional regeneration. What IS load-bearing: the array has the
// right length, META keys are present, and the weights actually load into
// the Predictor and produce usable predictions.

describe("pretrained.js — inlined artifact shape", () => {
  it("PRETRAINED_WEIGHTS is a Float32Array of length PARAM_COUNT (353)", () => {
    expect(PRETRAINED_WEIGHTS).toBeInstanceOf(Float32Array);
    expect(PRETRAINED_WEIGHTS.length).toBe(PARAM_COUNT);
  });

  it("PRETRAINED_META exposes provenance fields", () => {
    expect(PRETRAINED_META).toBeDefined();
    expect(typeof PRETRAINED_META.seed).toBe("number");
    expect(typeof PRETRAINED_META.epochs).toBe("number");
    expect(typeof PRETRAINED_META.batchSize).toBe("number");
    expect(typeof PRETRAINED_META.finalLoss).toBe("number");
    expect(Array.isArray(PRETRAINED_META.lossCurve)).toBe(true);
    expect(PRETRAINED_META.lossCurve.length).toBe(PRETRAINED_META.epochs);
    expect(typeof PRETRAINED_META.trainingSamples).toBe("number");
    expect(typeof PRETRAINED_META.sourceDataSHA256).toBe("string");
    expect(PRETRAINED_META.sourceDataSHA256).toMatch(/^[0-9a-f]{64}$/);
    expect(typeof PRETRAINED_META.timestamp).toBe("string");
  });

  it("loss curve is monotonic-ish (last epoch below first)", () => {
    // Exact monotonicity is not required — SGD with random shuffles can
    // have small upward blips. What matters is that 5 epochs produced a
    // net loss reduction, i.e. the optimizer actually did something.
    const c = PRETRAINED_META.lossCurve;
    expect(c[c.length - 1]).toBeLessThan(c[0]);
  });
});

describe("pretrained.js — Predictor.loadPretrained integration", () => {
  it("loads into a Predictor without throwing", () => {
    const p = new Predictor();
    expect(() => p.loadPretrained(PRETRAINED_WEIGHTS)).not.toThrow();
  });

  it("produces finite p_miss values for a representative input", () => {
    const p = new Predictor();
    p.loadPretrained(PRETRAINED_WEIGHTS);
    const x = new Float32Array(MLP_INPUT_DIM);
    x[0] = 0.8; // dt_ema_fast
    x[1] = 0.6; // dt_ema_slow
    x[3] = 1.0; // dt_max / budget
    x[11] = 1; // device_tier
    const out = p.forward(x);
    expect(Number.isFinite(out.p_miss)).toBe(true);
    expect(out.p_miss).toBeGreaterThanOrEqual(0);
    expect(out.p_miss).toBeLessThanOrEqual(1);
  });

  it("distinguishes low-jank vs high-jank inputs (training actually learned something)", () => {
    // This is a sanity check on the pretrained weights: given two inputs
    // that differ in the obvious "is the budget being exceeded" direction,
    // the pretrained model should produce different p_miss. We do NOT
    // prescribe which direction — B0's labels are what they are — but a
    // learned model should not be input-invariant.
    const p = new Predictor();
    p.loadPretrained(PRETRAINED_WEIGHTS);

    const calm = new Float32Array(MLP_INPUT_DIM);
    calm[0] = 0.1;
    calm[1] = 0.1;
    calm[3] = 0.2;
    const calmOut = p.forward(calm).p_miss;

    const spiky = new Float32Array(MLP_INPUT_DIM);
    spiky[0] = 1.5;
    spiky[1] = 1.2;
    spiky[3] = 2.0;
    spiky[4] = 0.5; // 50% recent miss rate
    const spikyOut = p.forward(spiky).p_miss;

    // Distinguishable by at least 0.05 p_miss. Tight enough to catch a
    // broken-weights regression, loose enough to survive re-training.
    expect(Math.abs(calmOut - spikyOut)).toBeGreaterThan(0.05);
  });
});
