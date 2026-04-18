import { describe, it, expect } from "vitest";
import { FeatureExtractor } from "../src/core/features.js";
import {
  DT_EMA_FAST_ALPHA,
  DT_WINDOW_MISS,
  FRAME_BUDGET_60,
  MLP_INPUT_DIM,
} from "../src/core/constants.js";

describe("FeatureExtractor — observe field semantics", () => {
  it("throws when dt is missing or non-finite", () => {
    const fe = new FeatureExtractor();
    expect(() => fe.observe({})).toThrow();
    expect(() => fe.observe({ dt: undefined })).toThrow();
    expect(() => fe.observe({ dt: NaN })).toThrow();
    expect(() => fe.observe({ dt: Infinity })).toThrow();
  });

  it("state-like fields (domMutations, visibleAnimating) preserve across frames", () => {
    const fe = new FeatureExtractor();
    fe.observe({ dt: 10, domMutations: 5, visibleAnimating: 3 });
    fe.observe({ dt: 10 }); // omit state-like
    const out = fe.extract();
    expect(out[8]).toBeCloseTo(Math.log(4), 5); // visible_animating: log(1+3)
    expect(out[9]).toBeCloseTo(Math.log(6), 5); // dom_mutation_rate: log(1+5)
  });

  it("rate-like fields (inputEvents, scrollVelocity) default to 0 when omitted", () => {
    const fe = new FeatureExtractor(FRAME_BUDGET_60, { viewportH: 100 });
    fe.observe({ dt: 10, scrollVelocity: 50, inputEvents: 5 });
    fe.observe({ dt: 10 }); // omit rate-like → 0
    const out = fe.extract();
    expect(out[7]).toBe(0); // scroll_velocity
    // input_activity EMA decayed toward 0 after one 0-observation
    expect(out[6]).toBeLessThan(5 / 10);
  });
});

describe("FeatureExtractor — normalization", () => {
  it("dt_ema_fast and dt_ema_slow converge to dt/budget after many frames", () => {
    const fe = new FeatureExtractor(FRAME_BUDGET_60);
    const dt = 12;
    for (let i = 0; i < 500; i++) fe.observe({ dt });
    const out = fe.extract();
    expect(out[0]).toBeCloseTo(dt / FRAME_BUDGET_60, 3);
    expect(out[1]).toBeCloseTo(dt / FRAME_BUDGET_60, 2);
  });

  it("miss_rate_32: 0 when every frame is fast, 1 when every frame is slow", () => {
    let fe = new FeatureExtractor(FRAME_BUDGET_60);
    for (let i = 0; i < DT_WINDOW_MISS; i++) fe.observe({ dt: 5 });
    expect(fe.extract()[4]).toBe(0);

    fe = new FeatureExtractor(FRAME_BUDGET_60);
    for (let i = 0; i < DT_WINDOW_MISS; i++) fe.observe({ dt: 50 });
    expect(fe.extract()[4]).toBe(1);
  });

  it("dt_max_8 and dt_var capture the most recent 8-frame window", () => {
    const fe = new FeatureExtractor(10);
    for (let i = 0; i < 7; i++) fe.observe({ dt: 5 });
    fe.observe({ dt: 30 }); // newest frame is the spike
    const out = fe.extract();
    expect(out[3]).toBeCloseTo(30 / 10, 5); // dt_max_8 normalized
    expect(out[2]).toBeGreaterThan(0); // variance non-zero after spike
  });

  it("device_tier maps hardwareConcurrency into {0,1,2}", () => {
    const make = (hc) => new FeatureExtractor(FRAME_BUDGET_60, {
      hardwareConcurrency: hc,
    });
    const observeAnd = (fe) => {
      fe.observe({ dt: 10 });
      return fe.extract();
    };
    expect(observeAnd(make(2))[11]).toBe(0);
    expect(observeAnd(make(4))[11]).toBe(1);
    expect(observeAnd(make(8))[11]).toBe(1);
    expect(observeAnd(make(16))[11]).toBe(2);
    expect(observeAnd(make(undefined))[11]).toBe(1); // default tier
  });

  it("scroll_velocity uses abs divided by viewport height", () => {
    const fe = new FeatureExtractor(FRAME_BUDGET_60, { viewportH: 100 });
    fe.observe({ dt: 10, scrollVelocity: -30 });
    expect(fe.extract()[7]).toBeCloseTo(0.3, 5);
  });

  it("workload_delta parameter of extract() normalizes by budget", () => {
    const fe = new FeatureExtractor(FRAME_BUDGET_60);
    fe.observe({ dt: 10 });
    expect(fe.extract(10)[10]).toBeCloseTo(10 / FRAME_BUDGET_60, 5);
  });

  it("gc_pressure is 0 when getMemoryUsed is explicitly null (non-Chrome)", () => {
    const fe = new FeatureExtractor(FRAME_BUDGET_60, { getMemoryUsed: null });
    fe.observe({ dt: 10 });
    expect(fe.extract()[5]).toBe(0);
  });

  it("gc_pressure uses log(1 + positive delta) when the heap grows", () => {
    let mem = 1000;
    const fe = new FeatureExtractor(FRAME_BUDGET_60, {
      getMemoryUsed: () => mem,
    });
    fe.observe({ dt: 10 });
    fe.extract(); // seeds _lastMemory = 1000
    mem = 1000 + 1_000_000;
    fe.observe({ dt: 10 });
    expect(fe.extract()[5]).toBeCloseTo(Math.log(1 + 1_000_000), 3);
  });

  it("gc_pressure ignores heap shrinkage (post-GC) instead of going negative", () => {
    let mem = 5_000_000;
    const fe = new FeatureExtractor(FRAME_BUDGET_60, {
      getMemoryUsed: () => mem,
    });
    fe.observe({ dt: 10 });
    fe.extract();
    mem = 1_000_000; // heap drops after GC
    fe.observe({ dt: 10 });
    expect(fe.extract()[5]).toBe(0);
  });

  it("input_activity is bounded in [0, 1] even under a flood of events", () => {
    const fe = new FeatureExtractor(FRAME_BUDGET_60);
    for (let i = 0; i < 100; i++) fe.observe({ dt: 10, inputEvents: 50 });
    expect(fe.extract()[6]).toBeLessThanOrEqual(1);
    expect(fe.extract()[6]).toBeGreaterThan(0.5);
  });
});

describe("FeatureExtractor — output shape & reuse", () => {
  it("extract returns Float32Array of length MLP_INPUT_DIM (12)", () => {
    const fe = new FeatureExtractor();
    fe.observe({ dt: 10 });
    const out = fe.extract();
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(MLP_INPUT_DIM);
  });

  it("extract() reuses the same Float32Array across calls", () => {
    const fe = new FeatureExtractor();
    fe.observe({ dt: 10 });
    expect(fe.extract()).toBe(fe.extract());
  });
});

describe("FeatureExtractor — EMA alpha sanity", () => {
  it("dt_ema_fast matches an analytically computed EMA sequence", () => {
    const fe = new FeatureExtractor(FRAME_BUDGET_60);
    const alpha = DT_EMA_FAST_ALPHA;
    const dts = [8, 16, 24, 12, 10];
    let expected = 0;
    for (const dt of dts) {
      const n = dt / FRAME_BUDGET_60;
      expected = alpha * n + (1 - alpha) * expected;
      fe.observe({ dt });
    }
    expect(fe.extract()[0]).toBeCloseTo(expected, 5);
  });
});
