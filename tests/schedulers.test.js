import { describe, it, expect } from "vitest";
import {
  B0_AlwaysFull,
  B1_EmaThreshold,
  PredictorScheduler,
} from "../src/core/schedulers.js";
import {
  B1_DEGRADE_RATIO,
  B1_EMA_ALPHA,
  B1_REDUCE_RATIO,
  FRAME_BUDGET_60,
  PRED_DEGRADE_THRESHOLD,
  PRED_REDUCE_THRESHOLD,
} from "../src/core/constants.js";
import { Predictor } from "../src/core/predictor.js";
import { OnlineTrainer } from "../src/core/trainer.js";
import { mulberry32 } from "./helpers/rng.js";

function featureVec(f0) {
  const v = new Float32Array(12);
  v[0] = f0;
  return v;
}

/**
 * Minimal Predictor stub that returns a fixed p_miss. Lets threshold tests
 * drive the decision without fighting Float32 rounding in a real sigmoid.
 */
function mockPredictor(fixedPMiss) {
  return {
    _out: { p_miss: fixedPMiss },
    forward(_x) {
      return this._out;
    },
  };
}

function makeRealScheduler() {
  const predictor = new Predictor({ rng: mulberry32(1) });
  const trainer = new OnlineTrainer(predictor, { rng: mulberry32(1) });
  const scheduler = new PredictorScheduler({ predictor, trainer });
  return { predictor, trainer, scheduler };
}

describe("B0_AlwaysFull", () => {
  it("returns 'full' regardless of features", () => {
    const s = new B0_AlwaysFull();
    expect(s.decide(featureVec(0))).toBe("full");
    expect(s.decide(featureVec(5))).toBe("full");
    expect(s.decide(featureVec(-1))).toBe("full");
  });

  it("returns 'full' regardless of options (including priority='high')", () => {
    const s = new B0_AlwaysFull();
    expect(s.decide(featureVec(0))).toBe("full");
    expect(s.decide(featureVec(0), { priority: "high" })).toBe("full");
    expect(s.decide(featureVec(0), { priority: "low" })).toBe("full");
    expect(s.decide(featureVec(0), { tag: "test" })).toBe("full");
  });

  it("onFrameComplete is a no-op and must not throw", () => {
    const s = new B0_AlwaysFull();
    expect(() => s.onFrameComplete(10, false)).not.toThrow();
    expect(() => s.onFrameComplete(50, true)).not.toThrow();
  });
});

describe("B1_EmaThreshold — decide rule", () => {
  it("returns 'full' when features[0] is clearly below reduceRatio", () => {
    const s = new B1_EmaThreshold();
    expect(s.decide(featureVec(0))).toBe("full");
    expect(s.decide(featureVec(0.5))).toBe("full");
    // Slightly below reduceRatio — Float32 round-trip safe.
    expect(s.decide(featureVec(B1_REDUCE_RATIO - 0.01))).toBe("full");
  });

  it("returns 'reduce' when features[0] is between reduce and degrade", () => {
    const s = new B1_EmaThreshold();
    expect(s.decide(featureVec(B1_REDUCE_RATIO + 0.01))).toBe("reduce");
    expect(s.decide(featureVec(1.0))).toBe("reduce");
    // Slightly below degradeRatio — Float32 round-trip safe.
    expect(s.decide(featureVec(B1_DEGRADE_RATIO - 0.01))).toBe("reduce");
  });

  it("returns 'degrade' when features[0] > degradeRatio", () => {
    const s = new B1_EmaThreshold();
    expect(s.decide(featureVec(B1_DEGRADE_RATIO + 0.01))).toBe("degrade");
    expect(s.decide(featureVec(2.0))).toBe("degrade");
    expect(s.decide(featureVec(10))).toBe("degrade");
  });

  it("honours constructor overrides for reduceRatio and degradeRatio", () => {
    const s = new B1_EmaThreshold(0.3, FRAME_BUDGET_60, 0.5, 0.9);
    expect(s.decide(featureVec(0.4))).toBe("full");
    expect(s.decide(featureVec(0.6))).toBe("reduce");
    expect(s.decide(featureVec(1.0))).toBe("degrade");
  });
});

describe("B1_EmaThreshold — priority override", () => {
  it("options.priority='high' forces 'full' regardless of features", () => {
    const s = new B1_EmaThreshold();
    // Without override, these would be reduce/degrade.
    expect(s.decide(featureVec(1.0), { priority: "high" })).toBe("full");
    expect(s.decide(featureVec(5.0), { priority: "high" })).toBe("full");
  });

  it("options.priority='low' or 'normal' does not alter the rule", () => {
    const s = new B1_EmaThreshold();
    expect(s.decide(featureVec(1.0), { priority: "normal" })).toBe("reduce");
    expect(s.decide(featureVec(1.0), { priority: "low" })).toBe("reduce");
    expect(s.decide(featureVec(5.0), { priority: "normal" })).toBe("degrade");
  });

  it("options.tag is pure metadata and MUST NOT change the decision", () => {
    const s = new B1_EmaThreshold();
    expect(s.decide(featureVec(0.5), { tag: "foo" })).toBe("full");
    expect(s.decide(featureVec(1.0), { tag: "bar" })).toBe("reduce");
    expect(s.decide(featureVec(2.0), { tag: "baz" })).toBe("degrade");
  });
});

describe("B1_EmaThreshold — EMA update", () => {
  it("updates EMA per ema_t = alpha*dt + (1-alpha)*ema_{t-1}, ema_0 = 0", () => {
    const alpha = B1_EMA_ALPHA; // 0.3
    const s = new B1_EmaThreshold(alpha);
    expect(s.getEma()).toBe(0);

    // Analytic: e1 = 0.3*10 + 0.7*0 = 3
    s.onFrameComplete(10, false);
    expect(s.getEma()).toBeCloseTo(3, 10);

    // e2 = 0.3*10 + 0.7*3 = 5.1
    s.onFrameComplete(10, false);
    expect(s.getEma()).toBeCloseTo(5.1, 10);

    // e3 = 0.3*10 + 0.7*5.1 = 6.57
    s.onFrameComplete(10, false);
    expect(s.getEma()).toBeCloseTo(6.57, 10);
  });

  it("matches analytic expected value within 1e-6 for a random sequence", () => {
    const alpha = 0.3;
    const s = new B1_EmaThreshold(alpha);

    const dts = [14.5, 17.2, 22.8, 9.1, 12.0, 28.5, 11.3, 15.0, 16.7, 18.9];
    let expected = 0;
    for (const dt of dts) {
      expected = alpha * dt + (1 - alpha) * expected;
      s.onFrameComplete(dt, dt > FRAME_BUDGET_60);
    }
    expect(Math.abs(s.getEma() - expected)).toBeLessThan(1e-6);
  });

  it("updates EMA irrespective of decide() being called — shadow contract", () => {
    const s = new B1_EmaThreshold(0.3);
    // Only onFrameComplete is called; decide is never invoked.
    // EMA must still advance (shadow-mode requirement).
    for (let i = 0; i < 5; i++) s.onFrameComplete(10, false);
    expect(s.getEma()).toBeGreaterThan(0);
  });
});

describe("PredictorScheduler — threshold mapping", () => {
  // Use a mock predictor so p_miss is set exactly. A real Predictor's sigmoid
  // at a float32-rounded b3 can drift a few 1e-8 off the target, which would
  // make exact-boundary assertions against strict `>` flaky.
  function schedWithP(pMiss) {
    const predictor = mockPredictor(pMiss);
    // Trainer is unused by decide(); pass a real one to satisfy the contract.
    const real = new Predictor({ rng: mulberry32(1) });
    const trainer = new OnlineTrainer(real, { rng: mulberry32(1) });
    return new PredictorScheduler({ predictor, trainer });
  }

  it("p_miss clearly below PRED_REDUCE_THRESHOLD → 'full'", () => {
    expect(schedWithP(0.05).decide(featureVec(0))).toBe("full");
    expect(schedWithP(0).decide(featureVec(0))).toBe("full");
  });

  it("PRED_REDUCE_THRESHOLD < p_miss ≤ PRED_DEGRADE_THRESHOLD → 'reduce'", () => {
    expect(schedWithP(0.2).decide(featureVec(0))).toBe("reduce");
  });

  it("p_miss > PRED_DEGRADE_THRESHOLD → 'degrade'", () => {
    expect(schedWithP(0.5).decide(featureVec(0))).toBe("degrade");
    expect(schedWithP(0.99).decide(featureVec(0))).toBe("degrade");
  });

  it("exact boundary p_miss = PRED_REDUCE_THRESHOLD (0.1) → 'full' (strict >)", () => {
    expect(schedWithP(PRED_REDUCE_THRESHOLD).decide(featureVec(0))).toBe("full");
  });

  it("exact boundary p_miss = PRED_DEGRADE_THRESHOLD (0.3) → 'reduce' (strict >)", () => {
    expect(schedWithP(PRED_DEGRADE_THRESHOLD).decide(featureVec(0))).toBe(
      "reduce",
    );
  });
});

describe("PredictorScheduler — priority override", () => {
  it("options.priority='high' forces 'full' regardless of p_miss", () => {
    const predictor = mockPredictor(0.9);
    const real = new Predictor({ rng: mulberry32(1) });
    const trainer = new OnlineTrainer(real, { rng: mulberry32(1) });
    const s = new PredictorScheduler({ predictor, trainer });
    expect(s.decide(featureVec(0), { priority: "high" })).toBe("full");
  });

  it("priority='normal' or missing does not alter the rule", () => {
    const predictor = mockPredictor(0.5);
    const real = new Predictor({ rng: mulberry32(1) });
    const trainer = new OnlineTrainer(real, { rng: mulberry32(1) });
    const s = new PredictorScheduler({ predictor, trainer });
    expect(s.decide(featureVec(0))).toBe("degrade");
    expect(s.decide(featureVec(0), { priority: "normal" })).toBe("degrade");
    expect(s.decide(featureVec(0), { priority: "low" })).toBe("degrade");
  });

  it("options.tag is metadata and MUST NOT affect the decision", () => {
    const predictor = mockPredictor(0.2);
    const real = new Predictor({ rng: mulberry32(1) });
    const trainer = new OnlineTrainer(real, { rng: mulberry32(1) });
    const s = new PredictorScheduler({ predictor, trainer });
    expect(s.decide(featureVec(0), { tag: "abc" })).toBe("reduce");
    expect(s.decide(featureVec(0), { tag: "xyz" })).toBe("reduce");
  });

  it("priority='high' still records features for onFrameComplete (subtle)", () => {
    // Even though we early-return 'full' without calling predictor.forward,
    // we must have captured the feature vector so a subsequent
    // onFrameComplete pushes the right sample into the trainer.
    const { scheduler, trainer } = makeRealScheduler();
    const x = featureVec(0.77);
    scheduler.decide(x, { priority: "high" });
    scheduler.onFrameComplete(25, true);
    expect(trainer.bufferCount()).toBe(1);
    expect(trainer._features[0]).toBeCloseTo(0.77, 6);
  });
});

describe("PredictorScheduler — features copy (reference-leak guard)", () => {
  it("mutating the caller's features after decide() does not affect the stored sample", () => {
    const { scheduler, trainer } = makeRealScheduler();
    const x = featureVec(0.42);
    scheduler.decide(x);
    x[0] = 999; // caller mutates after decide — MUST NOT leak
    scheduler.onFrameComplete(20, true);
    expect(trainer.bufferCount()).toBe(1);
    expect(trainer._features[0]).toBeCloseTo(0.42, 6);
  });
});

describe("PredictorScheduler — onFrameComplete → trainer.push", () => {
  it("pushes (last features, wasMiss ? 1 : 0) once per decide+complete pair", () => {
    const { scheduler, trainer } = makeRealScheduler();
    const x = featureVec(0.61);
    scheduler.decide(x);
    scheduler.onFrameComplete(30, true);
    expect(trainer.bufferCount()).toBe(1);
    expect(trainer._features[0]).toBeCloseTo(0.61, 6);
    expect(trainer._targets[0]).toBe(1);
  });

  it("wasMiss=false pushes target=0", () => {
    const { scheduler, trainer } = makeRealScheduler();
    scheduler.decide(featureVec(0.3));
    scheduler.onFrameComplete(10, false);
    expect(trainer._targets[0]).toBe(0);
  });
});

describe("PredictorScheduler — first-frame guard", () => {
  it("onFrameComplete without a prior decide() does NOT push a sample", () => {
    const { scheduler, trainer } = makeRealScheduler();
    scheduler.onFrameComplete(10, false);
    expect(trainer.bufferCount()).toBe(0);
  });

  it("after the first decide(), onFrameComplete pushes normally", () => {
    const { scheduler, trainer } = makeRealScheduler();
    scheduler.onFrameComplete(10, false); // ignored (no features yet)
    scheduler.decide(featureVec(0.4));
    scheduler.onFrameComplete(10, false); // now records
    expect(trainer.bufferCount()).toBe(1);
  });
});

describe("PredictorScheduler — shadow-mode integration", () => {
  it("three schedulers fed the same features+dt all advance their state", () => {
    // Simulates a Sequential+Shadow benchmark tick: one features/dt stream,
    // three schedulers each call decide() + onFrameComplete() in lockstep.
    const predictor = new Predictor({ rng: mulberry32(7) });
    const trainer = new OnlineTrainer(predictor, { rng: mulberry32(7) });
    const pred = new PredictorScheduler({ predictor, trainer });
    const b0 = new B0_AlwaysFull();
    const b1 = new B1_EmaThreshold();

    const x = new Float32Array(12);
    for (let i = 0; i < 32; i++) {
      x[0] = (i % 5) * 0.25;
      const dt = (i % 5) * 6 + 8; // varies 8..32ms
      const wasMiss = dt > FRAME_BUDGET_60;
      // Decide and complete for every scheduler — contract §4 Phase 2.
      b0.decide(x);
      b0.onFrameComplete(dt, wasMiss);
      b1.decide(x);
      b1.onFrameComplete(dt, wasMiss);
      pred.decide(x);
      pred.onFrameComplete(dt, wasMiss);
    }

    // Predictor trainer accumulated 32 samples even though only one
    // scheduler is "active" at a time in real benchmark semantics.
    expect(trainer.bufferCount()).toBe(32);
    // B1 EMA advanced.
    expect(b1.getEma()).toBeGreaterThan(0);
  });
});
