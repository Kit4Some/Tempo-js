import { describe, it, expect } from "vitest";
import { B0_AlwaysFull, B1_EmaThreshold } from "../src/core/baselines.js";
import {
  B1_DEGRADE_RATIO,
  B1_EMA_ALPHA,
  B1_REDUCE_RATIO,
  FRAME_BUDGET_60,
} from "../src/core/constants.js";

function featureVec(f0) {
  const v = new Float32Array(12);
  v[0] = f0;
  return v;
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
