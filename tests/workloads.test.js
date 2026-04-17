import { describe, it, expect } from "vitest";
import {
  constant,
  sawtooth,
  burst,
  scrollCorrelated,
} from "../src/harness/workloads.js";

function measure(fn, frameIdx) {
  const start = performance.now();
  fn(frameIdx);
  return performance.now() - start;
}

const TOL_MS = 5;

function expectNear(actual, target) {
  expect(actual).toBeGreaterThanOrEqual(Math.max(0, target - 1));
  expect(actual).toBeLessThanOrEqual(target + TOL_MS);
}

describe("constant", () => {
  it("burns approximately ms per call", () => {
    const w = constant(5);
    for (let i = 0; i < 3; i++) {
      expectNear(measure(w, i), 5);
    }
  });

  it("accepts 0 ms (essentially no-op)", () => {
    const w = constant(0);
    expect(measure(w, 0)).toBeLessThan(2);
  });
});

describe("sawtooth", () => {
  it("ramps linearly from min to max over periodFrames and wraps", () => {
    const w = sawtooth(0, 10, 4);
    expectNear(measure(w, 0), 0);
    expectNear(measure(w, 1), 2.5);
    expectNear(measure(w, 2), 5);
    expectNear(measure(w, 3), 7.5);
    expectNear(measure(w, 4), 0); // wraps
  });
});

describe("burst", () => {
  it("stays at base load, spikes for spikeDurationFrames every spikeEveryFrames", () => {
    const w = burst(2, 12, 10, 2);
    expectNear(measure(w, 0), 2);
    expectNear(measure(w, 9), 2);
    expectNear(measure(w, 10), 12); // spike start
    expectNear(measure(w, 11), 12); // spike
    expectNear(measure(w, 12), 2); // back to base
    expectNear(measure(w, 20), 12); // next spike
  });
});

describe("scrollCorrelated", () => {
  it("scales linearly with velocity through k", () => {
    let v = 0;
    const w = scrollCorrelated(() => v, 0.3);
    expect(measure(w, 0)).toBeLessThan(2);
    v = 20; // 20 * 0.3 = 6ms
    expectNear(measure(w, 1), 6);
    v = 40; // 12ms
    expectNear(measure(w, 2), 12);
  });
});
