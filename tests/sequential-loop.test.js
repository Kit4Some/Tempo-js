import { describe, it, expect, vi } from "vitest";
import { SequentialLoop } from "../src/harness/sequential-loop.js";
import { Predictor } from "../src/core/predictor.js";
import { OnlineTrainer } from "../src/core/trainer.js";
import { FeatureExtractor } from "../src/core/features.js";
import {
  B0_AlwaysFull,
  B1_EmaThreshold,
  PredictorScheduler,
} from "../src/core/schedulers.js";
import { mulberry32 } from "./helpers/rng.js";
import { FRAME_BUDGET_60, PARAM_COUNT } from "../src/core/constants.js";

// Default clock — tests advance this manually via busyWait.
function makeFakeClock() {
  let t = 0;
  return {
    now: () => t,
    busyWait: (ms) => {
      if (ms > 0) t += ms;
    },
    peek: () => t,
  };
}

// Real-scheduler factory with injected seeded RNG. reset() re-invokes this,
// so re-initialization is from the same seed → deterministic.
function makeRealFactory(seed = 42) {
  return () => {
    const rng = mulberry32(seed);
    const predictor = new Predictor({ rng });
    const trainer = new OnlineTrainer(predictor, { rng });
    const extractor = new FeatureExtractor();
    return {
      schedulers: {
        B0: new B0_AlwaysFull(),
        B1: new B1_EmaThreshold(),
        Predictor: new PredictorScheduler({ predictor, trainer }),
      },
      extractor,
      trainer,
      predictor,
    };
  };
}

// Mock-scheduler factory for decision-control tests. Returns schedulers
// whose decide() is a spy returning a configured decision.
function makeMockFactory({ decisions = { B0: "full", B1: "full", Predictor: "full" } } = {}) {
  return () => {
    const extractor = new FeatureExtractor();
    const spies = {
      B0: vi.fn(() => decisions.B0),
      B1: vi.fn(() => decisions.B1),
      Predictor: vi.fn(() => decisions.Predictor),
    };
    const onFrameSpies = {
      B0: vi.fn(),
      B1: vi.fn(),
      Predictor: vi.fn(),
    };
    const schedulers = {
      B0: { decide: spies.B0, onFrameComplete: onFrameSpies.B0 },
      B1: { decide: spies.B1, onFrameComplete: onFrameSpies.B1 },
      Predictor: { decide: spies.Predictor, onFrameComplete: onFrameSpies.Predictor },
    };
    // Minimal trainer / predictor stubs — tests that hit trainTick use the
    // real factory instead.
    const trainer = {
      trainStep: vi.fn(() => null),
      bufferCount: () => 0,
    };
    const predictor = { params: new Float32Array(PARAM_COUNT) };
    return {
      schedulers,
      extractor,
      trainer,
      predictor,
      _spies: spies,
      _onFrameSpies: onFrameSpies,
    };
  };
}

describe("SequentialLoop — shared-features invariant", () => {
  it("passes the exact same Float32Array reference to every scheduler's decide()", () => {
    const clock = makeFakeClock();
    const factory = makeMockFactory();
    const loop = new SequentialLoop({
      buildState: factory,
      workload: () => 5,
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "Predictor",
    });
    loop.step();
    const state = loop._state;
    const callFeatures = {
      B0: state._spies.B0.mock.calls[0][0],
      B1: state._spies.B1.mock.calls[0][0],
      Predictor: state._spies.Predictor.mock.calls[0][0],
    };
    // Identity (===), not just equality.
    expect(callFeatures.B0).toBe(callFeatures.B1);
    expect(callFeatures.B1).toBe(callFeatures.Predictor);
    // Also must be the extractor's preallocated output buffer.
    expect(callFeatures.B0).toBe(state.extractor._out);
  });
});

describe("SequentialLoop — active-only cost application", () => {
  it("applies workCostFor only to the active scheduler's decision", () => {
    const clock = makeFakeClock();
    // Predictor says 'reduce' (0.7x), B0 says 'degrade' (0.35x), B1 says 'degrade' (0.35x).
    // active=Predictor, so cost = 0.7 × base, not 0.35.
    const factory = makeMockFactory({
      decisions: { B0: "degrade", B1: "degrade", Predictor: "reduce" },
    });
    const loop = new SequentialLoop({
      buildState: factory,
      workload: () => 10, // base 10 ms
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "Predictor",
    });
    loop.step(); // step 0: dt=0 (no previous), busyWait(10 * 0.7) = 7
    const step2 = loop.step(); // step 1: dt should reflect the prior 7 ms
    expect(step2.dt).toBeCloseTo(7, 5);
  });

  it("follows the new active's decision after setActive", () => {
    const clock = makeFakeClock();
    const factory = makeMockFactory({
      decisions: { B0: "full", B1: "degrade", Predictor: "reduce" },
    });
    const loop = new SequentialLoop({
      buildState: factory,
      workload: () => 10,
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "B0",
    });
    loop.step(); // busyWait(10 * 1.0) = 10
    const s2 = loop.step();
    expect(s2.dt).toBeCloseTo(10, 5);
    loop.setActive("B1");
    loop.step(); // busyWait(10 * 0.35) = 3.5
    const s4 = loop.step();
    expect(s4.dt).toBeCloseTo(3.5, 5);
  });
});

describe("SequentialLoop — onFrameComplete delivery", () => {
  it("calls onFrameComplete on every scheduler with the same (dt, wasMiss)", () => {
    const clock = makeFakeClock();
    const factory = makeMockFactory();
    const loop = new SequentialLoop({
      buildState: factory,
      workload: () => 20, // busyWait(20) → dt exceeds budget
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "Predictor",
    });
    loop.step();
    loop.step();
    const s = loop._state;
    // Second frame's dt is the first frame's busyWait (20), which > budget.
    const lastB0 = s._onFrameSpies.B0.mock.calls.at(-1);
    const lastB1 = s._onFrameSpies.B1.mock.calls.at(-1);
    const lastP = s._onFrameSpies.Predictor.mock.calls.at(-1);
    expect(lastB0).toEqual(lastB1);
    expect(lastB1).toEqual(lastP);
    expect(lastB0[0]).toBeCloseTo(20, 5);
    expect(lastB0[1]).toBe(true); // wasMiss
  });
});

describe("SequentialLoop — active-only metrics recording", () => {
  it("records dt only into the active scheduler's metrics", () => {
    const clock = makeFakeClock();
    const factory = makeMockFactory();
    const loop = new SequentialLoop({
      buildState: factory,
      workload: () => 5,
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "B0",
    });
    for (let i = 0; i < 10; i++) loop.step();
    const m = loop.getMetrics();
    expect(m.B0.all.getStats().frameCount).toBe(10);
    expect(m.B1.all.getStats().frameCount).toBe(0);
    expect(m.Predictor.all.getStats().frameCount).toBe(0);
    expect(m.B0.recent.getStats().frameCount).toBe(10);
  });

  it("setActive swaps the recording target without clearing prior data", () => {
    const clock = makeFakeClock();
    const factory = makeMockFactory();
    const loop = new SequentialLoop({
      buildState: factory,
      workload: () => 5,
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "B0",
    });
    for (let i = 0; i < 5; i++) loop.step();
    loop.setActive("B1");
    for (let i = 0; i < 5; i++) loop.step();
    const m = loop.getMetrics();
    expect(m.B0.all.getStats().frameCount).toBe(5);
    expect(m.B1.all.getStats().frameCount).toBe(5);
    expect(m.Predictor.all.getStats().frameCount).toBe(0);
  });
});

describe("SequentialLoop — reset reproducibility", () => {
  it("two resets with the same factory seed produce identical params", () => {
    const clock = makeFakeClock();
    const loop = new SequentialLoop({
      buildState: makeRealFactory(42),
      workload: () => 5,
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "Predictor",
    });
    const p1 = new Float32Array(loop.getParams());
    loop.reset();
    const p2 = new Float32Array(loop.getParams());
    loop.reset();
    const p3 = new Float32Array(loop.getParams());
    // All three should be elementwise identical.
    for (let i = 0; i < PARAM_COUNT; i++) {
      expect(p1[i]).toBe(p2[i]);
      expect(p2[i]).toBe(p3[i]);
    }
  });

  it("reset clears metrics to zero", () => {
    const clock = makeFakeClock();
    const loop = new SequentialLoop({
      buildState: makeRealFactory(42),
      workload: () => 5,
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "Predictor",
    });
    for (let i = 0; i < 10; i++) loop.step();
    expect(loop.getMetrics().Predictor.all.getStats().frameCount).toBe(10);
    loop.reset();
    for (const name of ["B0", "B1", "Predictor"]) {
      expect(loop.getMetrics()[name].all.getStats().frameCount).toBe(0);
      expect(loop.getMetrics()[name].recent.getStats().frameCount).toBe(0);
    }
  });
});

describe("SequentialLoop — trainTick isolation", () => {
  it("preserves trainer buffer semantics (non-destructive) and updates Predictor params", () => {
    const clock = makeFakeClock();
    const loop = new SequentialLoop({
      buildState: makeRealFactory(42),
      workload: () => 5,
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "Predictor",
    });
    for (let i = 0; i < 20; i++) loop.step();
    const paramsBefore = new Float32Array(loop.getParams());
    // 1. bufferCount preserved post-train (non-destructive trainer design).
    //    Count is N-1, not N: PredictorScheduler's first-frame guard skips
    //    the very first onFrameComplete (no prior decide to pair features
    //    with). From step 2 onward each onFrameComplete pushes.
    const trainer = loop._state.trainer;
    expect(trainer.bufferCount()).toBe(19);
    const tick = loop.trainTick();
    expect(tick).not.toBeNull();
    expect(trainer.bufferCount()).toBe(19);
    // 2. Params changed (learning happened).
    const paramsAfter = loop.getParams();
    let maxDelta = 0;
    for (let i = 0; i < PARAM_COUNT; i++) {
      const d = Math.abs(paramsAfter[i] - paramsBefore[i]);
      if (d > maxDelta) maxDelta = d;
    }
    expect(maxDelta).toBeGreaterThan(1e-9);
    // 3. B0/B1 structurally isolated — they do not reference the trainer
    //    and have no learning path. Asserted by construction; no runtime
    //    check needed beyond the factory wiring above.
  });

  it("trainTick returns null when the trainer is empty", () => {
    const clock = makeFakeClock();
    const loop = new SequentialLoop({
      buildState: makeRealFactory(42),
      workload: () => 5,
      busyWait: clock.busyWait,
      now: clock.now,
      initialActive: "B0", // Predictor is not active → no training pairs pushed
    });
    // PredictorScheduler still runs decide+onFrameComplete (contract §4 Phase 2),
    // so the trainer DOES receive pushes. Use B0 as active so Predictor is
    // still in shadow and still pushes — wait, PredictorScheduler pushes
    // unconditionally in onFrameComplete. So trainer will have samples after
    // any step. Instead call trainTick before any step.
    const tick = loop.trainTick();
    expect(tick).toBeNull();
  });
});

describe("SequentialLoop — determinism", () => {
  it("identical seed + workload + active sequence → identical step results", () => {
    const runOnce = () => {
      const clock = makeFakeClock();
      const loop = new SequentialLoop({
        buildState: makeRealFactory(42),
        workload: (i) => 2 + (i % 5), // deterministic workload
        busyWait: clock.busyWait,
        now: clock.now,
        initialActive: "Predictor",
      });
      const dts = [];
      for (let i = 0; i < 20; i++) {
        const r = loop.step();
        dts.push(r.dt);
      }
      return { dts, params: new Float32Array(loop.getParams()) };
    };
    const a = runOnce();
    const b = runOnce();
    expect(a.dts).toEqual(b.dts);
    for (let i = 0; i < PARAM_COUNT; i++) {
      expect(a.params[i]).toBe(b.params[i]);
    }
  });
});
