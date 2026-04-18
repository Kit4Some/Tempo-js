import { describe, it, expect } from "vitest";
import { FrameMetrics, RollingFrameMetrics } from "../src/harness/metrics.js";

function makeMockRaf() {
  const pending = [];
  return {
    raf(cb) {
      pending.push(cb);
      return pending.length;
    },
    caf() {},
    tick(ts) {
      const now = pending.splice(0);
      for (const cb of now) cb(ts);
    },
  };
}

function makeMetrics(budgetMs = 16.67) {
  const raf = makeMockRaf();
  const m = new FrameMetrics(budgetMs, { raf: raf.raf, caf: raf.caf });
  return { m, raf };
}

describe("FrameMetrics", () => {
  it("records dt from the second tick onwards (first tick is baseline)", () => {
    const { m, raf } = makeMetrics();
    m.start();
    raf.tick(0);
    raf.tick(10);
    raf.tick(25);
    const s = m.getStats();
    expect(s.frameCount).toBe(2);
    expect(s.meanDt).toBeCloseTo(12.5, 5);
  });

  it("computes jankRate = fraction of dts exceeding budget", () => {
    const { m, raf } = makeMetrics(16.67);
    m.start();
    raf.tick(0);
    raf.tick(10); // dt=10, ok
    raf.tick(20); // dt=10, ok
    raf.tick(40); // dt=20, jank
    const s = m.getStats();
    expect(s.frameCount).toBe(3);
    expect(s.jankRate).toBeCloseTo(1 / 3, 5);
  });

  it("returns zero-valued stats with no frames", () => {
    const { m } = makeMetrics();
    const s = m.getStats();
    expect(s).toEqual({
      jankRate: 0,
      p95: 0,
      p99: 0,
      meanDt: 0,
      frameCount: 0,
    });
  });

  it("reports p95 and p99 from recorded dts", () => {
    const { m, raf } = makeMetrics(16.67);
    m.start();
    let t = 0;
    raf.tick(t);
    for (let i = 1; i <= 100; i++) {
      t += i; // dts = 1, 2, 3, ..., 100
      raf.tick(t);
    }
    const s = m.getStats();
    expect(s.frameCount).toBe(100);
    expect(s.meanDt).toBeCloseTo(50.5, 5);
    expect(s.p95).toBeGreaterThanOrEqual(95);
    expect(s.p95).toBeLessThanOrEqual(100);
    expect(s.p99).toBeGreaterThanOrEqual(99);
    expect(s.p99).toBeLessThanOrEqual(100);
  });

  it("caps ring buffer at 1024 entries but keeps frameCount growing", () => {
    const { m, raf } = makeMetrics();
    m.start();
    raf.tick(0);
    for (let i = 1; i <= 2000; i++) {
      raf.tick(i * 10); // all dts = 10
    }
    const s = m.getStats();
    expect(s.frameCount).toBe(2000);
    expect(s.meanDt).toBeCloseTo(10, 5);
  });

  it("reset() clears frames but keeps start() state usable", () => {
    const { m, raf } = makeMetrics();
    m.start();
    raf.tick(0);
    raf.tick(10);
    raf.tick(20);
    m.reset();
    expect(m.getStats().frameCount).toBe(0);
    raf.tick(100);
    raf.tick(115);
    const s = m.getStats();
    expect(s.frameCount).toBe(1);
    expect(s.meanDt).toBeCloseTo(15, 5);
  });

  it("onFrame callback fires with (frameIdx, dt) each recorded frame", () => {
    const { m, raf } = makeMetrics();
    const calls = [];
    m.onFrame((idx, dt) => calls.push({ idx, dt }));
    m.start();
    raf.tick(0);
    raf.tick(10);
    raf.tick(25);
    expect(calls).toEqual([
      { idx: 0, dt: 10 },
      { idx: 1, dt: 15 },
    ]);
  });

  it("stop() halts further frame recording", () => {
    const { m, raf } = makeMetrics();
    m.start();
    raf.tick(0);
    raf.tick(10);
    m.stop();
    raf.tick(100); // should not re-schedule, and _tick is guarded
    expect(m.getStats().frameCount).toBe(1);
  });
});

describe("RollingFrameMetrics", () => {
  it("returns zero-valued stats when no frames recorded", () => {
    const r = new RollingFrameMetrics(300);
    expect(r.getStats()).toEqual({
      jankRate: 0,
      p95: 0,
      p99: 0,
      meanDt: 0,
      frameCount: 0,
    });
  });

  it("reports the same shape as FrameMetrics.getStats()", () => {
    const r = new RollingFrameMetrics(300);
    r.record(10);
    const keys = Object.keys(r.getStats()).sort();
    expect(keys).toEqual(["frameCount", "jankRate", "meanDt", "p95", "p99"]);
  });

  it("records dts and reflects them in stats", () => {
    const r = new RollingFrameMetrics(300, 16.67);
    r.record(10);
    r.record(20);
    r.record(30);
    const s = r.getStats();
    expect(s.frameCount).toBe(3);
    expect(s.meanDt).toBeCloseTo(20, 5);
    expect(s.jankRate).toBeCloseTo(2 / 3, 5); // 20, 30 > 16.67
  });

  it("keeps at most windowSize frames, dropping oldest", () => {
    const r = new RollingFrameMetrics(3, 16.67);
    r.record(10); // eventually dropped
    r.record(20);
    r.record(30);
    r.record(40);
    // window now holds [20, 30, 40]
    const s = r.getStats();
    expect(s.frameCount).toBe(3);
    expect(s.meanDt).toBeCloseTo(30, 5);
  });

  it("p95 and p99 are computed within the rolling window, not lifetime", () => {
    const r = new RollingFrameMetrics(10, 16.67);
    // Pollute first with huge values that will be evicted.
    for (let i = 0; i < 10; i++) r.record(1000);
    // Fill window with small values.
    for (let i = 0; i < 10; i++) r.record(5);
    const s = r.getStats();
    expect(s.frameCount).toBe(10);
    expect(s.p95).toBeCloseTo(5, 5);
    expect(s.p99).toBeCloseTo(5, 5);
    expect(s.jankRate).toBeCloseTo(0, 5);
  });

  it("jankRate counts only dts strictly greater than budget", () => {
    // Budget 20 is exactly representable in Float32, unlike 16.67 which would
    // round up and falsely treat the boundary sample as jank.
    const r = new RollingFrameMetrics(10, 20);
    r.record(20); // equal, not jank (strict >)
    r.record(20.5); // jank
    const s = r.getStats();
    expect(s.frameCount).toBe(2);
    expect(s.jankRate).toBeCloseTo(0.5, 5);
  });

  it("reset() clears the window but leaves the instance reusable", () => {
    const r = new RollingFrameMetrics(10, 16.67);
    r.record(10);
    r.record(20);
    r.reset();
    expect(r.getStats().frameCount).toBe(0);
    r.record(15);
    const s = r.getStats();
    expect(s.frameCount).toBe(1);
    expect(s.meanDt).toBeCloseTo(15, 5);
  });

  it("defaults to FRAME_BUDGET_60 when budget is not given", () => {
    // 5ms is clearly below 16.67 budget, 20ms is clearly above — if the
    // default were wildly wrong, jankRate would be 0 or 1, not 0.5.
    const r = new RollingFrameMetrics(10);
    r.record(5);
    r.record(20);
    expect(r.getStats().jankRate).toBeCloseTo(0.5, 5);
  });
});
