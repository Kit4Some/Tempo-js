import { describe, it, expect } from "vitest";
import { computeDriftReport } from "../scripts/drift-check.js";

function rec(workload, jankRate, opts = {}) {
  return {
    status: "ok",
    workload,
    active: opts.active ?? "B1",
    condition: opts.condition ?? "scratch",
    pretrained: false,
    freeze: false,
    jankRate,
  };
}

describe("computeDriftReport", () => {
  it("PASS when all drift runs lie inside Part 1's [mean ± 2σ] band", () => {
    // Part 1 B1: small spread → tight band.
    const p1 = ["constant", "sawtooth", "burst", "scroll"].flatMap((wl) =>
      [0.050, 0.051, 0.049, 0.052, 0.050].map((j) => rec(wl, j)),
    );
    // Drift runs dead-center.
    const p2 = ["constant", "sawtooth", "burst", "scroll"].flatMap((wl) =>
      [0.050, 0.051].map((j) => rec(wl, j)),
    );
    const r = computeDriftReport(p1, p2);
    expect(r.verdict).toBe("PASS");
    for (const s of r.sections) {
      expect(s.status).toBe("PASS");
      expect(s.outliers).toBe(0);
    }
  });

  it("WARNING on exactly one out-of-band drift run when mean shift stays within tolerance", () => {
    // Part 1 B1: tight spread (μ=0.0502, σ≈0.0004). With the 0.5pp min
    // halfband floor the band is [0.0452, 0.0552].
    // Part 2 drift: one outlier (0.060 — outside upper band) + one near
    //   μ (0.050 — inside band). Mean = 0.055, shift = 0.48pp ≤ 1.0pp →
    //   absShift rule does not escalate to STOP → WARNING.
    const p1 = ["burst"].flatMap((wl) =>
      [0.050, 0.050, 0.050, 0.051, 0.050].map((j) => rec(wl, j)),
    );
    const p2 = [rec("burst", 0.060), rec("burst", 0.050)];
    const r = computeDriftReport(p1, p2, { workloads: ["burst"] });
    expect(r.sections[0].outliers).toBe(1);
    expect(r.sections[0].absShift).toBeLessThan(0.01);
    expect(r.sections[0].status).toBe("WARNING");
    expect(r.verdict).toBe("WARNING");
  });

  it("STOP on ≥ 2 out-of-band drift runs for a single workload", () => {
    const p1 = ["burst"].flatMap((wl) =>
      [0.050, 0.051, 0.049, 0.052, 0.050].map((j) => rec(wl, j)),
    );
    const p2 = [rec("burst", 0.090), rec("burst", 0.095)];
    const r = computeDriftReport(p1, p2, { workloads: ["burst"] });
    expect(r.verdict).toBe("STOP");
    expect(r.sections[0].status).toBe("STOP");
    expect(r.sections[0].outliers).toBe(2);
  });

  it("STOP on mean drift > 1.0 percentage points (absolute)", () => {
    // Individual runs inside band but the aggregate mean shifts > 1pp.
    // Part 1 mean ≈ 5.04%; Part 2 runs all ~6.20% → shift 1.16pp.
    // Each 6.20% could lie inside [mean ± 2σ] if σ is wide enough. We
    // use a wider Part 1 spread so individual runs don't trip outliers
    // but the aggregate mean does.
    const p1 = ["burst"].flatMap((wl) =>
      [0.030, 0.040, 0.050, 0.060, 0.070].map((j) => rec(wl, j)),
    );
    // band ≈ [0.050 - 2×0.0158, 0.050 + 2×0.0158] ≈ [0.018, 0.082]
    // Drift runs both at 0.070 → inside band, but aggregate mean 0.070
    // vs Part 1 mean 0.050 → 2.0pp shift > 1.0pp threshold.
    const p2 = [rec("burst", 0.070), rec("burst", 0.070)];
    const r = computeDriftReport(p1, p2, { workloads: ["burst"] });
    expect(r.sections[0].outliers).toBe(0);
    expect(r.sections[0].absShift).toBeGreaterThan(0.01);
    expect(r.sections[0].status).toBe("STOP");
    expect(r.verdict).toBe("STOP");
  });

  it("aggregate verdict is STOP if any workload is STOP", () => {
    const p1 = ["burst", "scroll"].flatMap((wl) =>
      [0.050, 0.051, 0.049, 0.052, 0.050].map((j) => rec(wl, j)),
    );
    const p2 = [
      rec("burst", 0.050),
      rec("burst", 0.051),
      rec("scroll", 0.090), // STOP (outlier #1)
      rec("scroll", 0.095), // STOP (outlier #2)
    ];
    const r = computeDriftReport(p1, p2, { workloads: ["burst", "scroll"] });
    expect(r.sections.find((s) => s.workload === "burst").status).toBe("PASS");
    expect(r.sections.find((s) => s.workload === "scroll").status).toBe(
      "STOP",
    );
    expect(r.verdict).toBe("STOP");
  });

  it("NO_DATA when either side is missing B1 for a workload", () => {
    const p1 = [rec("burst", 0.05)]; // Only burst
    const p2 = [rec("burst", 0.05)];
    const r = computeDriftReport(p1, p2, {
      workloads: ["burst", "scroll"],
    });
    const scroll = r.sections.find((s) => s.workload === "scroll");
    expect(scroll.status).toBe("NO_DATA");
  });

  it("ignores non-B1 records when pooling Part 1 baseline", () => {
    // Predictor / B0 records on the same workload must not leak into B1's band.
    const p1 = [
      rec("burst", 0.050),
      rec("burst", 0.051),
      rec("burst", 0.049),
      rec("burst", 0.052),
      rec("burst", 0.050),
      rec("burst", 0.500, { active: "Predictor" }), // should be excluded
    ];
    const p2 = [rec("burst", 0.050), rec("burst", 0.051)];
    const r = computeDriftReport(p1, p2, { workloads: ["burst"] });
    expect(r.sections[0].n1).toBe(5);
    expect(r.sections[0].status).toBe("PASS");
  });

  it("enforces 0.5pp min-halfband floor so zero-σ baselines (deterministic B1) stay usable", () => {
    // Deterministic B1: 10 runs, identical jankRate → σ = 0.
    // Strict 2σ band would collapse to {μ}; with the 0.5pp floor the band
    // becomes [μ-0.5pp, μ+0.5pp]. Drift runs 0.3pp above μ should stay
    // inliers (no outliers, no status change).
    const p1 = ["burst"].flatMap((wl) =>
      Array.from({ length: 10 }, () => rec(wl, 0.050)),
    );
    const p2 = [rec("burst", 0.053), rec("burst", 0.053)];
    const r = computeDriftReport(p1, p2, { workloads: ["burst"] });
    expect(r.sections[0].p1SD).toBeCloseTo(0, 10);
    expect(r.sections[0].band[0]).toBeCloseTo(0.045, 5); // 0.050 - 0.005
    expect(r.sections[0].band[1]).toBeCloseTo(0.055, 5); // 0.050 + 0.005
    expect(r.sections[0].outliers).toBe(0);
    expect(r.sections[0].status).toBe("PASS");
  });

  it("surfaces per-section numeric fields for the Markdown report", () => {
    const p1 = ["burst"].flatMap((wl) =>
      [0.050, 0.051, 0.049, 0.052, 0.050].map((j) => rec(wl, j)),
    );
    const p2 = [rec("burst", 0.050), rec("burst", 0.051)];
    const r = computeDriftReport(p1, p2, { workloads: ["burst"] });
    const s = r.sections[0];
    expect(typeof s.p1Mean).toBe("number");
    expect(typeof s.p1SD).toBe("number");
    expect(Array.isArray(s.band)).toBe(true);
    expect(s.band.length).toBe(2);
    expect(typeof s.p2Mean).toBe("number");
    expect(typeof s.absShift).toBe("number");
  });
});
