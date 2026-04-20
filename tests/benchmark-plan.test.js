import { describe, it, expect } from "vitest";
import {
  parseArgs,
  buildRunPlan,
  buildRunURL,
} from "../scripts/benchmark.js";

describe("parseArgs", () => {
  it("defaults to mode=part1, reps=10, resume=false", () => {
    const args = parseArgs(["node", "benchmark.js"]);
    expect(args).toEqual({ reps: 10, resume: false, mode: "part1" });
  });

  it("parses --reps / --resume / --mode", () => {
    const args = parseArgs([
      "node",
      "benchmark.js",
      "--reps=3",
      "--mode=part2",
      "--resume",
    ]);
    expect(args).toEqual({ reps: 3, resume: true, mode: "part2" });
  });

  it("rejects invalid --reps", () => {
    expect(() => parseArgs(["node", "bench.js", "--reps=0"])).toThrow();
    expect(() => parseArgs(["node", "bench.js", "--reps=-3"])).toThrow();
    expect(() => parseArgs(["node", "bench.js", "--reps=foo"])).toThrow();
  });

  it("rejects invalid --mode", () => {
    expect(() => parseArgs(["node", "bench.js", "--mode=part3"])).toThrow();
    expect(() => parseArgs(["node", "bench.js", "--mode=bogus"])).toThrow();
  });

  it("rejects unknown arguments", () => {
    expect(() => parseArgs(["node", "bench.js", "--foo=bar"])).toThrow();
  });
});

describe("buildRunPlan — part1 (backward compatible)", () => {
  it("produces 4 workloads × 3 schedulers × reps runs", () => {
    const plan = buildRunPlan(10, "part1");
    expect(plan.length).toBe(4 * 3 * 10);
  });

  it("every part1 entry has condition=scratch, no pretrained / freeze", () => {
    const plan = buildRunPlan(2, "part1");
    for (const p of plan) {
      expect(p.condition).toBe("scratch");
      expect(p.pretrained).toBe(false);
      expect(p.freeze).toBe(false);
    }
  });

  it("assigns a unique runIndex and sequential executionPosition", () => {
    const plan = buildRunPlan(2, "part1");
    const idx = new Set(plan.map((p) => p.runIndex));
    expect(idx.size).toBe(plan.length);
    plan.forEach((p, i) => expect(p.executionPosition).toBe(i));
  });
});

describe("buildRunPlan — part2", () => {
  it("emits 4 workloads × (2 Predictor conditions × reps + PART2_DRIFT_REPS B1) entries", () => {
    const reps = 10;
    const plan = buildRunPlan(reps, "part2");
    // 4 workloads × (2 × 10 Predictor + 2 B1 drift) = 4 × 22 = 88
    expect(plan.length).toBe(4 * (2 * reps + 2));
  });

  it("Predictor runs split into pretrained+online and pretrained+frozen", () => {
    const plan = buildRunPlan(10, "part2");
    const pred = plan.filter((p) => p.active === "Predictor");
    const online = pred.filter((p) => p.condition === "pretrained+online");
    const frozen = pred.filter((p) => p.condition === "pretrained+frozen");
    expect(online.length).toBe(40);
    expect(frozen.length).toBe(40);
    for (const p of online) {
      expect(p.pretrained).toBe(true);
      expect(p.freeze).toBe(false);
    }
    for (const p of frozen) {
      expect(p.pretrained).toBe(true);
      expect(p.freeze).toBe(true);
    }
  });

  it("B1 drift runs use condition=scratch with scratch init", () => {
    const plan = buildRunPlan(10, "part2");
    const b1 = plan.filter((p) => p.active === "B1");
    expect(b1.length).toBe(8); // 2 reps × 4 workloads
    for (const p of b1) {
      expect(p.condition).toBe("scratch");
      expect(p.pretrained).toBe(false);
      expect(p.freeze).toBe(false);
    }
  });

  it("does not schedule B0 in part2 (training-data role is complete)", () => {
    const plan = buildRunPlan(10, "part2");
    expect(plan.filter((p) => p.active === "B0").length).toBe(0);
  });

  it("plan is deterministic: same reps+mode → identical order (seeded shuffle)", () => {
    const a = buildRunPlan(5, "part2");
    const b = buildRunPlan(5, "part2");
    // executionPosition reflects the post-shuffle order — stable across runs.
    expect(a.map((p) => p.runIndex)).toEqual(b.map((p) => p.runIndex));
  });

  it("rejects unknown mode", () => {
    expect(() => buildRunPlan(10, "part3")).toThrow();
  });
});

describe("buildRunURL", () => {
  const BASE = "http://localhost:4173/Tempo-js/";

  it("returns the bare URL for scratch (no query)", () => {
    const plan = { pretrained: false, freeze: false };
    expect(buildRunURL(BASE, plan)).toBe(BASE);
  });

  it("sets ?init=pretrained for pretrained+online", () => {
    const plan = { pretrained: true, freeze: false };
    expect(buildRunURL(BASE, plan)).toBe(`${BASE}?init=pretrained`);
  });

  it("sets both params for pretrained+frozen", () => {
    const plan = { pretrained: true, freeze: true };
    expect(buildRunURL(BASE, plan)).toBe(
      `${BASE}?init=pretrained&freeze=true`,
    );
  });

  it("scratch+freeze emits freeze=true even without init (parser accepts it)", () => {
    // Odd combination — we don't gate it here because the plan builder does
    // not produce this combo for any scheduled condition. Parser tolerates
    // it, which is what matters for live-URL exploration.
    const plan = { pretrained: false, freeze: true };
    expect(buildRunURL(BASE, plan)).toBe(`${BASE}?freeze=true`);
  });
});
