import { describe, it, expect } from "vitest";
import { writeFileSync, readFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import {
  loadResults,
  normalizeRecords,
  selectCell,
  compareCells,
  runCompareMode,
  parseAnalyzeArgs,
} from "../scripts/analyze.js";
import { mulberry32 } from "../tests/helpers/rng.js";

describe("parseAnalyzeArgs", () => {
  it("empty argv → no compare, no out", () => {
    const a = parseAnalyzeArgs(["node", "analyze.js"]);
    expect(a).toEqual({ compare: null, out: null });
  });

  it("--compare=a,b parses into {part1, part2}", () => {
    const a = parseAnalyzeArgs([
      "node",
      "analyze.js",
      "--compare=docs/p1.jsonl,docs/p2.jsonl",
    ]);
    expect(a.compare).toEqual({
      part1: "docs/p1.jsonl",
      part2: "docs/p2.jsonl",
    });
  });

  it("--out= overrides the default output path", () => {
    const a = parseAnalyzeArgs([
      "node",
      "analyze.js",
      "--compare=a,b",
      "--out=foo/bar.md",
    ]);
    expect(a.out).toBe("foo/bar.md");
  });

  it("rejects --compare with wrong arity", () => {
    expect(() =>
      parseAnalyzeArgs(["node", "analyze.js", "--compare=only-one"]),
    ).toThrow();
    expect(() =>
      parseAnalyzeArgs(["node", "analyze.js", "--compare=a,b,c"]),
    ).toThrow();
  });

  it("rejects unknown arguments", () => {
    expect(() => parseAnalyzeArgs(["node", "analyze.js", "--bogus"])).toThrow();
  });
});

describe("normalizeRecords", () => {
  it("defaults missing condition/pretrained/freeze for Part 1 records", () => {
    const records = [
      { status: "ok", workload: "burst", active: "B1", jankRate: 0.05 },
    ];
    const out = normalizeRecords(records);
    expect(out).toHaveLength(1);
    expect(out[0].condition).toBe("scratch");
    expect(out[0].pretrained).toBe(false);
    expect(out[0].freeze).toBe(false);
  });

  it("preserves condition/pretrained/freeze on Part 2 records", () => {
    const records = [
      {
        status: "ok",
        workload: "burst",
        active: "Predictor",
        condition: "pretrained+frozen",
        pretrained: true,
        freeze: true,
        jankRate: 0.03,
      },
    ];
    const out = normalizeRecords(records);
    expect(out[0].condition).toBe("pretrained+frozen");
    expect(out[0].pretrained).toBe(true);
    expect(out[0].freeze).toBe(true);
  });

  it("drops non-ok records", () => {
    const records = [
      { status: "ok", jankRate: 0.05, workload: "burst", active: "B1" },
      { status: "error", error: "boom" },
    ];
    expect(normalizeRecords(records)).toHaveLength(1);
  });

  it("passes through records with no status field (pre-Phase-5 fixtures)", () => {
    const records = [{ workload: "burst", active: "B1", jankRate: 0.05 }];
    expect(normalizeRecords(records)).toHaveLength(1);
  });
});

describe("selectCell", () => {
  const R = normalizeRecords([
    { status: "ok", workload: "burst", active: "Predictor", condition: "scratch", jankRate: 0.1 },
    { status: "ok", workload: "burst", active: "Predictor", condition: "pretrained+online", jankRate: 0.05 },
    { status: "ok", workload: "burst", active: "B1", jankRate: 0.06 },
    { status: "ok", workload: "scroll", active: "Predictor", condition: "scratch", jankRate: 0.07 },
  ]);

  it("filters by workload", () => {
    expect(selectCell(R, { workload: "burst" })).toHaveLength(3);
    expect(selectCell(R, { workload: "scroll" })).toHaveLength(1);
  });

  it("filters by active", () => {
    expect(selectCell(R, { active: "Predictor" })).toHaveLength(3);
    expect(selectCell(R, { active: "B1" })).toHaveLength(1);
  });

  it("filters by condition", () => {
    expect(selectCell(R, { condition: "scratch" })).toHaveLength(3);
    expect(selectCell(R, { condition: "pretrained+online" })).toHaveLength(1);
  });

  it("combines filters conjunctively", () => {
    const sel = selectCell(R, {
      workload: "burst",
      active: "Predictor",
      condition: "scratch",
    });
    expect(sel).toHaveLength(1);
    expect(sel[0].jankRate).toBe(0.1);
  });
});

describe("compareCells", () => {
  function fakeCell(jankRates) {
    return jankRates.map((j) => ({ jankRate: j }));
  }

  it("flags INSUFFICIENT when either cell has < 2 records", () => {
    const rng = mulberry32(1);
    const r = compareCells(fakeCell([0.1]), fakeCell([0.2, 0.3]), rng);
    expect(r.insufficient).toBe(true);
    expect(r.verdict).toBe("INSUFFICIENT");
  });

  it("returns sign-correct Cohen's d (A<B ⇒ d<0)", () => {
    const rng = mulberry32(1);
    // Cell A clearly lower than Cell B.
    const r = compareCells(
      fakeCell([0.01, 0.02, 0.01, 0.02, 0.01]),
      fakeCell([0.10, 0.12, 0.11, 0.10, 0.09]),
      rng,
    );
    expect(r.d).toBeLessThan(0);
    expect(r.meanA).toBeLessThan(r.meanB);
    // Large effect size + low p → GO (A<B).
    expect(r.verdict).toBe("GO (A<B)");
  });

  it("reports NO-GO when distributions overlap", () => {
    const rng = mulberry32(1);
    const r = compareCells(
      fakeCell([0.05, 0.06, 0.04, 0.05, 0.06]),
      fakeCell([0.05, 0.06, 0.04, 0.05, 0.06]),
      rng,
    );
    expect(r.verdict).toBe("NO-GO");
    // Identical samples: d = 0, p = 1 (Mann-Whitney U gives tie center).
    expect(Math.abs(r.d)).toBeLessThan(1e-9);
  });

  it("CIs bracket the sample mean", () => {
    const rng = mulberry32(1);
    const a = [0.05, 0.06, 0.04, 0.05, 0.06];
    const b = [0.10, 0.12, 0.11, 0.10, 0.09];
    const r = compareCells(fakeCell(a), fakeCell(b), rng);
    expect(r.ciA[0]).toBeLessThanOrEqual(r.meanA);
    expect(r.ciA[1]).toBeGreaterThanOrEqual(r.meanA);
    expect(r.ciB[0]).toBeLessThanOrEqual(r.meanB);
    expect(r.ciB[1]).toBeGreaterThanOrEqual(r.meanB);
  });
});

describe("runCompareMode — integration against synthetic JSONL fixtures", () => {
  it("writes a Markdown file with all three pre-registered sections", () => {
    const tmpDir = tmpdir();
    const p1Path = resolve(tmpDir, `tempo-p1-${Date.now()}.jsonl`);
    const p2Path = resolve(tmpDir, `tempo-p2-${Date.now()}.jsonl`);
    const outPath = resolve(tmpDir, `tempo-out-${Date.now()}.md`);

    // Part 1: B0, B1, Predictor × 4 workloads × 3 reps each.
    const p1 = [];
    const WL = ["constant", "sawtooth", "burst", "scroll"];
    const SC = ["B0", "B1", "Predictor"];
    let idx = 0;
    for (const wl of WL) {
      for (const sc of SC) {
        for (let rep = 0; rep < 3; rep++) {
          p1.push({
            status: "ok",
            runIndex: idx++,
            workload: wl,
            active: sc,
            rep,
            jankRate: 0.05 + rep * 0.001,
            p95: 20,
            meanDt: 10,
          });
        }
      }
    }
    // Part 2: Predictor × 2 conditions × 4 workloads × 3 reps + B1 drift.
    const p2 = [];
    for (const wl of WL) {
      for (const cond of ["pretrained+online", "pretrained+frozen"]) {
        for (let rep = 0; rep < 3; rep++) {
          p2.push({
            status: "ok",
            runIndex: idx++,
            workload: wl,
            active: "Predictor",
            condition: cond,
            pretrained: true,
            freeze: cond === "pretrained+frozen",
            rep,
            jankRate: cond === "pretrained+frozen" ? 0.03 : 0.04,
            p95: 20,
            meanDt: 10,
          });
        }
      }
      for (let rep = 0; rep < 2; rep++) {
        p2.push({
          status: "ok",
          runIndex: idx++,
          workload: wl,
          active: "B1",
          condition: "scratch",
          pretrained: false,
          freeze: false,
          rep,
          jankRate: 0.05,
          p95: 20,
          meanDt: 10,
        });
      }
    }
    writeFileSync(p1Path, p1.map((r) => JSON.stringify(r)).join("\n"));
    writeFileSync(p2Path, p2.map((r) => JSON.stringify(r)).join("\n"));

    try {
      runCompareMode({ part1Path: p1Path, part2Path: p2Path, outPath });
      const md = readFileSync(outPath, "utf-8");
      expect(md).toContain("Scratch (Part 1) vs Pretrained + Online");
      expect(md).toContain("Pretrained + Online vs Pretrained + Frozen");
      expect(md).toContain("B1 (hand-crafted frozen prior) vs Pretrained + Frozen");
      // Basic table shape checks.
      expect(md).toContain("| Workload |");
      // At least one workload row per section — 4 workloads × 3 sections = 12 rows.
      const rowCount = (md.match(/^\| (constant|sawtooth|burst|scroll) \|/gm) || [])
        .length;
      expect(rowCount).toBe(12);
    } finally {
      unlinkSync(p1Path);
      unlinkSync(p2Path);
      unlinkSync(outPath);
    }
  });
});
