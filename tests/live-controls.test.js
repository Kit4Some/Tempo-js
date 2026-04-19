import { describe, it, expect, vi } from "vitest";
import {
  formatJank,
  formatMs,
  buildStatsRows,
  runSequence,
  parseInitOpts,
} from "../benchmark/live-controls.js";

describe("formatJank", () => {
  it("renders 0 as 0.0%", () => {
    expect(formatJank(0)).toBe("0.0%");
  });
  it("renders 0.123 as 12.3%", () => {
    expect(formatJank(0.123)).toBe("12.3%");
  });
  it("renders 1 as 100.0%", () => {
    expect(formatJank(1)).toBe("100.0%");
  });
  it("renders 0.9999 as 100.0% (rounding)", () => {
    expect(formatJank(0.9999)).toBe("100.0%");
  });
});

describe("formatMs", () => {
  it("renders 5.42 as 5.4", () => {
    expect(formatMs(5.42)).toBe("5.4");
  });
  it("renders 100.01 as 100.0", () => {
    expect(formatMs(100.01)).toBe("100.0");
  });
  it("renders 0 as 0.0 (still displays a number, unlike empty indicator)", () => {
    // formatMs is for non-empty values; empty state is handled in
    // buildStatsRows.
    expect(formatMs(0)).toBe("0.0");
  });
});

function mockMetrics({ B0 = {}, B1 = {}, Predictor = {} } = {}) {
  const mk = (o = {}) => ({
    all: {
      getStats: () => ({
        frameCount: o.allFrames ?? 0,
        jankRate: o.allJank ?? 0,
        p95: o.allP95 ?? 0,
        p99: o.allP99 ?? 0,
        meanDt: o.allMean ?? 0,
      }),
    },
    recent: {
      getStats: () => ({
        frameCount: o.recentFrames ?? 0,
        jankRate: o.recentJank ?? 0,
        p95: o.recentP95 ?? 0,
        p99: o.recentP99 ?? 0,
        meanDt: o.recentMean ?? 0,
      }),
    },
  });
  return { B0: mk(B0), B1: mk(B1), Predictor: mk(Predictor) };
}

describe("buildStatsRows", () => {
  it("returns three rows in B0 → B1 → Predictor order", () => {
    const rows = buildStatsRows(mockMetrics(), "Predictor");
    expect(rows.map((r) => r.name)).toEqual(["B0", "B1", "Predictor"]);
  });

  it("marks only the active scheduler row as active", () => {
    const rows = buildStatsRows(mockMetrics(), "B1");
    expect(rows.find((r) => r.name === "B0").active).toBe(false);
    expect(rows.find((r) => r.name === "B1").active).toBe(true);
    expect(rows.find((r) => r.name === "Predictor").active).toBe(false);
  });

  it("shows — placeholders for schedulers with no frames recorded", () => {
    const rows = buildStatsRows(mockMetrics(), "Predictor");
    expect(rows[0].jankAll).toBe("—");
    expect(rows[0].jankRecent).toBe("—");
    expect(rows[0].p95All).toBe("—");
    expect(rows[0].p95Recent).toBe("—");
    expect(rows[0].meanDt).toBe("—");
  });

  it("formats non-empty stats into percentage/ms strings", () => {
    const m = mockMetrics({
      B0: {
        allFrames: 100,
        allJank: 0.123,
        allP95: 18.25,
        allMean: 12.3,
        recentFrames: 50,
        recentJank: 0.2,
        recentP95: 20.1,
      },
    });
    const rows = buildStatsRows(m, "B0");
    const b0 = rows.find((r) => r.name === "B0");
    expect(b0.jankAll).toBe("12.3%");
    expect(b0.jankRecent).toBe("20.0%");
    expect(b0.p95All).toBe("18.3");
    expect(b0.p95Recent).toBe("20.1");
    expect(b0.meanDt).toBe("12.3");
  });

  it("treats recent-empty but all-populated as a valid mixed state", () => {
    // If recent window was just reset, all may still have values.
    const m = mockMetrics({
      B0: { allFrames: 5, allJank: 0.4, allP95: 17, allMean: 12 },
    });
    const rows = buildStatsRows(m, "B0");
    const b0 = rows.find((r) => r.name === "B0");
    expect(b0.jankAll).toBe("40.0%");
    expect(b0.jankRecent).toBe("—");
  });
});

describe("runSequence", () => {
  function fakeLoop() {
    return { reset: vi.fn(), setActive: vi.fn() };
  }

  it("runs B0 → B1 → Predictor in order, resetting + setActive each phase", async () => {
    const loop = fakeLoop();
    const phases = [];
    await runSequence({
      loop,
      signal: new AbortController().signal,
      onPhase: (p) => phases.push(p),
      runMs: 1,
      cooldownMs: 1,
      sleep: () => Promise.resolve(),
    });
    expect(loop.reset).toHaveBeenCalledTimes(3);
    expect(loop.setActive.mock.calls).toEqual([
      ["B0"],
      ["B1"],
      ["Predictor"],
    ]);
    const kinds = phases.map((p) => p.kind);
    // started, finished, cooldown, started, finished, cooldown, started, finished, complete
    expect(kinds).toEqual([
      "started",
      "finished",
      "cooldown",
      "started",
      "finished",
      "cooldown",
      "started",
      "finished",
      "complete",
    ]);
  });

  it("fires no cooldown after the final scheduler", async () => {
    const loop = fakeLoop();
    const phases = [];
    await runSequence({
      loop,
      signal: new AbortController().signal,
      onPhase: (p) => phases.push(p),
      runMs: 1,
      cooldownMs: 1,
      sleep: () => Promise.resolve(),
    });
    const cooldowns = phases.filter((p) => p.kind === "cooldown");
    expect(cooldowns).toHaveLength(2);
  });

  it("throws AbortError immediately on a pre-aborted signal", async () => {
    const loop = fakeLoop();
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(
      runSequence({
        loop,
        signal: ctrl.signal,
        onPhase: () => {},
        runMs: 1,
        cooldownMs: 1,
        sleep: () => Promise.resolve(),
      }),
    ).rejects.toThrow(/abort/i);
    expect(loop.reset).not.toHaveBeenCalled();
  });

  it("aborts mid-sequence when signal trips during a sleep", async () => {
    const loop = fakeLoop();
    const ctrl = new AbortController();
    let sleepCalls = 0;
    const sleep = () => {
      sleepCalls++;
      if (sleepCalls === 2) ctrl.abort(); // right after B0's run sleep
      return Promise.resolve();
    };
    await expect(
      runSequence({
        loop,
        signal: ctrl.signal,
        onPhase: () => {},
        runMs: 1,
        cooldownMs: 1,
        sleep,
      }),
    ).rejects.toThrow(/abort/i);
    // B0 got reset/setActive; abort fires before B1 starts.
    expect(loop.setActive.mock.calls.length).toBeLessThan(3);
  });
});

describe("parseInitOpts — URL query → init options", () => {
  it("defaults to scratch + online when no params are present", () => {
    expect(parseInitOpts("")).toEqual({ pretrained: false, freeze: false });
    expect(parseInitOpts("?")).toEqual({ pretrained: false, freeze: false });
  });

  it("?init=pretrained sets pretrained:true (freeze defaults to false)", () => {
    expect(parseInitOpts("?init=pretrained")).toEqual({
      pretrained: true,
      freeze: false,
    });
  });

  it("?init=pretrained&freeze=true sets both", () => {
    expect(parseInitOpts("?init=pretrained&freeze=true")).toEqual({
      pretrained: true,
      freeze: true,
    });
  });

  it("?init=scratch explicitly selects scratch (same as default)", () => {
    expect(parseInitOpts("?init=scratch")).toEqual({
      pretrained: false,
      freeze: false,
    });
  });

  it("freeze=true without pretrained is a no-op combination (freeze still parsed)", () => {
    // We accept the flag but it's semantically odd — scratch+frozen means
    // "random init, never trained". Legal to request, useful for measuring
    // the random-init baseline in ablations; the benchmark script does not
    // schedule this cell but we don't gate it at the parser layer.
    expect(parseInitOpts("?freeze=true")).toEqual({
      pretrained: false,
      freeze: true,
    });
  });

  it("unknown init value falls back to scratch (strict whitelist)", () => {
    expect(parseInitOpts("?init=foo")).toEqual({
      pretrained: false,
      freeze: false,
    });
  });

  it("freeze only accepts literal 'true' (anything else is false)", () => {
    expect(parseInitOpts("?freeze=True").freeze).toBe(false);
    expect(parseInitOpts("?freeze=1").freeze).toBe(false);
    expect(parseInitOpts("?freeze=yes").freeze).toBe(false);
    expect(parseInitOpts("?freeze=true").freeze).toBe(true);
  });

  it("extra unknown params are ignored", () => {
    expect(parseInitOpts("?init=pretrained&foo=bar&freeze=true")).toEqual({
      pretrained: true,
      freeze: true,
    });
  });
});
