// Phase 4 commit 4: frame loop + shared FeatureExtractor + Sequential
// execution wiring. DOM controls (button clicks, select onchange, Run-all-3
// orchestration) are deliberately deferred to commit 5 — this commit
// auto-starts the default configuration (workload=constant, active=Predictor
// from the HTML select defaults) and exposes `window.tempo` for console
// debugging so commit 5 can focus on UI binding.

import { SequentialLoop } from "../src/harness/sequential-loop.js";
import { Predictor } from "../src/core/predictor.js";
import { OnlineTrainer } from "../src/core/trainer.js";
import { FeatureExtractor } from "../src/core/features.js";
import {
  B0_AlwaysFull,
  B1_EmaThreshold,
  PredictorScheduler,
} from "../src/core/schedulers.js";
import { LineChart, LayeredHeatmap, makeDefaultLayers } from "./charts.js";
import {
  WL_BURST_BASE_MS,
  WL_BURST_SPIKE_DURATION,
  WL_BURST_SPIKE_EVERY,
  WL_BURST_SPIKE_MS,
  WL_CONSTANT_MS,
  WL_SAWTOOTH_MAX,
  WL_SAWTOOTH_MIN,
  WL_SAWTOOTH_PERIOD,
  WL_SCROLL_K,
} from "../src/core/constants.js";

// Mulberry32 — 3-line seeded PRNG. Inlined here rather than imported from
// tests/helpers so there's no src → tests dependency; if a third caller
// emerges, promote to src/core/rng.js.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Production busyWait. Workloads.js has a private copy for its own
// in-module cost functions — reproducing it here keeps this file
// standalone without exposing internal workload helpers.
function realBusyWait(ms) {
  if (ms <= 0) return;
  const start = performance.now();
  while (performance.now() - start < ms) {
    /* burn */
  }
}

// Map a workload select value to a cost-only function (no busyWait inside
// — SequentialLoop applies workCostFor() multiplier and calls busyWait
// itself). Math mirrors workloads.js; a future refactor may consolidate.
let fakeScrollT = 0;
function scrollVelocity() {
  fakeScrollT += 1;
  const phase = (fakeScrollT % 120) / 120;
  return 60 * Math.abs(Math.sin(Math.PI * phase));
}

function workloadCostFor(name) {
  switch (name) {
    case "constant":
      return () => WL_CONSTANT_MS;
    case "sawtooth": {
      const span = WL_SAWTOOTH_MAX - WL_SAWTOOTH_MIN;
      return (i) => {
        const phase =
          ((i % WL_SAWTOOTH_PERIOD) + WL_SAWTOOTH_PERIOD) % WL_SAWTOOTH_PERIOD;
        return WL_SAWTOOTH_MIN + span * (phase / WL_SAWTOOTH_PERIOD);
      };
    }
    case "burst":
      return (i) => {
        const inSpike =
          i >= WL_BURST_SPIKE_EVERY &&
          ((i % WL_BURST_SPIKE_EVERY) + WL_BURST_SPIKE_EVERY) %
            WL_BURST_SPIKE_EVERY <
            WL_BURST_SPIKE_DURATION;
        return inSpike ? WL_BURST_SPIKE_MS : WL_BURST_BASE_MS;
      };
    case "scroll":
      fakeScrollT = 0;
      return () => Math.abs(scrollVelocity()) * WL_SCROLL_K;
    default:
      throw new Error(`workloadCostFor: unknown workload '${name}'`);
  }
}

function makeFactory(seed) {
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

// Module-level runtime handle. Commit 5's DOM handlers will mutate .active
// and .workload via start/stop/reset cycles.
let state = null;

function init(seed = 42, workloadName = "constant", activeName = "Predictor") {
  stop();
  const loop = new SequentialLoop({
    buildState: makeFactory(seed),
    workload: workloadCostFor(workloadName),
    busyWait: realBusyWait,
    now: () => performance.now(),
    initialActive: activeName,
  });
  const chartCanvas = document.getElementById("chart-canvas");
  const heatmapCanvas = document.getElementById("heatmap-canvas");
  const chart = new LineChart(chartCanvas, {
    maxPoints: 300,
    strokeStyle: activeName === "B0" ? "#888" : activeName === "B1" ? "#ff9b3d" : "#4dd0e1",
  });
  const heatmap = new LayeredHeatmap(heatmapCanvas, {
    layers: makeDefaultLayers(),
  });
  state = {
    loop,
    chart,
    heatmap,
    workloadName,
    activeName,
    seed,
    running: false,
    rafId: null,
    ricId: null,
  };
  return state;
}

function scheduleFrame() {
  if (!state || !state.running) return;
  state.rafId = requestAnimationFrame(() => {
    const r = state.loop.step();
    state.chart.push(r.dt);
    scheduleFrame();
  });
}

function scheduleIdle() {
  if (!state || !state.running) return;
  state.ricId = requestIdleCallback((deadline) => {
    // trainTick first (cheaper than paint + has upstream gradient work).
    if (state.activeName === "Predictor" && deadline.timeRemaining() > 2) {
      state.loop.trainTick();
    }
    // Heatmap + chart repaint. rIC cadence keeps paint off the rAF critical
    // path so the benchmark page doesn't self-referentially jank.
    if (deadline.timeRemaining() > 1) {
      state.heatmap.update(state.loop.getParams());
      state.heatmap.render();
      state.chart.render();
    }
    scheduleIdle();
  });
}

function start() {
  if (!state || state.running) return;
  state.running = true;
  scheduleFrame();
  scheduleIdle();
}

function stop() {
  if (!state) return;
  state.running = false;
  if (state.rafId != null) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
  if (state.ricId != null) {
    cancelIdleCallback(state.ricId);
    state.ricId = null;
  }
}

function detectSupport() {
  return typeof window.requestIdleCallback === "function";
}

function showUnsupportedWarning() {
  const banner = document.getElementById("unsupported-browser");
  if (banner) banner.hidden = false;
  for (const id of ["run-btn", "run-all-btn"]) {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (!detectSupport()) {
    showUnsupportedWarning();
    return;
  }
  const workloadName =
    document.getElementById("workload-select")?.value ?? "constant";
  const activeName =
    document.getElementById("scheduler-select")?.value ?? "Predictor";
  init(42, workloadName, activeName);
  start();
  // Expose for console debugging + commit 5 DOM handler wiring.
  window.tempo = {
    init,
    start,
    stop,
    getState: () => state,
  };
});
