// Phase 4 commit 5: integration — wires charts, heatmap, stats table,
// DOM controls, and Run-all-3 to the SequentialLoop from commit 4.
//
// Control semantics (per commit 5 decision 5):
//   - Workload onchange → stop + init + start (new workload invalidates
//     accumulated metrics, so a fresh loop is correct).
//   - Active onchange    → loop.setActive() only. SequentialLoop was
//     designed to swap active without losing metric history — keeping the
//     table populated lets the user compare runs side-by-side in place.
//   - Run               → start() (no reset; accumulate over prior data).
//   - Reset             → stop + init + start (explicit clean-slate).
//   - Run all 3         → async sequence (B0 → cooldown → B1 → cooldown →
//     Predictor) with per-scheduler reset for independent runs; disables
//     other controls; Run-all-3 button morphs into Cancel.
//
// rIC render cadence (decision 2): trainTick every idle tick; paint/stats
// throttled to ≥1 Hz so a 300-point redraw + heatmap + stats rebuild never
// competes with the active frame loop.

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
import { buildStatsRows, runSequence } from "./live-controls.js";
import {
  LIVE_COOLDOWN_MS,
  LIVE_RUN_MS,
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

// --- Helpers ---------------------------------------------------------------

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

function realBusyWait(ms) {
  if (ms <= 0) return;
  const start = performance.now();
  while (performance.now() - start < ms) {
    /* burn */
  }
}

function realSleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("aborted", "AbortError"));
      return;
    }
    const tid = setTimeout(resolve, ms);
    if (signal) {
      signal.addEventListener(
        "abort",
        () => {
          clearTimeout(tid);
          reject(new DOMException("aborted", "AbortError"));
        },
        { once: true },
      );
    }
  });
}

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

function colorFor(activeName) {
  switch (activeName) {
    case "B0":
      return "#888";
    case "B1":
      return "#ff9b3d";
    case "Predictor":
    default:
      return "#4dd0e1";
  }
}

// --- Runtime state + lifecycle --------------------------------------------

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
    strokeStyle: colorFor(activeName),
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
    lastRender: 0,
  };
  renderStatsTable();
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
    // trainTick runs every idle tick (if there's budget) — cheap enough
    // that a higher cadence keeps the Predictor learning responsively.
    if (state.activeName === "Predictor" && deadline.timeRemaining() > 2) {
      state.loop.trainTick();
    }
    // Paint + stats throttled to ~1 Hz. 300-point chart redraw + 353-cell
    // heatmap + 3-row table is light, but doing it every idle tick on a
    // burst workload shows up in DevTools Performance.
    const now = performance.now();
    if (now - state.lastRender > 1000 && deadline.timeRemaining() > 1) {
      state.heatmap.update(state.loop.getParams());
      state.heatmap.render();
      state.chart.render();
      renderStatsTable();
      state.lastRender = now;
    }
    scheduleIdle();
  });
}

function start() {
  if (!state || state.running) return;
  state.running = true;
  state.lastRender = 0; // force a render on the first idle tick
  scheduleFrame();
  scheduleIdle();
  setRunStopEnabled(true);
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
  setRunStopEnabled(false);
}

function reset() {
  if (!state) return;
  stop();
  init(state.seed, state.workloadName, state.activeName);
  start();
}

// --- DOM writes -----------------------------------------------------------

function renderStatsTable() {
  if (!state) return;
  const tbody = document.querySelector("#stats-table tbody");
  if (!tbody) return;
  const rows = buildStatsRows(state.loop.getMetrics(), state.activeName);
  tbody.innerHTML = rows
    .map(
      (r) =>
        `<tr data-scheduler="${r.name}"${r.active ? ' class="active"' : ""}>` +
        `<th scope="row">${r.name}</th>` +
        `<td>${r.jankAll}</td>` +
        `<td>${r.jankRecent}</td>` +
        `<td>${r.p95All}</td>` +
        `<td>${r.p95Recent}</td>` +
        `<td>${r.meanDt}</td>` +
        `</tr>`,
    )
    .join("");
}

function setStatus(text) {
  const el = document.getElementById("status-line");
  if (!el) return;
  el.textContent = text;
  el.hidden = text.length === 0;
}

function clearStatus() {
  setStatus("");
}

function setRunStopEnabled(running) {
  const run = document.getElementById("run-btn");
  const stopBtn = document.getElementById("stop-btn");
  if (run) run.disabled = running;
  if (stopBtn) stopBtn.disabled = !running;
}

// --- Run-all-3 orchestration ----------------------------------------------

const NON_RUNALL_CONTROLS = [
  "workload-select",
  "scheduler-select",
  "run-btn",
  "stop-btn",
  "reset-btn",
];

function setControlsForRunAll(running) {
  for (const id of NON_RUNALL_CONTROLS) {
    const el = document.getElementById(id);
    if (el) el.disabled = running;
  }
}

async function handleRunAll() {
  if (!state) return;
  const btn = document.getElementById("run-all-btn");
  const originalLabel = btn.innerHTML;
  const ctrl = new AbortController();
  const onCancel = () => ctrl.abort();

  btn.innerHTML = "Cancel";
  btn.addEventListener("click", onCancel, { once: true });
  setControlsForRunAll(true);
  stop();

  const phaseLog = [];
  const onPhase = (p) => {
    if (p.kind === "started") {
      state.activeName = p.name;
      state.loop.setActive(p.name);
      // Rebuild chart for the new scheduler's color + fresh history.
      const chartCanvas = document.getElementById("chart-canvas");
      state.chart = new LineChart(chartCanvas, {
        maxPoints: 300,
        strokeStyle: colorFor(p.name),
      });
      const summary = phaseLog.length ? phaseLog.join(" ") + " " : "";
      setStatus(`${summary}Now running ${p.name} (${p.index + 1}/${p.total})`);
      start();
    } else if (p.kind === "finished") {
      stop();
      const jank =
        state.loop.getMetrics()[p.name].all.getStats().jankRate * 100;
      phaseLog.push(`${p.name} done (jank ${jank.toFixed(1)}%).`);
      setStatus(phaseLog.join(" "));
    } else if (p.kind === "cooldown") {
      setStatus(
        phaseLog.join(" ") + ` Cooling down ${(LIVE_COOLDOWN_MS / 1000) | 0}s…`,
      );
    } else if (p.kind === "complete") {
      setStatus(phaseLog.join(" ") + " Run all 3 complete.");
    }
  };

  try {
    await runSequence({
      loop: state.loop,
      signal: ctrl.signal,
      onPhase,
      runMs: LIVE_RUN_MS,
      cooldownMs: LIVE_COOLDOWN_MS,
      sleep: realSleep,
    });
  } catch (e) {
    if (e?.name !== "AbortError") throw e;
    setStatus(phaseLog.join(" ") + " Cancelled.");
    stop();
  } finally {
    btn.innerHTML = originalLabel;
    btn.removeEventListener("click", onCancel);
    setControlsForRunAll(false);
  }
}

// --- Event wiring ---------------------------------------------------------

function detectSupport() {
  return typeof window.requestIdleCallback === "function";
}

function showUnsupportedWarning() {
  const banner = document.getElementById("unsupported-browser");
  if (banner) banner.hidden = false;
  for (const id of ["run-btn", "run-all-btn", "reset-btn"]) {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  }
}

function wireControls() {
  const workloadSelect = document.getElementById("workload-select");
  const schedulerSelect = document.getElementById("scheduler-select");
  const runBtn = document.getElementById("run-btn");
  const stopBtn = document.getElementById("stop-btn");
  const resetBtn = document.getElementById("reset-btn");
  const runAllBtn = document.getElementById("run-all-btn");

  workloadSelect?.addEventListener("change", (e) => {
    clearStatus();
    stop();
    init(state.seed, e.target.value, state.activeName);
    start();
  });

  schedulerSelect?.addEventListener("change", (e) => {
    const name = e.target.value;
    state.activeName = name;
    state.loop.setActive(name);
    // Recreate chart so the line color + history match the new scheduler.
    const chartCanvas = document.getElementById("chart-canvas");
    state.chart = new LineChart(chartCanvas, {
      maxPoints: 300,
      strokeStyle: colorFor(name),
    });
    renderStatsTable();
  });

  runBtn?.addEventListener("click", () => {
    clearStatus();
    start();
  });
  stopBtn?.addEventListener("click", () => stop());
  resetBtn?.addEventListener("click", () => {
    clearStatus();
    reset();
  });
  runAllBtn?.addEventListener("click", handleRunAll);
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
  wireControls();
  window.tempo = { init, start, stop, reset, getState: () => state };
});
