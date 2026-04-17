import { B0_AlwaysFull, B1_EmaThreshold } from "../src/core/schedulers.js";
import { FRAME_BUDGET_60 } from "../src/core/constants.js";
import { FrameMetrics } from "../src/harness/metrics.js";
import {
  burst,
  constant,
  sawtooth,
  scrollCorrelated,
} from "../src/harness/workloads.js";

// Fake scroll velocity for the `scroll` workload: oscillates 0..60 over ~2s
// at 60fps. Phase 4 will replace this with actual scroll input.
let fakeScrollT = 0;
function fakeVelocity() {
  fakeScrollT += 1;
  const phase = (fakeScrollT % 120) / 120;
  return 60 * Math.abs(Math.sin(Math.PI * phase));
}

const workloads = {
  constant: () => constant(),
  sawtooth: () => sawtooth(),
  burst: () => burst(),
  scroll: () => scrollCorrelated(fakeVelocity),
};

const featureBuf = new Float32Array(12);

let currentWorkloadName = "constant";
let currentWorkload = workloads[currentWorkloadName]();
let metrics = null;
let scheduler = null;
let logInterval = null;

function buildScheduler(name) {
  return name === "B0" ? new B0_AlwaysFull() : new B1_EmaThreshold();
}

function computeFeatureZero() {
  // Phase 2 shim: piggyback on B1's internal EMA as the normalized feature.
  // Phase 3 replaces this with FeatureExtractor.extract()[0].
  const emaMs = typeof scheduler.getEma === "function" ? scheduler.getEma() : 0;
  return emaMs / FRAME_BUDGET_60;
}

function start() {
  if (metrics) return; // already running

  const schedulerSelect = document.getElementById("scheduler");
  const decisionEl = document.getElementById("decision");
  const framesEl = document.getElementById("frames");
  const schedName = schedulerSelect.value;
  scheduler = buildScheduler(schedName);
  metrics = new FrameMetrics();

  metrics.onFrame((idx, dt) => {
    scheduler.onFrameComplete(dt, dt > FRAME_BUDGET_60);
    featureBuf[0] = computeFeatureZero();
    const decision = scheduler.decide(featureBuf, {
      tag: currentWorkloadName,
    });
    decisionEl.textContent = decision;
    framesEl.textContent = String(idx + 1);
    currentWorkload(idx);
  });

  metrics.start();
  logInterval = setInterval(() => {
    if (!metrics) return;
    const s = metrics.getStats();
    console.log("[tempo]", {
      scheduler: schedName,
      workload: currentWorkloadName,
      jankRate: +s.jankRate.toFixed(3),
      p95: +s.p95.toFixed(2),
    });
  }, 1000);
}

function stop() {
  if (metrics) {
    metrics.stop();
    metrics = null;
  }
  if (logInterval) {
    clearInterval(logInterval);
    logInterval = null;
  }
}

function selectWorkload(name, clickedBtn) {
  currentWorkloadName = name;
  currentWorkload = workloads[name]();
  document.querySelectorAll("[data-workload]").forEach((b) => {
    b.removeAttribute("data-active");
  });
  clickedBtn.setAttribute("data-active", "true");
  document.getElementById("current-workload").textContent = name;
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-workload]").forEach((btn) => {
    btn.addEventListener("click", () => selectWorkload(btn.dataset.workload, btn));
  });
  document.getElementById("start").addEventListener("click", start);
  document.getElementById("stop").addEventListener("click", stop);
});
