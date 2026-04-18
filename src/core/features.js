// FeatureExtractor — turns frame-level observations into the 12-dim Float32
// feature vector consumed by the Predictor (spec §2.2).
//
// Field semantics inside observe():
//   dt                  REQUIRED. Frame duration in ms. Missing → throw.
//   scrollVelocity      rate-like. Missing → 0 (treat as "no activity").
//   inputEvents         rate-like. Missing → 0.
//   domMutations        state-like. Missing → preserve previous value.
//   visibleAnimating    state-like. Missing → preserve previous value.
//
// Rate-like defaults to 0 because not-reporting a rate means "nothing
// happened this frame". State-like preserves because a page that declares
// 3 active animations once doesn't usually re-emit that every frame.
//
// Two fields are best-effort (spec §2.2 and noted in the blog's limitations):
//   gc_pressure     requires performance.memory (Chrome-only). 0 elsewhere.
//   workload_delta  passed per extract() call; callers that don't know
//                   their workload leave it at 0.

import {
  DEVICE_TIER_DEFAULT,
  DEVICE_TIER_LOW_MAX,
  DEVICE_TIER_MID_MAX,
  DT_EMA_FAST_ALPHA,
  DT_EMA_SLOW_ALPHA,
  DT_WINDOW_MISS,
  DT_WINDOW_SHORT,
  FRAME_BUDGET_60,
  JANK_TOLERANCE_MS,
  MLP_INPUT_DIM,
} from "./constants.js";

// Clamp the EMA-smoothed inputEvents rate so the feature stays in [0, 1]
// without a rigid per-event cap. 10 events/frame is a generous saturation
// point (scroll + multiple keys + mousemove).
const INPUT_ACTIVITY_SAT = 10;

function resolveDeviceTier(hardwareConcurrency) {
  if (hardwareConcurrency === undefined || hardwareConcurrency === null) {
    return DEVICE_TIER_DEFAULT;
  }
  if (hardwareConcurrency <= DEVICE_TIER_LOW_MAX) return 0;
  if (hardwareConcurrency <= DEVICE_TIER_MID_MAX) return 1;
  return 2;
}

export class FeatureExtractor {
  /**
   * @param {number} [budgetMs=FRAME_BUDGET_60]
   * @param {object} [env]
   * @param {number} [env.viewportH] — px. Falls back to window.innerHeight,
   *   then 1000 in headless contexts.
   * @param {number} [env.hardwareConcurrency] — navigator.hardwareConcurrency
   *   override; useful in tests.
   * @param {(() => number) | null} [env.getMemoryUsed] — returns current
   *   JS heap bytes. Pass `null` to force "unsupported" (gc_pressure = 0).
   *   Defaults to performance.memory.usedJSHeapSize when available.
   */
  constructor(budgetMs = FRAME_BUDGET_60, env = {}) {
    this._budget = budgetMs;

    const { viewportH, getMemoryUsed } = env;

    this._viewportH =
      viewportH ??
      ((typeof window !== "undefined" && window.innerHeight) || 1000);

    // Distinguish "key absent" (no info → use navigator) from "key present with
    // any value, including undefined" (caller is explicit → respect it).
    const hc = "hardwareConcurrency" in env
      ? env.hardwareConcurrency
      : typeof navigator !== "undefined"
        ? navigator.hardwareConcurrency
        : undefined;
    this._deviceTier = resolveDeviceTier(hc);

    if (getMemoryUsed === null) {
      this._getMemoryUsed = null;
    } else if (typeof getMemoryUsed === "function") {
      this._getMemoryUsed = getMemoryUsed;
    } else if (
      typeof performance !== "undefined" &&
      performance.memory &&
      typeof performance.memory.usedJSHeapSize === "number"
    ) {
      this._getMemoryUsed = () => performance.memory.usedJSHeapSize;
    } else {
      this._getMemoryUsed = null;
    }
    this._lastMemory = null;

    // dt state — normalized EMAs + raw ring buffer for windowed stats.
    this._dtEmaFast = 0;
    this._dtEmaSlow = 0;
    this._dtBuffer = new Float32Array(DT_WINDOW_MISS);
    this._dtWriteIdx = 0;
    this._dtCount = 0;

    // Rate-like observations.
    this._inputActivity = 0; // EMA of inputEvents per frame
    this._scrollVelocity = 0; // last observed (rate-like, 0 if absent)

    // State-like observations.
    this._domMutations = 0;
    this._visibleAnimating = 0;

    // Pre-allocated output.
    this._out = new Float32Array(MLP_INPUT_DIM);
  }

  /**
   * @param {object} event
   * @param {number} event.dt — REQUIRED, ms.
   * @param {number} [event.domMutations] — state-like.
   * @param {number} [event.inputEvents] — rate-like.
   * @param {number} [event.scrollVelocity] — rate-like (signed px/frame).
   * @param {number} [event.visibleAnimating] — state-like.
   */
  observe(event) {
    if (
      !event ||
      typeof event.dt !== "number" ||
      !Number.isFinite(event.dt)
    ) {
      throw new Error("FeatureExtractor.observe: finite dt is required");
    }

    const dt = event.dt;
    const normDt = dt / this._budget;

    // EMA updates on normalized dt.
    this._dtEmaFast =
      DT_EMA_FAST_ALPHA * normDt + (1 - DT_EMA_FAST_ALPHA) * this._dtEmaFast;
    this._dtEmaSlow =
      DT_EMA_SLOW_ALPHA * normDt + (1 - DT_EMA_SLOW_ALPHA) * this._dtEmaSlow;

    // Raw dt ring buffer.
    this._dtBuffer[this._dtWriteIdx] = dt;
    this._dtWriteIdx = (this._dtWriteIdx + 1) % DT_WINDOW_MISS;
    if (this._dtCount < DT_WINDOW_MISS) this._dtCount++;

    // Rate-like: default to 0 when caller omits the field.
    const iev = typeof event.inputEvents === "number" ? event.inputEvents : 0;
    this._inputActivity =
      DT_EMA_FAST_ALPHA * iev + (1 - DT_EMA_FAST_ALPHA) * this._inputActivity;
    this._scrollVelocity =
      typeof event.scrollVelocity === "number" ? event.scrollVelocity : 0;

    // State-like: preserve previous value on omission.
    if (typeof event.domMutations === "number") {
      this._domMutations = event.domMutations;
    }
    if (typeof event.visibleAnimating === "number") {
      this._visibleAnimating = event.visibleAnimating;
    }
  }

  /**
   * Build the 12-dim feature vector from the current state.
   * Reuses a pre-allocated Float32Array — snapshot if you need to retain
   * values across calls.
   *
   * @param {number} [workloadDeltaMs=0] — caller-provided extra work budget.
   * @returns {Float32Array} length MLP_INPUT_DIM (12)
   */
  extract(workloadDeltaMs = 0) {
    const out = this._out;
    const budget = this._budget;
    const window = Math.min(this._dtCount, DT_WINDOW_SHORT);
    const missWindow = Math.min(this._dtCount, DT_WINDOW_MISS);

    // 1, 2: normalized dt EMAs.
    out[0] = this._dtEmaFast;
    out[1] = this._dtEmaSlow;

    // 3, 4: variance and max over the last DT_WINDOW_SHORT raw dts.
    //       Walking the buffer once for the mean, again for the variance and
    //       max to avoid numerical bias from single-pass variance formulas.
    if (window > 0) {
      let sum = 0;
      for (let i = 0; i < window; i++) {
        const idx =
          (this._dtWriteIdx - 1 - i + DT_WINDOW_MISS) % DT_WINDOW_MISS;
        sum += this._dtBuffer[idx];
      }
      const mean = sum / window;
      let varAcc = 0;
      let maxDt = 0;
      for (let i = 0; i < window; i++) {
        const idx =
          (this._dtWriteIdx - 1 - i + DT_WINDOW_MISS) % DT_WINDOW_MISS;
        const d = this._dtBuffer[idx];
        const diff = d - mean;
        varAcc += diff * diff;
        if (d > maxDt) maxDt = d;
      }
      out[2] = varAcc / window / (budget * budget);
      out[3] = maxDt / budget;
    } else {
      out[2] = 0;
      out[3] = 0;
    }

    // 5: miss rate over DT_WINDOW_MISS frames (unitless in [0, 1]).
    if (missWindow > 0) {
      let misses = 0;
      for (let i = 0; i < missWindow; i++) {
        const idx =
          (this._dtWriteIdx - 1 - i + DT_WINDOW_MISS) % DT_WINDOW_MISS;
        if (this._dtBuffer[idx] > budget + JANK_TOLERANCE_MS) misses++;
      }
      out[4] = misses / missWindow;
    } else {
      out[4] = 0;
    }

    // 6: gc pressure — log of positive JS heap delta since last extract().
    if (this._getMemoryUsed) {
      const cur = this._getMemoryUsed();
      const delta =
        this._lastMemory !== null ? Math.max(0, cur - this._lastMemory) : 0;
      out[5] = Math.log(1 + delta);
      this._lastMemory = cur;
    } else {
      out[5] = 0;
    }

    // 7: input activity — EMA-smoothed events, saturating at 1.
    const ia = this._inputActivity / INPUT_ACTIVITY_SAT;
    out[6] = ia > 1 ? 1 : ia < 0 ? 0 : ia;

    // 8: scroll velocity, abs, normalized by viewport height.
    out[7] = Math.abs(this._scrollVelocity) / this._viewportH;

    // 9, 10: log(1+n) for integer counts.
    out[8] = Math.log(1 + Math.max(0, this._visibleAnimating));
    out[9] = Math.log(1 + Math.max(0, this._domMutations));

    // 11: workload delta normalized by budget.
    out[10] = workloadDeltaMs / budget;

    // 12: device tier scalar (spec §2.2 uses {0, 1, 2}, not one-hot).
    out[11] = this._deviceTier;

    return out;
  }
}
