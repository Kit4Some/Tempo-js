import {
  FRAME_BUDGET_60,
  JANK_TOLERANCE_MS,
  METRICS_BUFFER_SIZE,
} from "../core/constants.js";
import { RollingFrameMetrics } from "./metrics.js";
import { workCostFor } from "./work-cost.js";

/**
 * Sequential benchmark loop (spec §4 Phase 4, decision 1).
 *
 * One active scheduler executes its decision on every frame; the other two
 * (shadow) still receive decide() + onFrameComplete() per the Scheduler
 * contract (§4 Phase 2), but their decisions are NOT applied to the frame
 * cost and their metrics are NOT updated — the shadow dt would just be the
 * active scheduler's execution, which is not a fair attribution (§6.3).
 *
 * The loop is explicitly step()-driven rather than rAF-driven so it can be
 * reused in Phase 5's Puppeteer benchmark (where timing is fully manual)
 * and in unit tests (where a fake clock advances time synchronously).
 *
 * Construction takes a `buildState` factory — reset() re-invokes it so
 * seeded-RNG tests get byte-identical re-initialization. The factory must
 * return { schedulers: {B0, B1, Predictor}, extractor, trainer, predictor }.
 */
export class SequentialLoop {
  constructor({
    buildState,
    workload,
    busyWait,
    now,
    initialActive = "Predictor",
  }) {
    if (typeof buildState !== "function") {
      throw new TypeError("SequentialLoop: buildState must be a function");
    }
    if (typeof workload !== "function") {
      throw new TypeError(
        "SequentialLoop: workload must be a function (frameIdx) => baseCostMs",
      );
    }
    if (typeof busyWait !== "function" || typeof now !== "function") {
      throw new TypeError(
        "SequentialLoop: busyWait and now must be injected (no defaults)",
      );
    }
    this._buildState = buildState;
    this._workload = workload;
    this._busyWait = busyWait;
    this._now = now;
    this._active = initialActive;
    this._buildInternal();
  }

  _buildInternal() {
    this._state = this._buildState();
    this._frameIdx = 0;
    this._lastNow = null;
    // Per-scheduler metrics. "all" uses METRICS_BUFFER_SIZE (1024) as its
    // window — the live UI never sees more than that historically anyway,
    // and it keeps both tracks in the same Rolling* abstraction.
    this._metrics = {
      B0: {
        all: new RollingFrameMetrics(METRICS_BUFFER_SIZE),
        recent: new RollingFrameMetrics(300),
      },
      B1: {
        all: new RollingFrameMetrics(METRICS_BUFFER_SIZE),
        recent: new RollingFrameMetrics(300),
      },
      Predictor: {
        all: new RollingFrameMetrics(METRICS_BUFFER_SIZE),
        recent: new RollingFrameMetrics(300),
      },
    };
  }

  setActive(name) {
    if (!(name in this._state.schedulers)) {
      throw new RangeError(`SequentialLoop.setActive: unknown scheduler ${name}`);
    }
    this._active = name;
  }

  getActive() {
    return this._active;
  }

  getMetrics() {
    return this._metrics;
  }

  getParams() {
    return this._state.predictor.params;
  }

  reset() {
    this._buildInternal();
  }

  /**
   * Advance the loop by one frame. Returns an object describing what
   * happened: { dt, wasMiss, decisions, features }.
   *
   * Order of operations per spec §4 Phase 4 pseudocode:
   *   1. Compute dt from now() since the previous step (0 on the first step).
   *   2. Record dt into the active scheduler's metrics.
   *   3. observe({ dt, ... }) — feeds the extractor.
   *   4. onFrameComplete(dt, wasMiss) on ALL schedulers (contract §4 Phase 2).
   *   5. extract() — one shared Float32Array reference.
   *   6. decide(features) on ALL schedulers. Shadow decisions are preserved
   *      in the return value but not executed.
   *   7. busyWait(baseCost × workCostFor(activeDecision)). This shows up in
   *      the NEXT step's dt — matches how a real rAF frame's paint cost
   *      shows up as the following rAF callback's timestamp delta.
   */
  step(observeExtras = {}) {
    const now1 = this._now();
    const dt = this._lastNow == null ? 0 : now1 - this._lastNow;
    const wasMiss = dt > FRAME_BUDGET_60 + JANK_TOLERANCE_MS;

    this._metrics[this._active].all.record(dt);
    this._metrics[this._active].recent.record(dt);

    this._state.extractor.observe({ dt, ...observeExtras });
    const schedulers = this._state.schedulers;
    for (const name of ["B0", "B1", "Predictor"]) {
      schedulers[name].onFrameComplete(dt, wasMiss);
    }

    const features = this._state.extractor.extract(0);
    const decisions = {};
    for (const name of ["B0", "B1", "Predictor"]) {
      decisions[name] = schedulers[name].decide(features);
    }

    const activeDecision = decisions[this._active];
    const cost = workCostFor(activeDecision);
    const baseMs = this._workload(this._frameIdx);
    this._busyWait(baseMs * cost);

    this._lastNow = now1;
    this._frameIdx++;

    return { dt, wasMiss, decisions, features };
  }

  /**
   * Run one trainer step. Returns whatever OnlineTrainer.trainStep returns
   * ({ loss, gradNorm } or null when the buffer is empty). Intentionally
   * out-of-band from step() so the UI can choose its own cadence
   * (requestIdleCallback in Phase 4, manual in Phase 5).
   */
  trainTick(batchSize) {
    return this._state.trainer.trainStep(batchSize);
  }
}
