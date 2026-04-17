/**
 * @typedef {Object} SchedulerOptions
 * @property {'low'|'normal'|'high'} [priority] - default 'normal'. `'high'`
 *   may force the scheduler to return 'full' regardless of features.
 * @property {string} [tag] - debug/log metadata. MUST NOT affect decide()
 *   output.
 */

/**
 * @typedef {'full'|'reduce'|'degrade'} Decision
 */

/**
 * Scheduler interface (contract). Both `decide` and `onFrameComplete` are
 * called once per frame for EVERY scheduler, regardless of whether it is the
 * active (executed) scheduler or a shadow observer.
 *
 * onFrameComplete(dt, wasMiss) contract:
 *   - Called after EVERY frame, for EVERY scheduler.
 *   - `dt` reflects the ACTIVE scheduler's execution, not this scheduler's
 *     hypothetical alternative.
 *   - Schedulers MUST NOT assume this dt resulted from their own decision.
 *   - Schedulers MAY update internal state (EMA, gradients, etc.).
 *
 * @typedef {Object} Scheduler
 * @property {(features: Float32Array, options?: SchedulerOptions) => Decision} decide
 * @property {(dt: number, wasMiss: boolean) => void} onFrameComplete
 */

import {
  B1_DEGRADE_RATIO,
  B1_EMA_ALPHA,
  B1_REDUCE_RATIO,
  FRAME_BUDGET_60,
} from "./constants.js";

/**
 * B0: returns 'full' for every frame. No state, no learning. Worst-case
 * baseline against which improvements are measured.
 *
 * @implements {Scheduler}
 */
export class B0_AlwaysFull {
  // eslint-disable-next-line no-unused-vars
  decide(_features, _options) {
    return "full";
  }

  // eslint-disable-next-line no-unused-vars
  onFrameComplete(_dt, _wasMiss) {
    // no-op (stateless)
  }
}

/**
 * B1: EMA-threshold heuristic scheduler.
 *
 * decide() reads features[0] = dt_ema_fast / budgetMs (normalized per §2.2)
 * and selects:
 *   features[0] > degradeRatio → 'degrade'  (EMA already ≥ 120% of budget)
 *   features[0] > reduceRatio  → 'reduce'   (EMA at ≥ 80% of budget)
 *   else                       → 'full'
 *
 * onFrameComplete(dt, wasMiss) maintains an internal EMA of the observed dt
 * (for logging / introspection). Per the Scheduler contract, this is called
 * whether or not B1 was the active scheduler for this frame — B1 MUST NOT
 * assume `dt` resulted from its own decision. EMA update uses the standard
 * exponential recurrence:
 *
 *   ema_t = alpha * dt + (1 - alpha) * ema_{t-1},   ema_0 = 0
 *
 * @implements {Scheduler}
 */
export class B1_EmaThreshold {
  constructor(
    alpha = B1_EMA_ALPHA,
    budgetMs = FRAME_BUDGET_60,
    reduceRatio = B1_REDUCE_RATIO,
    degradeRatio = B1_DEGRADE_RATIO,
  ) {
    this._alpha = alpha;
    this._budget = budgetMs;
    this._reduceRatio = reduceRatio;
    this._degradeRatio = degradeRatio;
    this._ema = 0;
  }

  decide(features, options) {
    if (options && options.priority === "high") return "full";
    const f0 = features[0];
    if (f0 > this._degradeRatio) return "degrade";
    if (f0 > this._reduceRatio) return "reduce";
    return "full";
  }

  // eslint-disable-next-line no-unused-vars
  onFrameComplete(dt, _wasMiss) {
    this._ema = this._alpha * dt + (1 - this._alpha) * this._ema;
  }

  /** @returns {number} Current internal EMA of dt (ms). Exposed for tests. */
  getEma() {
    return this._ema;
  }
}
