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
  MLP_INPUT_DIM,
  PRED_DEGRADE_THRESHOLD,
  PRED_REDUCE_THRESHOLD,
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

/**
 * PredictorScheduler: Scheduler adapter around a Predictor + OnlineTrainer.
 *
 * Design (spec §4 Phase 3 Part 4):
 *   - Injects predictor + trainer. No internal FeatureExtractor — the harness
 *     owns extraction so all three schedulers (B0/B1/Predictor) see the same
 *     features vector per frame, which is a prerequisite for the Sequential +
 *     Shadow benchmark (§5 / §4 Phase 5).
 *   - decide() copies the feature vector into _lastFeatures BEFORE the
 *     priority override check; onFrameComplete() pairs that cached vector
 *     with (wasMiss ? 1 : 0) and pushes into the trainer. The copy-before-
 *     override order is load-bearing: a 'high' priority frame still produces
 *     a valid training sample.
 *   - trainStep() is the HARNESS's responsibility (call it from
 *     requestIdleCallback, or on an arbitrary cadence). Keeping it out of
 *     the adapter lets Phase 5 ablate "train every frame" vs "train every
 *     100 frames" without changing adapter internals.
 *   - First-frame guard: if onFrameComplete fires before any decide() (can
 *     happen when a scheduler is observed in shadow before its first active
 *     invocation, or simply at startup), the push is skipped — we have no
 *     features to pair with the observed dt.
 *
 * @implements {Scheduler}
 */
export class PredictorScheduler {
  /**
   * @param {object} deps
   * @param {import("./predictor.js").Predictor} deps.predictor
   * @param {import("./trainer.js").OnlineTrainer} deps.trainer
   */
  constructor({ predictor, trainer }) {
    this.predictor = predictor;
    this.trainer = trainer;
    this._lastFeatures = new Float32Array(MLP_INPUT_DIM);
    this._hasLastFeatures = false;
  }

  /**
   * @param {Float32Array} features — length MLP_INPUT_DIM (§2.2)
   * @param {SchedulerOptions} [options]
   * @returns {Decision}
   */
  decide(features, options) {
    // Copy features first — onFrameComplete() needs this even when priority
    // override short-circuits the forward pass below.
    this._lastFeatures.set(features);
    this._hasLastFeatures = true;

    if (options && options.priority === "high") return "full";

    this.predictor.forward(features);
    const p = this.predictor._out.p_miss;

    if (p > PRED_DEGRADE_THRESHOLD) return "degrade";
    if (p > PRED_REDUCE_THRESHOLD) return "reduce";
    return "full";
  }

  /**
   * Scheduler contract (spec §4 Phase 2):
   *   - Called after EVERY frame, for EVERY scheduler.
   *   - `dt` reflects the ACTIVE scheduler's execution, not this scheduler's
   *     hypothetical alternative.
   *   - Schedulers MUST NOT assume this dt resulted from their own decision.
   *
   * We push (_lastFeatures, wasMiss ? 1 : 0) into the trainer so the
   * Predictor learns from every observed outcome, active or shadow. This is
   * the implementation origin of Phase 6's "we train from counterfactual
   * observations" limitation — do not remove or weaken this comment when
   * refactoring.
   *
   * @param {number} dt
   * @param {boolean} wasMiss
   */
  // eslint-disable-next-line no-unused-vars
  onFrameComplete(dt, wasMiss) {
    if (!this._hasLastFeatures) return; // first-frame guard
    this.trainer.push(this._lastFeatures, wasMiss ? 1 : 0);
    // trainStep() is NOT called here — harness owns the training cadence.
  }
}
