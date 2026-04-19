// OnlineTrainer — SGD with momentum + L2 grad clip, zero-alloc per step.
// Spec §2.3 / §4 Phase 3 Part 3.
//
// Batch update rule:
//   1. Draw up to batchSize most-recent samples from the ring buffer.
//   2. Accumulate per-sample gradients into _accumGrads (predictor.backward()
//      returns its own reused _grads buffer, so we must copy/add BEFORE the
//      next backward() call overwrites it).
//   3. Average: _accumGrads /= B.
//   4. Compute gradNorm on the averaged gradient (returned pre-clip).
//   5. If gradNorm > gradClip: scale _accumGrads by gradClip / gradNorm.
//   6. Velocity update:  v = momentum * v - lr * _accumGrads
//      Param update:     params += v
//   (Clip is applied to the gradient only; velocity is not clipped.)
//
// Ring buffer semantics:
//   - push() copies the input features into a flat Float32Array(bufferSize×12).
//     Storing the caller's reference would leak later mutations into all
//     samples — especially when the harness reuses a single feature vector.
//   - trainStep() draws B samples WITH REPLACEMENT from the valid window
//     using the injected rng. Random sampling (vs "latest B") is chosen to
//     decorrelate successive batches — with latest-B, adjacent trainSteps
//     share 15/16 samples, which starves the optimizer of gradient
//     diversity and slows convergence enough to miss the loss<0.1 target
//     in 10k steps with lr=1e-4.
//   - Non-destructive: samples remain in the buffer for future batches.
//
// Loss tracking:
//   - _lossWindow keeps the last 64 batch-average losses for getAvgLoss().
//   - 64 is arbitrary but matches "recent" semantics for convergence reporting.

import {
  BATCH_SIZE,
  GRAD_CLIP,
  LR,
  MLP_INPUT_DIM,
  MOMENTUM,
  PARAM_COUNT,
  PRED_LOSS_EPS,
  TRAIN_BUFFER_SIZE,
} from "./constants.js";

const LOSS_WINDOW_SIZE = 64;

export class OnlineTrainer {
  /**
   * @param {import("./predictor.js").Predictor} predictor
   * @param {object} [options]
   * @param {number} [options.lr=LR]
   * @param {number} [options.momentum=MOMENTUM]
   * @param {number} [options.gradClip=GRAD_CLIP]
   * @param {number} [options.bufferSize=TRAIN_BUFFER_SIZE]
   * @param {() => number} [options.rng=Math.random] — RNG for batch sampling.
   *   Inject a seeded RNG in tests so convergence is deterministic.
   */
  constructor(
    predictor,
    {
      lr = LR,
      momentum = MOMENTUM,
      gradClip = GRAD_CLIP,
      bufferSize = TRAIN_BUFFER_SIZE,
      rng = Math.random,
    } = {},
  ) {
    this._predictor = predictor;
    this._lr = lr;
    this._momentum = momentum;
    this._gradClip = gradClip;
    this._bufferSize = bufferSize;
    this._rng = rng;

    this._features = new Float32Array(bufferSize * MLP_INPUT_DIM);
    this._targets = new Uint8Array(bufferSize);
    this._writeIdx = 0;
    this._count = 0; // monotonically increasing; capped by bufferSize on read

    this._accumGrads = new Float32Array(PARAM_COUNT);
    this._velocity = new Float32Array(PARAM_COUNT);

    this._lossWindow = new Float32Array(LOSS_WINDOW_SIZE);
    this._lossWriteIdx = 0;
    this._lossCount = 0;

    // Enabled by default. setEnabled(false) freezes learning: trainStep()
    // becomes a no-op returning null, while push() still appends (so a
    // frozen-evaluation path can keep the ring buffer warm for inspection).
    // Used by Phase 5 Part 2's pretrained+frozen benchmark condition.
    this._enabled = true;
  }

  /**
   * Toggle training on/off.
   *  - false: trainStep() returns null without touching params/velocity.
   *    push() still accepts samples (frozen-eval can keep buffer semantics).
   *  - true: trainStep() resumes normally.
   */
  setEnabled(enabled) {
    this._enabled = !!enabled;
  }

  /**
   * Copy a (features, target) sample into the ring buffer.
   * @param {Float32Array} features — length MLP_INPUT_DIM
   * @param {0|1} target
   */
  push(features, target) {
    if (features.length !== MLP_INPUT_DIM) {
      throw new Error(
        `OnlineTrainer.push: expected features length ${MLP_INPUT_DIM}, got ${features.length}`,
      );
    }
    const offset = this._writeIdx * MLP_INPUT_DIM;
    this._features.set(features, offset);
    this._targets[this._writeIdx] = target;
    this._writeIdx = (this._writeIdx + 1) % this._bufferSize;
    this._count++;
  }

  /**
   * One SGD-with-momentum step over up to `batchSize` most-recent samples.
   * Returns { loss, gradNorm } where `gradNorm` is the L2 norm of the
   * averaged gradient BEFORE clipping (clip is still applied to the update).
   * Returns null when the buffer is empty.
   *
   * @param {number} [batchSize=BATCH_SIZE]
   * @returns {{ loss: number, gradNorm: number } | null}
   */
  trainStep(batchSize = BATCH_SIZE) {
    if (!this._enabled) return null;
    const available = Math.min(this._count, this._bufferSize);
    if (available === 0) return null;
    const B = Math.min(batchSize, available);

    const predictor = this._predictor;
    const acc = this._accumGrads;
    acc.fill(0);
    let totalLoss = 0;

    for (let s = 0; s < B; s++) {
      // Sample with replacement from the valid window. When the ring has
      // wrapped, every slot in _features/_targets holds live data; before
      // wrap, only slots [0, _count) are valid.
      const sampleIdx = Math.floor(this._rng() * available);
      const offset = sampleIdx * MLP_INPUT_DIM;
      // subarray shares the underlying buffer; no allocation. backward()
      // and loss() treat the slice as the input.
      const x = this._features.subarray(offset, offset + MLP_INPUT_DIM);
      const target = this._targets[sampleIdx];

      const sampleGrads = predictor.backward(x, target);
      // CRITICAL: accumulate NOW — predictor._grads is reused on the next
      // backward() call. Don't hold a reference past this loop iteration.
      for (let j = 0; j < PARAM_COUNT; j++) acc[j] += sampleGrads[j];

      // Compute per-sample loss from the p_miss that backward() just stored.
      // (backward internally calls forward, which writes p._out.p_miss.)
      const pm = predictor._out.p_miss;
      const pc =
        pm < PRED_LOSS_EPS
          ? PRED_LOSS_EPS
          : pm > 1 - PRED_LOSS_EPS
            ? 1 - PRED_LOSS_EPS
            : pm;
      totalLoss += -(target * Math.log(pc) + (1 - target) * Math.log(1 - pc));
    }

    // Average gradient.
    const inv = 1 / B;
    for (let j = 0; j < PARAM_COUNT; j++) acc[j] *= inv;

    // Grad norm (reported pre-clip).
    let normSq = 0;
    for (let j = 0; j < PARAM_COUNT; j++) normSq += acc[j] * acc[j];
    const gradNorm = Math.sqrt(normSq);

    // Clip in place if above threshold.
    if (gradNorm > this._gradClip) {
      const scale = this._gradClip / gradNorm;
      for (let j = 0; j < PARAM_COUNT; j++) acc[j] *= scale;
    }

    // Momentum update.
    const params = predictor.params;
    const v = this._velocity;
    const mu = this._momentum;
    const lr = this._lr;
    for (let j = 0; j < PARAM_COUNT; j++) {
      v[j] = mu * v[j] - lr * acc[j];
      params[j] += v[j];
    }

    const batchLoss = totalLoss * inv;
    this._lossWindow[this._lossWriteIdx] = batchLoss;
    this._lossWriteIdx = (this._lossWriteIdx + 1) % LOSS_WINDOW_SIZE;
    if (this._lossCount < LOSS_WINDOW_SIZE) this._lossCount++;

    return { loss: batchLoss, gradNorm };
  }

  /**
   * Rolling average of the last ≤64 batch-average losses.
   * Returns 0 before any trainStep().
   */
  getAvgLoss() {
    if (this._lossCount === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this._lossCount; i++) sum += this._lossWindow[i];
    return sum / this._lossCount;
  }

  /** Number of samples currently in the ring buffer (0..bufferSize). */
  bufferCount() {
    return Math.min(this._count, this._bufferSize);
  }
}
