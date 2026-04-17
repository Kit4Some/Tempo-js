// Predictor — 353-parameter MLP (spec §2.1, Phase 3 Part 1).
// x(12) → Linear(12→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → Sigmoid
//
// Parameters are laid out in a single Float32Array for cache locality and
// for Phase 5 Part 2 pretrained-weights inlining. Layout (offsets below):
//   W1: row-major [16 × 12]   — offset 0,   size 192
//   b1:           [16]        — offset 192, size 16
//   W2: row-major [8  × 16]   — offset 208, size 128
//   b2:           [8]         — offset 336, size 8
//   W3: row-major [1  × 8]    — offset 344, size 8
//   b3:           [1]         — offset 352, size 1
//   TOTAL                                      353
//
// Forward pass uses inline loops over pre-allocated hidden buffers so each
// call allocates zero new typed arrays. Backward pass + trainer arrive in
// Phase 3 Parts 2 and 3.

import {
  MLP_HIDDEN_1,
  MLP_HIDDEN_2,
  MLP_INPUT_DIM,
  MLP_OUTPUT_DIM,
  PARAM_COUNT,
  PRED_LOSS_EPS,
} from "./constants.js";

const W1_OFFSET = 0;
const B1_OFFSET = W1_OFFSET + MLP_HIDDEN_1 * MLP_INPUT_DIM;
const W2_OFFSET = B1_OFFSET + MLP_HIDDEN_1;
const B2_OFFSET = W2_OFFSET + MLP_HIDDEN_2 * MLP_HIDDEN_1;
const W3_OFFSET = B2_OFFSET + MLP_HIDDEN_2;
const B3_OFFSET = W3_OFFSET + MLP_OUTPUT_DIM * MLP_HIDDEN_2;

function gaussianSample() {
  // Box-Muller. Math.random avoids log(0) by mapping to (0, 1].
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class Predictor {
  constructor() {
    this.params = new Float32Array(PARAM_COUNT);

    // Pre-allocated hidden activations reused across forward calls.
    // Float64 (not Float32) because numerical-gradient precision collapses
    // under the accumulated round-off from Float32 mul-adds; params stay
    // Float32 per spec (pretrained-weight inlining, memory bound). The
    // final grads vector is written back as Float32 to match params.
    this._z1 = new Float64Array(MLP_HIDDEN_1);
    this._a1 = new Float64Array(MLP_HIDDEN_1);
    this._z2 = new Float64Array(MLP_HIDDEN_2);
    this._a2 = new Float64Array(MLP_HIDDEN_2);

    // Gradient buffers reused across backward() calls.
    this._grads = new Float32Array(PARAM_COUNT);
    this._dz1 = new Float64Array(MLP_HIDDEN_1);
    this._dz2 = new Float64Array(MLP_HIDDEN_2);

    // Return payload; mutated in place each forward() so we never allocate.
    this._hidden = {
      z1: this._z1,
      a1: this._a1,
      z2: this._z2,
      a2: this._a2,
    };
    this._out = { p_miss: 0, hidden: this._hidden };

    this._initHe();
  }

  _initHe() {
    const p = this.params;
    // W1: N(0, sqrt(2/fan_in)), fan_in = MLP_INPUT_DIM
    const s1 = Math.sqrt(2 / MLP_INPUT_DIM);
    for (let i = W1_OFFSET; i < B1_OFFSET; i++) p[i] = gaussianSample() * s1;
    // W2: fan_in = MLP_HIDDEN_1
    const s2 = Math.sqrt(2 / MLP_HIDDEN_1);
    for (let i = W2_OFFSET; i < B2_OFFSET; i++) p[i] = gaussianSample() * s2;
    // W3: fan_in = MLP_HIDDEN_2
    const s3 = Math.sqrt(2 / MLP_HIDDEN_2);
    for (let i = W3_OFFSET; i < B3_OFFSET; i++) p[i] = gaussianSample() * s3;
    // Biases initialized to 0 (Float32Array zero-fills on construction).
  }

  /**
   * Forward pass. Zero allocation per call (hidden buffers and the return
   * object are reused from the constructor).
   *
   * @param {Float32Array} x — length MLP_INPUT_DIM (12)
   * @returns {{ p_miss: number, hidden: { z1, a1, z2, a2 } }}
   * @throws if x.length !== MLP_INPUT_DIM
   */
  forward(x) {
    if (x.length !== MLP_INPUT_DIM) {
      throw new Error(
        `Predictor.forward: expected input length ${MLP_INPUT_DIM}, got ${x.length}`,
      );
    }

    const p = this.params;
    const z1 = this._z1;
    const a1 = this._a1;
    const z2 = this._z2;
    const a2 = this._a2;

    // Layer 1: z1 = W1·x + b1, a1 = relu(z1)
    for (let i = 0; i < MLP_HIDDEN_1; i++) {
      let s = p[B1_OFFSET + i];
      const rowBase = W1_OFFSET + i * MLP_INPUT_DIM;
      for (let j = 0; j < MLP_INPUT_DIM; j++) {
        s += p[rowBase + j] * x[j];
      }
      z1[i] = s;
      a1[i] = s > 0 ? s : 0;
    }

    // Layer 2: z2 = W2·a1 + b2, a2 = relu(z2)
    for (let i = 0; i < MLP_HIDDEN_2; i++) {
      let s = p[B2_OFFSET + i];
      const rowBase = W2_OFFSET + i * MLP_HIDDEN_1;
      for (let j = 0; j < MLP_HIDDEN_1; j++) {
        s += p[rowBase + j] * a1[j];
      }
      z2[i] = s;
      a2[i] = s > 0 ? s : 0;
    }

    // Output: z3 = W3·a2 + b3, p_miss = sigmoid(z3)
    let z3 = p[B3_OFFSET];
    for (let j = 0; j < MLP_HIDDEN_2; j++) {
      z3 += p[W3_OFFSET + j] * a2[j];
    }
    const pMiss = 1 / (1 + Math.exp(-z3));

    this._out.p_miss = pMiss;
    return this._out;
  }

  /**
   * Binary cross-entropy loss. p_miss from forward() is clamped to
   * [PRED_LOSS_EPS, 1 - PRED_LOSS_EPS] before taking log — the same clamp is
   * applied inside backward()'s dL/dz3 computation so that numerical and
   * analytic gradients reference the same p.
   *
   * @param {Float32Array} x — length MLP_INPUT_DIM
   * @param {0|1} target
   * @returns {number} BCE loss
   */
  loss(x, target) {
    const { p_miss } = this.forward(x);
    const p = p_miss < PRED_LOSS_EPS
      ? PRED_LOSS_EPS
      : p_miss > 1 - PRED_LOSS_EPS
        ? 1 - PRED_LOSS_EPS
        : p_miss;
    return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
  }

  /**
   * Backward pass: computes ∂L/∂θ for every parameter using the chain rule.
   * Writes into the pre-allocated this._grads buffer and returns it.
   * Zero allocation per call.
   *
   * Uses the same clamp as loss() when computing dL/dz3 so that numerical
   * and analytic gradients reference the same p (keeps gradcheck honest).
   *
   * Equations:
   *   dL/dz3 = p_clamped - target                 (BCE ∘ sigmoid simplification)
   *   dL/dW3[j] = dL/dz3 * a2[j]
   *   dL/db3    = dL/dz3
   *   dL/dz2[i] = (z2[i] > 0 ? 1 : 0) * dL/dz3 * W3[i]
   *   dL/dW2[i,j] = dL/dz2[i] * a1[j]
   *   dL/db2[i]   = dL/dz2[i]
   *   dL/dz1[i]   = (z1[i] > 0 ? 1 : 0) * Σ_k dL/dz2[k] * W2[k, i]
   *   dL/dW1[i,j] = dL/dz1[i] * x[j]
   *   dL/db1[i]   = dL/dz1[i]
   *
   * @param {Float32Array} x — length MLP_INPUT_DIM
   * @param {0|1} target
   * @returns {Float32Array} length PARAM_COUNT, same layout as this.params
   * @throws if x.length !== MLP_INPUT_DIM
   */
  backward(x, target) {
    // forward() guards the input length; we don't re-guard here.
    const { p_miss } = this.forward(x);

    const p = this.params;
    const z1 = this._z1;
    const a1 = this._a1;
    const z2 = this._z2;
    const a2 = this._a2;
    const dz1 = this._dz1;
    const dz2 = this._dz2;
    const grads = this._grads;

    // Clamp p_miss consistently with loss().
    const pc = p_miss < PRED_LOSS_EPS
      ? PRED_LOSS_EPS
      : p_miss > 1 - PRED_LOSS_EPS
        ? 1 - PRED_LOSS_EPS
        : p_miss;
    const dz3 = pc - target; // scalar

    // Output layer: dL/dW3, dL/db3, and gather dz2.
    for (let j = 0; j < MLP_HIDDEN_2; j++) {
      grads[W3_OFFSET + j] = dz3 * a2[j];
    }
    grads[B3_OFFSET] = dz3;

    // Layer 2 pre-activation gradient: ReLU mask × W3 × dz3.
    for (let i = 0; i < MLP_HIDDEN_2; i++) {
      dz2[i] = z2[i] > 0 ? dz3 * p[W3_OFFSET + i] : 0;
    }

    // Layer 2 weight/bias gradients.
    for (let i = 0; i < MLP_HIDDEN_2; i++) {
      const rowBase = W2_OFFSET + i * MLP_HIDDEN_1;
      const d = dz2[i];
      for (let j = 0; j < MLP_HIDDEN_1; j++) {
        grads[rowBase + j] = d * a1[j];
      }
      grads[B2_OFFSET + i] = d;
    }

    // Layer 1 pre-activation gradient: ReLU mask × (W2ᵀ · dz2).
    for (let i = 0; i < MLP_HIDDEN_1; i++) {
      if (z1[i] > 0) {
        let s = 0;
        for (let k = 0; k < MLP_HIDDEN_2; k++) {
          s += dz2[k] * p[W2_OFFSET + k * MLP_HIDDEN_1 + i];
        }
        dz1[i] = s;
      } else {
        dz1[i] = 0;
      }
    }

    // Layer 1 weight/bias gradients.
    for (let i = 0; i < MLP_HIDDEN_1; i++) {
      const rowBase = W1_OFFSET + i * MLP_INPUT_DIM;
      const d = dz1[i];
      for (let j = 0; j < MLP_INPUT_DIM; j++) {
        grads[rowBase + j] = d * x[j];
      }
      grads[B1_OFFSET + i] = d;
    }

    return grads;
  }
}
