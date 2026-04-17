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
    this._z1 = new Float32Array(MLP_HIDDEN_1);
    this._a1 = new Float32Array(MLP_HIDDEN_1);
    this._z2 = new Float32Array(MLP_HIDDEN_2);
    this._a2 = new Float32Array(MLP_HIDDEN_2);

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
}
