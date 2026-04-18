import {
  B1_OFFSET,
  B2_OFFSET,
  B3_OFFSET,
  FRAME_BUDGET_60,
  MLP_HIDDEN_1,
  MLP_HIDDEN_2,
  MLP_INPUT_DIM,
  MLP_OUTPUT_DIM,
  PARAM_COUNT,
  W1_OFFSET,
  W2_OFFSET,
  W3_OFFSET,
} from "../src/core/constants.js";
import { divergingColor } from "./colormap.js";

/**
 * Canvas-free ring buffer of the most recent `maxPoints` dts. push() is the
 * frame-loop path (O(1), allocation-free); render() is the explicit paint
 * path called at a lower cadence (rIC) to avoid self-referential jank.
 *
 * Constructor does not touch the canvas so the data path is testable without
 * jsdom / a real HTMLCanvasElement — render() lazily grabs the 2D context.
 */
export class LineChart {
  constructor(
    canvas,
    {
      maxPoints = 300,
      yMax = 50,
      budgetLines = [FRAME_BUDGET_60, FRAME_BUDGET_60 * 2],
      strokeStyle = "#00ffff",
    } = {},
  ) {
    this._canvas = canvas;
    this._maxPoints = maxPoints;
    this._yMax = yMax;
    this._budgetLines = budgetLines;
    this._strokeStyle = strokeStyle;
    this._buf = new Float32Array(maxPoints);
    this._writeIdx = 0;
    this._count = 0;
  }

  push(value) {
    this._buf[this._writeIdx] = value;
    this._writeIdx = (this._writeIdx + 1) % this._maxPoints;
    this._count++;
  }

  clear() {
    this._writeIdx = 0;
    this._count = 0;
    this._buf.fill(0);
  }

  get size() {
    return Math.min(this._count, this._maxPoints);
  }

  /**
   * Chronologically ordered copy of the current window. Allocates a fresh
   * Float32Array each call — intended for tests and for occasional reads
   * (e.g., CSV export), not for the per-frame render loop.
   */
  snapshot() {
    const n = this.size;
    const out = new Float32Array(n);
    if (this._count < this._maxPoints) {
      out.set(this._buf.subarray(0, n));
    } else {
      const tail = this._maxPoints - this._writeIdx;
      out.set(this._buf.subarray(this._writeIdx), 0);
      out.set(this._buf.subarray(0, this._writeIdx), tail);
    }
    return out;
  }

  render() {
    const canvas = this._canvas;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Budget reference lines (yellow = 1× budget, red = 2× budget).
    ctx.lineWidth = 1;
    for (const line of this._budgetLines) {
      const y = h - (line / this._yMax) * h;
      ctx.strokeStyle =
        line >= this._yMax * 0.5
          ? "rgba(255, 80, 80, 0.5)"
          : "rgba(255, 200, 0, 0.5)";
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Data polyline. Snapshot once so we don't re-enter ring-buffer logic
    // mid-render; for 300 points this is a 1.2 KB allocation per paint and
    // render runs at ~1 Hz so the allocation pressure is negligible.
    const values = this.snapshot();
    if (values.length < 2) return;
    ctx.strokeStyle = this._strokeStyle;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const dx = w / (this._maxPoints - 1);
    for (let i = 0; i < values.length; i++) {
      const x = i * dx;
      const y = h - (Math.min(values[i], this._yMax) / this._yMax) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

/**
 * Default layer decomposition for the 353-param Predictor (spec §4 Phase 4,
 * decision 4). Offsets/sizes are derived from constants.js so any future
 * architecture tweak flows through automatically and the Heatmap cannot drift
 * from the actual flat-buffer layout.
 */
export function makeDefaultLayers() {
  return [
    {
      name: "W1",
      rows: MLP_INPUT_DIM,
      cols: MLP_HIDDEN_1,
      offset: W1_OFFSET,
      size: MLP_INPUT_DIM * MLP_HIDDEN_1,
      kind: "heatmap",
    },
    {
      name: "b1",
      rows: 1,
      cols: MLP_HIDDEN_1,
      offset: B1_OFFSET,
      size: MLP_HIDDEN_1,
      kind: "heatmap",
    },
    {
      name: "W2",
      rows: MLP_HIDDEN_1,
      cols: MLP_HIDDEN_2,
      offset: W2_OFFSET,
      size: MLP_HIDDEN_1 * MLP_HIDDEN_2,
      kind: "heatmap",
    },
    {
      name: "b2",
      rows: 1,
      cols: MLP_HIDDEN_2,
      offset: B2_OFFSET,
      size: MLP_HIDDEN_2,
      kind: "heatmap",
    },
    {
      name: "W3",
      rows: MLP_OUTPUT_DIM,
      cols: MLP_HIDDEN_2,
      offset: W3_OFFSET,
      size: MLP_HIDDEN_2 * MLP_OUTPUT_DIM,
      kind: "bar",
    },
    {
      name: "b3",
      rows: 1,
      cols: MLP_OUTPUT_DIM,
      offset: B3_OFFSET,
      size: MLP_OUTPUT_DIM,
      kind: "heatmap",
    },
  ];
}

// Vertical height ratios used by render() — W1 dominates because it has the
// most parameters (192/353 = 54%). Biases are stacked in the bottom 10%.
// Values must sum to 1.
const RENDER_HEIGHT_RATIOS = {
  W1: 0.5,
  W2: 0.3,
  W3: 0.1,
  biases: 0.1,
};

/**
 * Multi-layer diverging-colormap heatmap for the Predictor's 353 parameters.
 * Per-layer independent normalization (abs max within each layer) so small
 * bias layers are not drowned out by the global weight distribution (decision
 * 4, spec §4 Phase 4).
 *
 * Canvas is not touched before render(); all data-path methods (update,
 * clear, getLayerValues, getLayerMaxAbs) work without a live canvas.
 */
export class LayeredHeatmap {
  constructor(canvas, { layers }) {
    this._canvas = canvas;
    this.layers = layers;
    this._values = new Float32Array(PARAM_COUNT);
    this._maxAbs = new Float32Array(layers.length);
  }

  update(params) {
    if (params.length !== PARAM_COUNT) {
      throw new RangeError(
        `LayeredHeatmap.update: expected params.length=${PARAM_COUNT}, got ${params.length}`,
      );
    }
    this._values.set(params);
    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i];
      let m = 0;
      for (let j = 0; j < L.size; j++) {
        const a = Math.abs(params[L.offset + j]);
        if (a > m) m = a;
      }
      this._maxAbs[i] = m;
    }
  }

  clear() {
    this._values.fill(0);
    this._maxAbs.fill(0);
  }

  getLayerValues(idx) {
    const L = this.layers[idx];
    return this._values.subarray(L.offset, L.offset + L.size);
  }

  getLayerMaxAbs(idx) {
    return this._maxAbs[idx];
  }

  render() {
    const canvas = this._canvas;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const hW1 = h * RENDER_HEIGHT_RATIOS.W1;
    const hW2 = h * RENDER_HEIGHT_RATIOS.W2;
    const hW3 = h * RENDER_HEIGHT_RATIOS.W3;
    const hBias = h * RENDER_HEIGHT_RATIOS.biases;

    const yTopW2 = hW1;
    const yTopW3 = hW1 + hW2;
    const yTopBiases = hW1 + hW2 + hW3;

    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i];
      const max = this._maxAbs[i];
      switch (L.name) {
        case "W1":
          this._drawHeatmap(ctx, i, 0, 0, w, hW1);
          break;
        case "W2":
          this._drawHeatmap(ctx, i, 0, yTopW2, w, hW2);
          break;
        case "W3":
          this._drawBar(ctx, i, 0, yTopW3, w, hW3, max);
          break;
        case "b1":
          this._drawHeatmap(ctx, i, 0, yTopBiases, w, hBias / 3);
          break;
        case "b2":
          this._drawHeatmap(ctx, i, 0, yTopBiases + hBias / 3, w, hBias / 3);
          break;
        case "b3":
          this._drawHeatmap(
            ctx,
            i,
            0,
            yTopBiases + (2 * hBias) / 3,
            w,
            hBias / 3,
          );
          break;
      }
    }
  }

  _drawHeatmap(ctx, idx, x, y, w, h) {
    const L = this.layers[idx];
    const max = this._maxAbs[idx];
    const values = this.getLayerValues(idx);
    const cellW = w / L.cols;
    const cellH = h / L.rows;
    for (let r = 0; r < L.rows; r++) {
      for (let c = 0; c < L.cols; c++) {
        const v = values[r * L.cols + c];
        ctx.fillStyle = divergingColor(v, max);
        ctx.fillRect(x + c * cellW, y + r * cellH, cellW + 1, cellH + 1);
      }
    }
  }

  _drawBar(ctx, idx, x, y, w, h, max) {
    const L = this.layers[idx];
    const values = this.getLayerValues(idx);
    const cellW = w / L.size;
    for (let i = 0; i < L.size; i++) {
      ctx.fillStyle = divergingColor(values[i], max);
      ctx.fillRect(x + i * cellW, y, cellW + 1, h);
    }
  }
}
