import { describe, it, expect } from "vitest";
import { LineChart, LayeredHeatmap, makeDefaultLayers } from "../benchmark/charts.js";
import {
  B1_OFFSET,
  B2_OFFSET,
  B3_OFFSET,
  MLP_HIDDEN_1,
  MLP_HIDDEN_2,
  MLP_INPUT_DIM,
  MLP_OUTPUT_DIM,
  PARAM_COUNT,
  W1_OFFSET,
  W2_OFFSET,
  W3_OFFSET,
} from "../src/core/constants.js";

// All tests pass `null` for the canvas: constructors must not touch the
// canvas (getContext is lazy, inside render()), so data-path tests work in
// plain Node without jsdom. Rendering is verified manually in commit 5.

describe("LineChart", () => {
  it("starts empty: size=0, snapshot is Float32Array(0)", () => {
    const c = new LineChart(null, { maxPoints: 5 });
    expect(c.size).toBe(0);
    const snap = c.snapshot();
    expect(snap).toBeInstanceOf(Float32Array);
    expect(snap.length).toBe(0);
  });

  it("push preserves insertion order while count <= maxPoints", () => {
    const c = new LineChart(null, { maxPoints: 5 });
    c.push(1);
    c.push(2);
    c.push(3);
    expect(c.size).toBe(3);
    expect(Array.from(c.snapshot())).toEqual([1, 2, 3]);
  });

  it("push exactly maxPoints times fills the buffer without dropping", () => {
    const c = new LineChart(null, { maxPoints: 5 });
    for (let i = 1; i <= 5; i++) c.push(i);
    expect(c.size).toBe(5);
    expect(Array.from(c.snapshot())).toEqual([1, 2, 3, 4, 5]);
  });

  it("drops the oldest value when pushing past maxPoints (ring buffer)", () => {
    const c = new LineChart(null, { maxPoints: 5 });
    for (let i = 1; i <= 6; i++) c.push(i);
    expect(c.size).toBe(5);
    expect(Array.from(c.snapshot())).toEqual([2, 3, 4, 5, 6]);
  });

  it("wraps correctly across many pushes", () => {
    const c = new LineChart(null, { maxPoints: 3 });
    for (let i = 1; i <= 10; i++) c.push(i);
    expect(c.size).toBe(3);
    expect(Array.from(c.snapshot())).toEqual([8, 9, 10]);
  });

  it("clear resets size to 0 and snapshot to empty", () => {
    const c = new LineChart(null, { maxPoints: 5 });
    c.push(1);
    c.push(2);
    c.push(3);
    c.clear();
    expect(c.size).toBe(0);
    expect(c.snapshot().length).toBe(0);
  });

  it("is reusable after clear", () => {
    const c = new LineChart(null, { maxPoints: 5 });
    c.push(1);
    c.push(2);
    c.clear();
    c.push(99);
    expect(c.size).toBe(1);
    expect(Array.from(c.snapshot())).toEqual([99]);
  });

  it("defaults maxPoints=300 when not specified", () => {
    const c = new LineChart(null);
    for (let i = 0; i < 350; i++) c.push(i);
    expect(c.size).toBe(300);
    // Oldest kept is i=50, newest is i=349.
    const snap = c.snapshot();
    expect(snap[0]).toBe(50);
    expect(snap[299]).toBe(349);
  });

  it("push does not auto-render (data path is canvas-free)", () => {
    // Passing null canvas and never calling render() must not throw — this
    // is the whole reason push/render are split.
    const c = new LineChart(null, { maxPoints: 5 });
    expect(() => {
      for (let i = 0; i < 20; i++) c.push(i);
      c.clear();
      c.push(1);
    }).not.toThrow();
  });
});

describe("makeDefaultLayers", () => {
  it("returns 6 layers covering exactly PARAM_COUNT without gaps or overlap", () => {
    const layers = makeDefaultLayers();
    expect(layers).toHaveLength(6);
    const total = layers.reduce((s, L) => s + L.size, 0);
    expect(total).toBe(PARAM_COUNT);

    // Sort by offset and verify adjacency covers [0, PARAM_COUNT).
    const sorted = [...layers].sort((a, b) => a.offset - b.offset);
    let cursor = 0;
    for (const L of sorted) {
      expect(L.offset).toBe(cursor);
      cursor += L.size;
    }
    expect(cursor).toBe(PARAM_COUNT);
  });

  it("matches the Predictor layout (constants.js is single source of truth)", () => {
    const layers = makeDefaultLayers();
    const byName = Object.fromEntries(layers.map((L) => [L.name, L]));
    expect(byName.W1).toMatchObject({
      offset: W1_OFFSET,
      size: MLP_INPUT_DIM * MLP_HIDDEN_1,
    });
    expect(byName.b1).toMatchObject({ offset: B1_OFFSET, size: MLP_HIDDEN_1 });
    expect(byName.W2).toMatchObject({
      offset: W2_OFFSET,
      size: MLP_HIDDEN_1 * MLP_HIDDEN_2,
    });
    expect(byName.b2).toMatchObject({ offset: B2_OFFSET, size: MLP_HIDDEN_2 });
    expect(byName.W3).toMatchObject({
      offset: W3_OFFSET,
      size: MLP_HIDDEN_2 * MLP_OUTPUT_DIM,
    });
    expect(byName.b3).toMatchObject({ offset: B3_OFFSET, size: MLP_OUTPUT_DIM });
  });
});

describe("LayeredHeatmap", () => {
  it("exposes the configured layers and stores them unchanged", () => {
    const layers = makeDefaultLayers();
    const h = new LayeredHeatmap(null, { layers });
    expect(h.layers).toHaveLength(6);
    expect(h.layers[0].name).toBe("W1");
  });

  it("update(params) stores the full params buffer internally", () => {
    const h = new LayeredHeatmap(null, { layers: makeDefaultLayers() });
    const params = new Float32Array(PARAM_COUNT);
    for (let i = 0; i < PARAM_COUNT; i++) params[i] = i * 0.01;
    h.update(params);
    // Any layer's values must match the corresponding slice of params.
    const w1 = h.getLayerValues(0);
    expect(w1.length).toBe(MLP_INPUT_DIM * MLP_HIDDEN_1);
    for (let i = 0; i < w1.length; i++) {
      expect(w1[i]).toBeCloseTo(params[W1_OFFSET + i], 6);
    }
  });

  it("computes per-layer maxAbs (independent normalization)", () => {
    const h = new LayeredHeatmap(null, { layers: makeDefaultLayers() });
    const params = new Float32Array(PARAM_COUNT);
    // Seed W1 range with max-abs 0.5.
    for (let i = 0; i < MLP_INPUT_DIM * MLP_HIDDEN_1; i++) {
      params[W1_OFFSET + i] = (i % 2 === 0 ? 1 : -1) * 0.5;
    }
    // Seed b1 range with max-abs 10.
    for (let i = 0; i < MLP_HIDDEN_1; i++) {
      params[B1_OFFSET + i] = i === 0 ? -10 : 0;
    }
    h.update(params);
    // Layer indices follow makeDefaultLayers() order.
    const idxW1 = h.layers.findIndex((L) => L.name === "W1");
    const idxB1 = h.layers.findIndex((L) => L.name === "b1");
    expect(h.getLayerMaxAbs(idxW1)).toBeCloseTo(0.5, 6);
    expect(h.getLayerMaxAbs(idxB1)).toBeCloseTo(10, 6);
  });

  it("maxAbs stays 0 for a layer whose params are all zero", () => {
    const h = new LayeredHeatmap(null, { layers: makeDefaultLayers() });
    const params = new Float32Array(PARAM_COUNT); // all zeros
    h.update(params);
    for (let i = 0; i < h.layers.length; i++) {
      expect(h.getLayerMaxAbs(i)).toBe(0);
    }
  });

  it("clear() resets values and maxAbs", () => {
    const h = new LayeredHeatmap(null, { layers: makeDefaultLayers() });
    const params = new Float32Array(PARAM_COUNT);
    params[0] = 5;
    params[B1_OFFSET] = -3;
    h.update(params);
    h.clear();
    for (let i = 0; i < h.layers.length; i++) {
      expect(h.getLayerMaxAbs(i)).toBe(0);
    }
    // Values also zeroed.
    const w1 = h.getLayerValues(0);
    for (let i = 0; i < w1.length; i++) expect(w1[i]).toBe(0);
  });

  it("update throws if params length does not match PARAM_COUNT", () => {
    const h = new LayeredHeatmap(null, { layers: makeDefaultLayers() });
    expect(() => h.update(new Float32Array(10))).toThrow();
    expect(() => h.update(new Float32Array(PARAM_COUNT + 1))).toThrow();
  });

  it("data path never touches canvas (render is lazy)", () => {
    const h = new LayeredHeatmap(null, { layers: makeDefaultLayers() });
    const params = new Float32Array(PARAM_COUNT);
    expect(() => {
      h.update(params);
      h.clear();
      h.getLayerValues(0);
      h.getLayerMaxAbs(0);
    }).not.toThrow();
  });
});
