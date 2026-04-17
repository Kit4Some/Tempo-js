import { describe, it, expect } from "vitest";
import { Predictor } from "../src/core/predictor.js";
import { MLP_INPUT_DIM, PARAM_COUNT } from "../src/core/constants.js";

function makeInput(fill = 0) {
  const x = new Float32Array(MLP_INPUT_DIM);
  if (typeof fill === "function") {
    for (let i = 0; i < x.length; i++) x[i] = fill(i);
  } else {
    x.fill(fill);
  }
  return x;
}

describe("Predictor — parameter layout", () => {
  it("params is a Float32Array of length PARAM_COUNT (353)", () => {
    const p = new Predictor();
    expect(p.params).toBeInstanceOf(Float32Array);
    expect(p.params.length).toBe(PARAM_COUNT);
    expect(PARAM_COUNT).toBe(353);
  });
});

describe("Predictor.forward", () => {
  it("returns { p_miss, hidden } with p_miss in [0, 1]", () => {
    const p = new Predictor();
    const x = makeInput((i) => Math.sin(i) * 0.5);
    const out = p.forward(x);
    expect(typeof out.p_miss).toBe("number");
    expect(out.p_miss).toBeGreaterThanOrEqual(0);
    expect(out.p_miss).toBeLessThanOrEqual(1);
    expect(out.hidden).toBeDefined();
  });

  it("hidden exposes z1, a1, z2, a2 with correct shapes", () => {
    const p = new Predictor();
    const out = p.forward(makeInput(0));
    expect(out.hidden.z1).toBeInstanceOf(Float32Array);
    expect(out.hidden.z1.length).toBe(16);
    expect(out.hidden.a1.length).toBe(16);
    expect(out.hidden.z2.length).toBe(8);
    expect(out.hidden.a2.length).toBe(8);
  });

  it("is deterministic: two calls with the same input give the same p_miss", () => {
    const p = new Predictor();
    const x = makeInput((i) => Math.cos(i * 0.37));
    const a = p.forward(x).p_miss;
    const b = p.forward(x).p_miss;
    expect(a).toBe(b);
  });

  it("with all-zero params returns p_miss = sigmoid(0) = 0.5", () => {
    const p = new Predictor();
    p.params.fill(0);
    const out = p.forward(makeInput(1));
    expect(out.p_miss).toBeCloseTo(0.5, 10);
  });

  it("matches a hand-computed forward for a sparse weight configuration", () => {
    // x = [1, 0, 0, ..., 0]
    // W1[0,0] = 1, rest = 0   →  z1 = [1, 0, ..., 0]
    // b1 = 0                  →  a1 = relu(z1) = [1, 0, ..., 0]
    // W2[0,0] = 1, rest = 0   →  z2 = [1, 0, ..., 0]
    // b2 = 0                  →  a2 = relu(z2) = [1, 0, ..., 0]
    // W3[0] = 1, rest = 0     →  z3 = 1
    // b3 = 0                  →  p_miss = sigmoid(1) = 1/(1+e^-1)
    const p = new Predictor();
    p.params.fill(0);
    // Offsets: W1[0..191], b1[192..207], W2[208..335], b2[336..343], W3[344..351], b3[352]
    p.params[0] = 1; // W1[row 0, col 0] — row-major, row_stride=12
    p.params[208] = 1; // W2[row 0, col 0] — row-major, row_stride=16
    p.params[344] = 1; // W3[0, 0]
    const x = makeInput(0);
    x[0] = 1;
    const expected = 1 / (1 + Math.exp(-1));
    expect(p.forward(x).p_miss).toBeCloseTo(expected, 6);
  });

  it("throws when input length is not MLP_INPUT_DIM", () => {
    const p = new Predictor();
    expect(() => p.forward(new Float32Array(11))).toThrow();
    expect(() => p.forward(new Float32Array(13))).toThrow();
    expect(() => p.forward(new Float32Array(0))).toThrow();
  });

  it("does not allocate a new Float32Array on each forward call", () => {
    // Pre-allocated buffer requirement: hidden.z1 etc. must be the SAME
    // Float32Array instance across calls (buffer reuse).
    const p = new Predictor();
    const out1 = p.forward(makeInput(0.1));
    const out2 = p.forward(makeInput(0.2));
    expect(out2.hidden.z1).toBe(out1.hidden.z1);
    expect(out2.hidden.a1).toBe(out1.hidden.a1);
    expect(out2.hidden.z2).toBe(out1.hidden.z2);
    expect(out2.hidden.a2).toBe(out1.hidden.a2);
  });
});

describe("Predictor — He initialization", () => {
  it("biases are zeroed after construction", () => {
    const p = new Predictor();
    // b1: 192..207, b2: 336..343, b3: 352
    for (let i = 192; i < 208; i++) expect(p.params[i]).toBe(0);
    for (let i = 336; i < 344; i++) expect(p.params[i]).toBe(0);
    expect(p.params[352]).toBe(0);
  });

  it("weights are non-zero (He init draws from a Gaussian)", () => {
    const p = new Predictor();
    // W1: 0..191 — at least some entries should be non-zero.
    let nonZero = 0;
    for (let i = 0; i < 192; i++) if (p.params[i] !== 0) nonZero++;
    expect(nonZero).toBeGreaterThan(150); // overwhelmingly non-zero
  });
});
