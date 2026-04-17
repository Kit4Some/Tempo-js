import { describe, it, expect } from "vitest";
import { Predictor } from "../src/core/predictor.js";
import {
  MLP_HIDDEN_1,
  MLP_HIDDEN_2,
  MLP_INPUT_DIM,
  PARAM_COUNT,
  PRED_LOSS_EPS,
} from "../src/core/constants.js";

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
    // Activations are Float64 internally (see predictor.js); we only check
    // length here — the type is an implementation detail that callers should
    // not rely on.
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

describe("Predictor.loss — BCE with symmetric clamp", () => {
  it("at p_miss = 0.5 (zero params), loss = -log(0.5) regardless of target", () => {
    const p = new Predictor();
    p.params.fill(0);
    const x = makeInput(1);
    const expected = -Math.log(0.5);
    expect(p.loss(x, 0)).toBeCloseTo(expected, 10);
    expect(p.loss(x, 1)).toBeCloseTo(expected, 10);
  });

  it("returns a finite number at extreme saturation (clamp prevents -Infinity)", () => {
    const p = new Predictor();
    p.params.fill(0);
    // Force saturation: set b3 extremely negative → p_miss ≈ 0.
    p.params[352] = -1000;
    const x = makeInput(0);
    // target=1 against p≈0 would be -log(0) = Infinity without clamp.
    const l = p.loss(x, 1);
    expect(Number.isFinite(l)).toBe(true);
    // Clamped: -log(EPS)
    expect(l).toBeCloseTo(-Math.log(PRED_LOSS_EPS), 3);
  });

  it("loss is lower when target matches than when it is wrong", () => {
    const p = new Predictor();
    // Push sigmoid toward 1.
    p.params.fill(0);
    p.params[352] = 2; // b3 = 2 → p_miss ≈ 0.88
    const x = makeInput(0);
    const lossRight = p.loss(x, 1);
    const lossWrong = p.loss(x, 0);
    expect(lossRight).toBeLessThan(lossWrong);
  });
});

describe("Predictor.backward — shape & sanity", () => {
  it("returns a Float32Array of length PARAM_COUNT", () => {
    const p = new Predictor();
    const g = p.backward(makeInput(0.1), 1);
    expect(g).toBeInstanceOf(Float32Array);
    expect(g.length).toBe(PARAM_COUNT);
  });

  it("reuses the same grad buffer across calls (zero-alloc)", () => {
    const p = new Predictor();
    const g1 = p.backward(makeInput(0.1), 1);
    const g2 = p.backward(makeInput(0.2), 0);
    expect(g2).toBe(g1);
  });

  it("zero params + zero input + target=0 → grads all zero except b3 = 0.5", () => {
    // With params = 0 and x = 0: every activation is 0, p_miss = 0.5.
    // dL/dz3 = p_clamped - target = 0.5 - 0 = 0.5. All other chain products
    // zero out because a1 = a2 = W* = 0. Only b3's gradient survives.
    const p = new Predictor();
    p.params.fill(0);
    const g = p.backward(makeInput(0), 0);
    for (let i = 0; i < PARAM_COUNT; i++) {
      if (i === 352) {
        expect(g[i]).toBeCloseTo(0.5, 8);
      } else {
        expect(g[i]).toBeCloseTo(0, 8);
      }
    }
  });

  it("zero params + zero input + target=1 → b3 grad = -0.5", () => {
    const p = new Predictor();
    p.params.fill(0);
    const g = p.backward(makeInput(0), 1);
    expect(g[352]).toBeCloseTo(-0.5, 8);
  });
});

describe("Predictor — numerical gradient check", () => {
  function randomInput() {
    const x = new Float32Array(MLP_INPUT_DIM);
    // Small scale keeps p_miss away from sigmoid saturation with He-init.
    for (let i = 0; i < x.length; i++) x[i] = (Math.random() - 0.5) * 0.5;
    return x;
  }

  function makeGradcheckFixture(predictor) {
    // Reject samples that land near the ReLU boundary (z ≈ 0) or sigmoid
    // saturation (p_miss at the extremes). A small eps-perturbation at a
    // ReLU-boundary pre-activation would flip the mask between f(θ+eps) and
    // f(θ-eps), producing spurious numeric gradients.
    for (let tries = 0; tries < 200; tries++) {
      const x = randomInput();
      const { p_miss, hidden } = predictor.forward(x);
      let zMin = Infinity;
      for (const z of hidden.z1) {
        const abz = Math.abs(z);
        if (abz < zMin) zMin = abz;
      }
      for (const z of hidden.z2) {
        const abz = Math.abs(z);
        if (abz < zMin) zMin = abz;
      }
      const zSafe = zMin > 0.01;
      const pSafe = p_miss > 0.05 && p_miss < 0.95;
      if (zSafe && pSafe) return x;
    }
    throw new Error("Could not sample a gradcheck-safe fixture in 200 tries");
  }

  it(
    "analytic gradient matches numerical across 3 fixtures × 353 params (rel < 1e-4)",
    () => {
      const p = new Predictor();
      const eps = 1e-4;
      const tolerance = 1e-4;

      for (let fixId = 0; fixId < 3; fixId++) {
        const x = makeGradcheckFixture(p);
        const target = Math.random() < 0.5 ? 0 : 1;
        // Snapshot analytic gradients (backward returns a buffer reused
        // across calls; copy so subsequent loss() calls can't touch it).
        const analytic = new Float32Array(p.backward(x, target));

        for (let i = 0; i < PARAM_COUNT; i++) {
          const orig = p.params[i];
          p.params[i] = orig + eps;
          const lossPlus = p.loss(x, target);
          p.params[i] = orig - eps;
          const lossMinus = p.loss(x, target);
          p.params[i] = orig;

          const numeric = (lossPlus - lossMinus) / (2 * eps);
          const a = analytic[i];
          const relErr = Math.abs(numeric - a) /
            (Math.abs(numeric) + Math.abs(a) + 1e-8);

          if (relErr > tolerance) {
            throw new Error(
              `Gradient mismatch at fixture ${fixId}, param ${i}: ` +
                `numeric=${numeric}, analytic=${a}, relErr=${relErr}`,
            );
          }
        }
      }

      // Sanity sizes (mostly to remind future readers of the shapes).
      expect(MLP_HIDDEN_1).toBe(16);
      expect(MLP_HIDDEN_2).toBe(8);
    },
    30000,
  );
});
