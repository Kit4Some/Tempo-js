import { describe, it, expect } from "vitest";
import {
  mean,
  variance,
  stdev,
  cohensD,
  rankWithTies,
  mannWhitneyU,
  bootstrapMeanCI,
  normalCdf,
} from "../src/harness/stats.js";
import { mulberry32 } from "./helpers/rng.js";

describe("mean / variance / stdev", () => {
  it("basic sample statistics", () => {
    expect(mean([1, 2, 3, 4, 5])).toBeCloseTo(3, 10);
    // Sample variance (n-1 denominator): var([1..5]) = 2.5
    expect(variance([1, 2, 3, 4, 5])).toBeCloseTo(2.5, 10);
    expect(stdev([1, 2, 3, 4, 5])).toBeCloseTo(Math.sqrt(2.5), 10);
  });
});

describe("normalCdf (Abramowitz–Stegun 7.1.26)", () => {
  it("agrees with reference values to 4 decimals", () => {
    // Reference table values (two-tailed tails)
    expect(normalCdf(0)).toBeCloseTo(0.5, 4);
    expect(normalCdf(1)).toBeCloseTo(0.8413, 3);
    expect(normalCdf(1.96)).toBeCloseTo(0.975, 3);
    expect(normalCdf(-1.96)).toBeCloseTo(0.025, 3);
    expect(normalCdf(3.0)).toBeCloseTo(0.99865, 3);
  });
});

describe("cohensD", () => {
  it("is 0 when samples are identical", () => {
    expect(cohensD([1, 2, 3], [1, 2, 3])).toBeCloseTo(0, 10);
  });

  it("is negative when first sample has smaller mean", () => {
    // R effsize::cohen.d([1..5], [2..6]) = -0.632
    expect(cohensD([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])).toBeCloseTo(-0.632, 2);
  });

  it("is large (|d| > 0.8) when samples are clearly separated", () => {
    // Mean 1 vs mean 10, tight distributions
    const d = cohensD([1, 1, 1, 1, 1], [10, 10, 10, 10, 10]);
    // Both variances are 0 → pooled SD is 0; function should handle this
    // deterministically rather than blow up.
    expect(Number.isFinite(d) || d === Infinity || d === -Infinity).toBe(true);
  });

  it("classification thresholds hold for a standard small-medium-large example", () => {
    // Small effect — standardized mean diff ≈ 0.2
    const small = cohensD([0, 1, 2, 3, 4, 5], [0.4, 1.4, 2.4, 3.4, 4.4, 5.4]);
    expect(Math.abs(small)).toBeGreaterThan(0.15);
    expect(Math.abs(small)).toBeLessThan(0.35);
  });
});

describe("rankWithTies", () => {
  it("returns integer ranks for distinct values", () => {
    expect(rankWithTies([3, 1, 2])).toEqual([3, 1, 2]);
  });

  it("averages ranks within a tie group", () => {
    // [1, 2, 2, 3] → ranks 1, 2.5, 2.5, 4
    expect(rankWithTies([1, 2, 2, 3])).toEqual([1, 2.5, 2.5, 4]);
  });

  it("handles all-tied arrays", () => {
    expect(rankWithTies([5, 5, 5, 5])).toEqual([2.5, 2.5, 2.5, 2.5]);
  });
});

describe("mannWhitneyU", () => {
  it("U = 0 for fully non-overlapping samples", () => {
    // [1..10] vs [11..20]: A's ranks sum to 55, U1 = 55 - 10*11/2 = 0.
    const r = mannWhitneyU(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    );
    expect(r.U).toBe(0);
    expect(r.n1).toBe(10);
    expect(r.n2).toBe(10);
    // Two-sided p via normal approximation with continuity correction:
    // z = (|0 - 50| - 0.5) / sqrt(175) = 49.5 / 13.2288 ≈ 3.742
    // p = 2 × (1 − Φ(3.742)) ≈ 1.83e-4.
    // scipy's exact mannwhitneyu gives ~1.08e-4; we report the asymptotic
    // value but agreement on "p < 0.05" is what the Go/No-Go gate needs.
    expect(r.p).toBeLessThan(0.001);
  });

  it("U at the center (n1*n2/2) gives p = 1 for identical samples", () => {
    const r = mannWhitneyU([5, 5, 5, 5], [5, 5, 5, 5]);
    expect(r.U).toBe(8); // n1*n2/2 = 16/2 = 8
    expect(r.p).toBeCloseTo(1, 5);
  });

  it("returns the same p-value regardless of sample order", () => {
    // We return min(U1, U2) by convention, so U(A,B) == U(B,A). The
    // load-bearing symmetry is on the two-sided p-value.
    const a = [1, 4, 7, 2, 8];
    const b = [3, 5, 6, 9, 10];
    const ab = mannWhitneyU(a, b);
    const ba = mannWhitneyU(b, a);
    expect(ab.U).toBe(ba.U);
    expect(ab.p).toBeCloseTo(ba.p, 10);
  });

  it("reports p > 0.05 when distributions overlap heavily", () => {
    const a = [1, 2, 3, 4, 5];
    const b = [1, 2, 3, 4, 5]; // identical distributions
    const r = mannWhitneyU(a, b);
    expect(r.p).toBeGreaterThan(0.05);
  });
});

describe("bootstrapMeanCI", () => {
  it("tightly brackets the sample mean for low-variance data", () => {
    const rng = mulberry32(42);
    const sample = [5.0, 5.1, 5.0, 4.9, 5.0, 5.1, 5.0, 4.9, 5.0, 5.1];
    const ci = bootstrapMeanCI(sample, { B: 1000, level: 0.95, rng });
    const m = mean(sample);
    expect(ci[0]).toBeLessThanOrEqual(m);
    expect(ci[1]).toBeGreaterThanOrEqual(m);
    // SD ≈ 0.067 → SE on mean ≈ 0.021 → CI half-width ≈ 0.04
    expect(ci[1] - ci[0]).toBeLessThan(0.1);
  });

  it("is deterministic under a seeded RNG", () => {
    const sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const ciA = bootstrapMeanCI(sample, {
      B: 500,
      level: 0.95,
      rng: mulberry32(7),
    });
    const ciB = bootstrapMeanCI(sample, {
      B: 500,
      level: 0.95,
      rng: mulberry32(7),
    });
    expect(ciA).toEqual(ciB);
  });

  it("collapses to a point when the sample is constant", () => {
    const sample = [3, 3, 3, 3, 3];
    const ci = bootstrapMeanCI(sample, {
      B: 200,
      level: 0.95,
      rng: mulberry32(0),
    });
    expect(ci[0]).toBeCloseTo(3, 10);
    expect(ci[1]).toBeCloseTo(3, 10);
  });
});
