// Pure-JS statistics for Phase 5 Go/No-Go analysis (spec §4 Phase 5 Part 1,
// methodology PHASE5_NOTES.md § Statistical methodology). Intentionally
// zero-dependency: the blog-post promise is "353-parameter MLP + pure JS",
// and bringing scipy/R as a dependency would undermine that claim.
//
// Asymptotic approximations are used where scipy would use exact tests for
// small n. For n1 = n2 = 10 (Part 1's reps-per-cell), this shifts two-sided
// p-values by ~50% at the tails (e.g., scipy exact 1e-4 vs our asymptotic
// 2e-4). The Go/No-Go gate (`p < 0.05`) is insensitive to that gap —
// agreement on "significant" or "not" is what matters. The commit that
// surfaces a disagreement between this module and scipy is the commit that
// must revisit exact-distribution support.

// --- Summary stats -------------------------------------------------------

export function mean(xs) {
  if (xs.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

export function variance(xs) {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    const d = xs[i] - m;
    s += d * d;
  }
  return s / (xs.length - 1);
}

export function stdev(xs) {
  return Math.sqrt(variance(xs));
}

// --- Normal CDF (Abramowitz & Stegun 7.1.26) -----------------------------

export function normalCdf(z) {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * x);
  const y =
    1.0 -
    ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1.0 + sign * y);
}

// --- Cohen's d ------------------------------------------------------------

export function cohensD(a, b) {
  const mA = mean(a);
  const mB = mean(b);
  const vA = variance(a);
  const vB = variance(b);
  // Pooled SD with (n-1) weighting (Cohen 1988).
  const nA = a.length;
  const nB = b.length;
  const pooled = Math.sqrt(((nA - 1) * vA + (nB - 1) * vB) / (nA + nB - 2));
  if (pooled === 0) {
    // Degenerate: both samples have zero variance. Sign is still meaningful
    // if means differ; return signed Infinity so callers can distinguish
    // "constant samples, identical means" (0) from "constant samples,
    // separated means" (±Infinity).
    if (mA === mB) return 0;
    return mA < mB ? -Infinity : Infinity;
  }
  return (mA - mB) / pooled;
}

// --- Ranking (Wilcoxon-Mann-Whitney) -------------------------------------

// Returns an array of ranks aligned with the input. Tied values receive
// the average of the ranks they span (standard Wilcoxon convention).
export function rankWithTies(xs) {
  const n = xs.length;
  const indexed = xs.map((v, i) => ({ v, i }));
  indexed.sort((p, q) => p.v - q.v);

  const ranks = new Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && indexed[j + 1].v === indexed[i].v) j++;
    // Group [i..j] are tied; average rank is ((i+1) + (j+1)) / 2.
    const avgRank = (i + j + 2) / 2;
    for (let k = i; k <= j; k++) ranks[indexed[k].i] = avgRank;
    i = j + 1;
  }
  return ranks;
}

// --- Mann-Whitney U (two-sided) ------------------------------------------

// Returns { U, p, n1, n2 } where U is the standard U statistic (the smaller
// of U1/U2 is returned for consistency with common conventions) and p is a
// two-sided p-value via normal approximation with continuity correction and
// tie-corrected variance. For small n (< 20) scipy uses exact distributions;
// we accept the approximation — see module-level docstring.
export function mannWhitneyU(a, b) {
  const n1 = a.length;
  const n2 = b.length;
  if (n1 === 0 || n2 === 0) {
    throw new Error("mannWhitneyU: empty sample");
  }
  const combined = a.concat(b);
  const ranks = rankWithTies(combined);
  let R1 = 0;
  for (let i = 0; i < n1; i++) R1 += ranks[i];
  const U1 = R1 - (n1 * (n1 + 1)) / 2;
  const U2 = n1 * n2 - U1;
  const U = Math.min(U1, U2);

  // Tie correction: sum(t^3 - t) over each tie-group size t.
  const N = n1 + n2;
  const sorted = [...combined].sort((x, y) => x - y);
  let tieCorrection = 0;
  let i = 0;
  while (i < N) {
    let j = i;
    while (j + 1 < N && sorted[j + 1] === sorted[i]) j++;
    const t = j - i + 1;
    if (t > 1) tieCorrection += t * t * t - t;
    i = j + 1;
  }

  const meanU = (n1 * n2) / 2;
  const varU =
    ((n1 * n2) / 12) *
    (N + 1 - tieCorrection / (N * (N - 1)));
  if (varU <= 0) {
    // All values tied → U always equals n1*n2/2; the test has no power.
    return { U, p: 1, n1, n2 };
  }
  // |z| with continuity correction. Using |U - meanU| - 0.5 (not less than 0)
  // because the correction shrinks toward the mean.
  const diff = Math.max(0, Math.abs(U - meanU) - 0.5);
  const z = diff / Math.sqrt(varU);
  const p = 2 * (1 - normalCdf(z));
  return { U, p, n1, n2 };
}

// --- Bootstrap 95% CI of the mean ----------------------------------------

// Non-parametric percentile bootstrap. Resample WITH replacement B times,
// compute the mean of each resample, return the [lo, hi] quantiles that
// bracket `level` (default 0.95) of the distribution.
export function bootstrapMeanCI(sample, {
  B = 1000,
  level = 0.95,
  rng = Math.random,
} = {}) {
  if (sample.length === 0) return [0, 0];
  const n = sample.length;
  const means = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng() * n);
      s += sample[idx];
    }
    means[b] = s / n;
  }
  const sorted = Array.from(means).sort((x, y) => x - y);
  const alpha = (1 - level) / 2;
  const lo = sorted[Math.floor(alpha * B)];
  const hi = sorted[Math.min(B - 1, Math.ceil((1 - alpha) * B) - 1)];
  return [lo, hi];
}
