// Deterministic PRNG for tests (Mulberry32).
// Use a fixed seed so convergence / gradcheck / fixture generation results
// are reproducible across machines and CI runs. A flake here is a real bug,
// not bad luck.

/**
 * @param {number} seed — any integer; different seeds produce independent
 *   sequences.
 * @returns {() => number} RNG returning a uniform [0, 1) float per call.
 */
export function mulberry32(seed) {
  return function next() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
