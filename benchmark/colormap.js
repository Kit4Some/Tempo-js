/**
 * Diverging colormap mapping a signed value to an rgb() string for the
 * LayeredHeatmap (spec §4 Phase 4). Value 0 is white, +maxAbs is pure red,
 * -maxAbs is pure blue; values outside [-maxAbs, +maxAbs] saturate.
 *
 * Kept as a pure string-returning function (no Canvas dependency) so the
 * color logic is fully unit-testable — rendering itself is verified manually
 * on the live page.
 *
 * Degenerate case `maxAbs === 0` (e.g., a fresh all-zero bias layer) returns
 * white rather than throwing — a blank heatmap is the honest visualization,
 * and the heatmap's update loop shouldn't need to branch on "did this layer
 * see any non-zero value yet".
 *
 * @param {number} value
 * @param {number} maxAbs  non-negative; 0 is treated as "no variance → white"
 * @returns {string} rgb() CSS color
 */
export function divergingColor(value, maxAbs) {
  if (maxAbs === 0) return "rgb(255, 255, 255)";
  const t = Math.max(-1, Math.min(1, value / maxAbs));
  if (t >= 0) {
    const v = Math.round(255 * (1 - t));
    return `rgb(255, ${v}, ${v})`;
  }
  const v = Math.round(255 * (1 + t));
  return `rgb(${v}, ${v}, 255)`;
}
