import { describe, it, expect } from "vitest";
import { divergingColor } from "../benchmark/colormap.js";

describe("divergingColor", () => {
  it("returns pure white at value=0", () => {
    expect(divergingColor(0, 1)).toBe("rgb(255, 255, 255)");
  });

  it("returns pure red at value=+maxAbs", () => {
    expect(divergingColor(1, 1)).toBe("rgb(255, 0, 0)");
  });

  it("returns pure blue at value=-maxAbs", () => {
    expect(divergingColor(-1, 1)).toBe("rgb(0, 0, 255)");
  });

  it("saturates at values beyond +maxAbs to pure red", () => {
    expect(divergingColor(2, 1)).toBe("rgb(255, 0, 0)");
    expect(divergingColor(100, 1)).toBe("rgb(255, 0, 0)");
  });

  it("saturates at values beyond -maxAbs to pure blue", () => {
    expect(divergingColor(-2, 1)).toBe("rgb(0, 0, 255)");
    expect(divergingColor(-100, 1)).toBe("rgb(0, 0, 255)");
  });

  it("interpolates midway positive to mid-pink", () => {
    // t = 0.5 → v = round(255 * (1 - 0.5)) = 128 → rgb(255, 128, 128)
    expect(divergingColor(0.5, 1)).toBe("rgb(255, 128, 128)");
  });

  it("interpolates midway negative to mid-light-blue", () => {
    // t = -0.5 → v = round(255 * (1 + (-0.5))) = 128 → rgb(128, 128, 255)
    expect(divergingColor(-0.5, 1)).toBe("rgb(128, 128, 255)");
  });

  it("scales with maxAbs (value/maxAbs ratio preserved)", () => {
    // Both should be halfway between white and red.
    expect(divergingColor(0.5, 1)).toBe(divergingColor(5, 10));
    expect(divergingColor(-0.5, 1)).toBe(divergingColor(-50, 100));
  });

  it("returns white when maxAbs=0 (degenerate layer, no variance)", () => {
    // Ergonomic fallback: a freshly-initialized layer with all zeros should
    // render as white rather than throw — the heatmap should come up blank.
    expect(divergingColor(0, 0)).toBe("rgb(255, 255, 255)");
    expect(divergingColor(0.5, 0)).toBe("rgb(255, 255, 255)");
    expect(divergingColor(-0.5, 0)).toBe("rgb(255, 255, 255)");
  });

  it("result is always a valid rgb() string", () => {
    const cases = [
      [0, 1],
      [0.3, 1],
      [-0.3, 1],
      [0.75, 1],
      [1, 1],
      [-1, 1],
      [2.5, 1],
      [0, 0],
    ];
    for (const [v, m] of cases) {
      const s = divergingColor(v, m);
      expect(s).toMatch(/^rgb\((\d{1,3}), (\d{1,3}), (\d{1,3})\)$/);
      const match = s.match(/^rgb\((\d{1,3}), (\d{1,3}), (\d{1,3})\)$/);
      for (let i = 1; i <= 3; i++) {
        const ch = Number(match[i]);
        expect(ch).toBeGreaterThanOrEqual(0);
        expect(ch).toBeLessThanOrEqual(255);
      }
    }
  });
});
