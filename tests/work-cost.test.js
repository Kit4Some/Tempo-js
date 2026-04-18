import { describe, it, expect } from "vitest";
import { workCostFor } from "../src/harness/work-cost.js";
import {
  COST_FULL,
  COST_REDUCE,
  COST_DEGRADE,
} from "../src/core/constants.js";

describe("workCostFor", () => {
  it("maps 'full' to COST_FULL", () => {
    expect(workCostFor("full")).toBe(COST_FULL);
  });

  it("maps 'reduce' to COST_REDUCE", () => {
    expect(workCostFor("reduce")).toBe(COST_REDUCE);
  });

  it("maps 'degrade' to COST_DEGRADE", () => {
    expect(workCostFor("degrade")).toBe(COST_DEGRADE);
  });

  it("throws on unknown decision string", () => {
    expect(() => workCostFor("explode")).toThrow();
  });

  it("throws on undefined", () => {
    expect(() => workCostFor(undefined)).toThrow();
  });

  it("throws on null", () => {
    expect(() => workCostFor(null)).toThrow();
  });

  it("constants satisfy the documented ordering full > reduce > degrade > 0", () => {
    expect(COST_FULL).toBeGreaterThan(COST_REDUCE);
    expect(COST_REDUCE).toBeGreaterThan(COST_DEGRADE);
    expect(COST_DEGRADE).toBeGreaterThan(0);
  });
});
