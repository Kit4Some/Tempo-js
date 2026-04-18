import {
  COST_DEGRADE,
  COST_FULL,
  COST_REDUCE,
} from "../core/constants.js";

/**
 * Map a Scheduler Decision to its multiplicative cost factor (spec §4 Phase 4).
 *
 * Applied on the live benchmark page: the workload's base frame cost is
 * multiplied by this factor before busy-waiting, so the active Scheduler's
 * decision actually influences observed dt. Unknown inputs throw — Decision
 * is a closed literal type ('full' | 'reduce' | 'degrade'), and silent
 * fallbacks would mask scheduler bugs.
 *
 * @param {'full'|'reduce'|'degrade'} decision
 * @returns {number}
 */
export function workCostFor(decision) {
  switch (decision) {
    case "full":
      return COST_FULL;
    case "reduce":
      return COST_REDUCE;
    case "degrade":
      return COST_DEGRADE;
    default:
      throw new TypeError(
        `workCostFor: unknown decision ${JSON.stringify(decision)}`,
      );
  }
}
