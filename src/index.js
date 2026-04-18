// Library entry re-exports. Spec §3 lists src/index.js as the project's
// nominal module entry; we keep it as pure re-exports so test suites and
// any future consumer can `import { Predictor } from '../src/index.js'`
// without reaching into `core/` internals.
//
// No DOM / no side effects — the live benchmark page drives its own setup
// through benchmark/app.js, not through this module.

export { Predictor } from "./core/predictor.js";
export { OnlineTrainer } from "./core/trainer.js";
export { FeatureExtractor } from "./core/features.js";
export {
  B0_AlwaysFull,
  B1_EmaThreshold,
  PredictorScheduler,
} from "./core/schedulers.js";
export { FrameMetrics, RollingFrameMetrics } from "./harness/metrics.js";
export { SequentialLoop } from "./harness/sequential-loop.js";
export { workCostFor } from "./harness/work-cost.js";
export {
  burst,
  constant,
  sawtooth,
  scrollCorrelated,
} from "./harness/workloads.js";
