// Sanity check for Phase 3 Part 4: run PredictorScheduler on a sawtooth
// workload for 500 frames and print the decision distribution. Scratch
// weights (no pretrained) mean the model is basically chance at the start,
// so the overwhelming majority should be 'full'.
//
// Usage: node scripts/demo-predictor-scheduler.js

import { PredictorScheduler } from "../src/core/schedulers.js";
import { Predictor } from "../src/core/predictor.js";
import { OnlineTrainer } from "../src/core/trainer.js";
import { FeatureExtractor } from "../src/core/features.js";
import { sawtooth } from "../src/harness/workloads.js";
import { FRAME_BUDGET_60 } from "../src/core/constants.js";

const predictor = new Predictor();
const trainer = new OnlineTrainer(predictor);
const scheduler = new PredictorScheduler({ predictor, trainer });
const features = new FeatureExtractor();
const workload = sawtooth();

const counts = { full: 0, reduce: 0, degrade: 0 };

// Simulate 500 frames with a sawtooth workload. We fake dt from the workload
// amplitude at each frame (phase * max/period) + a small baseline so the
// extractor has something non-trivial to chew on.
const PERIOD = 60;
const MAX_LOAD = 20;
for (let frameIdx = 0; frameIdx < 500; frameIdx++) {
  const phase = (frameIdx % PERIOD) / PERIOD;
  const simulatedDt = 5 + phase * MAX_LOAD; // 5..25 ms
  features.observe({ dt: simulatedDt });
  const x = features.extract();
  const decision = scheduler.decide(x);
  counts[decision]++;
  scheduler.onFrameComplete(simulatedDt, simulatedDt > FRAME_BUDGET_60);
}

// Touch the workload default so eslint doesn't flag the import.
void workload;

process.stdout.write(
  `PredictorScheduler decision distribution over 500 frames (scratch weights):\n` +
    `  full=${counts.full}  reduce=${counts.reduce}  degrade=${counts.degrade}\n` +
    `  trainer buffer=${trainer.bufferCount()}\n`,
);
