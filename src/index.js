import { FrameMetrics } from "./harness/metrics.js";
import {
  burst,
  constant,
  sawtooth,
  scrollCorrelated,
} from "./harness/workloads.js";

export { FrameMetrics, burst, constant, sawtooth, scrollCorrelated };

if (typeof window !== "undefined" && typeof document !== "undefined") {
  const metrics = new FrameMetrics();
  const workload = sawtooth(0, 15, 60);
  metrics.onFrame((idx, _dt) => workload(idx));
  metrics.start();
  setInterval(() => {
    // eslint-disable-next-line no-console
    console.log("[tempo]", metrics.getStats());
  }, 1000);
}
