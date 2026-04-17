// Single source of truth for tunable numeric constants.
// Derived values (e.g., jank = dt > FRAME_BUDGET_60) live in the code that
// uses them — duplicating them here would risk drift.

// === Frame budget ==========================================================
// 60Hz frame budget in ms. Spec §2.1 pins this at 16.67 (not 1000/60) to keep
// benchmark thresholds and blog-post numbers human-readable.
export const FRAME_BUDGET_60 = 16.67;

// === Metrics ===============================================================
// FrameMetrics ring buffer capacity. Bounds memory at 4 KB per instance
// (Float32Array * 1024) while still covering >17s of 60fps history.
export const METRICS_BUFFER_SIZE = 1024;

// Percentile quantiles reported by FrameMetrics.getStats().
export const PERCENTILE_P95 = 0.95;
export const PERCENTILE_P99 = 0.99;

// === Workloads =============================================================
// Default parameters for synthetic frame-load generators (spec §4 Phase 1).
// Callers may override any of these; Phase 5 ablations sweep across them.
export const WL_CONSTANT_MS = 5;

export const WL_SAWTOOTH_MIN = 0;
export const WL_SAWTOOTH_MAX = 20;
export const WL_SAWTOOTH_PERIOD = 60;

export const WL_BURST_BASE_MS = 3;
export const WL_BURST_SPIKE_MS = 30;
export const WL_BURST_SPIKE_EVERY = 90;
export const WL_BURST_SPIKE_DURATION = 5;

export const WL_SCROLL_K = 0.3;

// === Schedulers ============================================================
// Two scheduler families share this section. Naming convention:
//   - *_RATIO      → normalized dt (dimensionless: dt / FRAME_BUDGET_60).
//                    Used by B1 which thresholds on EMA of normalized frame
//                    time. Value ≈ 1 means "frame took one budget".
//   - *_THRESHOLD  → probability of miss (∈ [0, 1]). Used by
//                    PredictorScheduler which thresholds on the Predictor's
//                    sigmoid output.
// These are semantically different — do not unify the suffix.

// B1_EmaThreshold — EMA-threshold heuristic (spec §4 Phase 2).
// features[0] = dt_ema_fast / FRAME_BUDGET_60 (normalized, spec §2.2).
// Decision rule:
//   features[0] > B1_DEGRADE_RATIO → 'degrade'
//   features[0] > B1_REDUCE_RATIO  → 'reduce'
//   else                           → 'full'
export const B1_EMA_ALPHA = 0.3;
export const B1_REDUCE_RATIO = 0.8;
export const B1_DEGRADE_RATIO = 1.2;

// PredictorScheduler — probability thresholds on sigmoid(Predictor.forward).
// Decision rule (spec §2.4):
//   p_miss > PRED_DEGRADE_THRESHOLD → 'degrade'
//   p_miss > PRED_REDUCE_THRESHOLD  → 'reduce'
//   else                            → 'full'
// Wired up in Phase 3 Part 4 (PredictorScheduler adapter).
export const PRED_REDUCE_THRESHOLD = 0.1;
export const PRED_DEGRADE_THRESHOLD = 0.3;

// === Predictor (MLP) =======================================================
// Architecture, learning, and training-buffer hyperparameters.
// Kept separate from scheduler thresholds: policy-level knobs (above) and
// model-level knobs (here) are ablated independently in Phase 5 §6.5.
//
// Populated during Phase 3 Parts 1–3:
//   Part 1 (forward): MLP_HIDDEN_1, MLP_HIDDEN_2, PARAM_COUNT
//   Part 3 (trainer): LR, MOMENTUM, GRAD_CLIP, TRAIN_BUFFER_SIZE, BATCH_SIZE
