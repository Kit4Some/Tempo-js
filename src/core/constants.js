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

// === Baselines =============================================================
// B1 EMA-threshold scheduler (spec §4 Phase 2).
// features[0] = dt_ema_fast / FRAME_BUDGET_60 (normalized, spec §2.2).
// Decision rule:
//   features[0] > B1_DEGRADE_RATIO → 'degrade'
//   features[0] > B1_REDUCE_RATIO  → 'reduce'
//   else                           → 'full'
export const B1_EMA_ALPHA = 0.3;
export const B1_REDUCE_RATIO = 0.8;
export const B1_DEGRADE_RATIO = 1.2;
