// Single source of truth for tunable numeric constants.
// Derived values (e.g., jank = dt > FRAME_BUDGET_60) live in the code that
// uses them — duplicating them here would risk drift.

// === Frame budget ==========================================================
// 60Hz frame budget in ms. Spec §2.1 pins this at 16.67 (not 1000/60) to keep
// benchmark thresholds and blog-post numbers human-readable.
export const FRAME_BUDGET_60 = 16.67;

// Jank detection tolerance. rAF callbacks are vsync-aligned and fire at
// multiples of ~16.67 ms on a 60 Hz display, but the actual delta has ~0.1–0.3
// ms of float jitter from scheduler/OS variance — a strict `dt > 16.67` check
// falsely classifies that noise as jank. Phase 4 live-page verification
// measured ~57% "jank" on a constant 5ms workload (actual work 1.75ms) due
// to this jitter; the Predictor's Phase 5 Go/No-Go then depends on a 15%
// difference that would otherwise be swamped by the measurement noise.
// 1.0 ms margin swallows the jitter without hiding real misses (anything
// >17.67 ms is a genuine missed frame on 60 Hz). Applied at every jank
// comparison: FrameMetrics, RollingFrameMetrics, SequentialLoop.step's
// wasMiss, and FeatureExtractor's miss_rate_32.
//
// Calibration history: 0.5 ms was tried first — live page constant-5ms
// workload reported ~14% jank (down from 57% pre-tolerance) because P95
// consistently sat at 18 ms from Canvas paint + heatmap update overhead
// that rIC only mostly isolates. Raised to 1.0 ms per user calibration
// spec; 2.0 ms would start hiding real jank and demands a different root
// cause (not vsync jitter).
export const JANK_TOLERANCE_MS = 1.0;

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

// Phase 4 UI — work-cost scale applied to the workload's base frame cost when
// a Scheduler's decision is executed on the live benchmark page (Sequential
// mode, one active scheduler at a time). Ratios approximate a real UI
// framework's decorative/essential split (Framer Motion et al.):
//   full    — 100% of the workload's base cost.
//   reduce  — skip decorative frames (~30%), keep essential work.
//   degrade — CSS-transition fallback; compositing + parse overhead remain.
// These are NOT physical measurements — sensitivity to these numbers must be
// documented in blog post §6.5 (limitations).
export const COST_FULL = 1.0;
export const COST_REDUCE = 0.7;
export const COST_DEGRADE = 0.35;

// Run-all-3 pacing. Live and headless protocols deliberately diverge:
//   - Live (Phase 4): 30 s per scheduler + 10 s cooldown = 110 s total.
//     Total ≤ ~2 min so a blog visitor won't assume the tab is frozen.
//   - Headless (Phase 5 §5): 60 s per scheduler + 10 s cooldown = 200 s.
//     Longer runs give Mann-Whitney U the sample size it needs; a Puppeteer
//     worker has no UX budget to worry about.
// README should call out the divergence: "Live demo uses shorter runs (30 s)
// for UX; headless benchmark uses 60 s per §5 protocol."
export const LIVE_RUN_MS = 30_000;
export const LIVE_COOLDOWN_MS = 10_000;
export const HEADLESS_RUN_MS = 60_000;
export const HEADLESS_COOLDOWN_MS = 10_000;

// === Predictor (MLP) =======================================================
// Architecture, learning, and training-buffer hyperparameters.
// Kept separate from scheduler thresholds: policy-level knobs (above) and
// model-level knobs (here) are ablated independently in Phase 5 §6.5.

// Architecture (spec §2.1): x(12) → Linear(12→16) → ReLU → Linear(16→8) →
// ReLU → Linear(8→1) → Sigmoid → p_miss.
export const MLP_INPUT_DIM = 12;
export const MLP_HIDDEN_1 = 16;
export const MLP_HIDDEN_2 = 8;
export const MLP_OUTPUT_DIM = 1;

// Parameter count: W1(16×12) + b1(16) + W2(8×16) + b2(8) + W3(1×8) + b3(1)
//                = 192 + 16 + 128 + 8 + 8 + 1 = 353.
// The spec §2.1 diagram's "321" was an arithmetic mistake (bias-inclusive
// sum is 353); we pin PARAM_COUNT here and align the rest of the codebase.
export const PARAM_COUNT = 353;

// Flat Float32Array parameter layout (row-major). Offsets derived from the
// MLP_*_DIM constants so any architecture change flows through automatically.
// Exposed at module scope (rather than kept local to predictor.js) so Phase 4
// LayeredHeatmap can slice the 353-param vector by layer without duplicating
// the layout math — a single source of truth for "where does W2 live".
//   W1: [0,   192)   size MLP_INPUT_DIM * MLP_HIDDEN_1 = 192
//   b1: [192, 208)   size MLP_HIDDEN_1                 = 16
//   W2: [208, 336)   size MLP_HIDDEN_1 * MLP_HIDDEN_2  = 128
//   b2: [336, 344)   size MLP_HIDDEN_2                 = 8
//   W3: [344, 352)   size MLP_HIDDEN_2 * MLP_OUTPUT_DIM = 8
//   b3: [352, 353)   size MLP_OUTPUT_DIM               = 1
export const W1_OFFSET = 0;
export const B1_OFFSET = W1_OFFSET + MLP_INPUT_DIM * MLP_HIDDEN_1;
export const W2_OFFSET = B1_OFFSET + MLP_HIDDEN_1;
export const B2_OFFSET = W2_OFFSET + MLP_HIDDEN_1 * MLP_HIDDEN_2;
export const W3_OFFSET = B2_OFFSET + MLP_HIDDEN_2;
export const B3_OFFSET = W3_OFFSET + MLP_HIDDEN_2 * MLP_OUTPUT_DIM;

// BCE loss clamp epsilon. p_miss is clamped to [PRED_LOSS_EPS, 1-PRED_LOSS_EPS]
// in both Predictor.loss() and Predictor.backward()'s dL/dz3 computation to
// keep log() finite and to keep the two computations using the same value of
// p (matters for gradcheck consistency).
export const PRED_LOSS_EPS = 1e-7;

// Training hyperparameters (spec §2.3). OnlineTrainer defaults; callers can
// override via constructor options.
// LR was 1e-4 in the original spec; raised to 1e-3 after the convergence
// diagnostic in commit "refactor(phase-3): tune LR to 1e-3". Effective step
// size at steady-state momentum is lr / (1 - mu) × avgGradNorm ≈ 3.5e-3,
// which reaches loss<0.1 well inside the 10k-step budget.
export const LR = 1e-3;
export const MOMENTUM = 0.9;
export const GRAD_CLIP = 1.0; // L2 norm threshold; clip applied to averaged grad
export const TRAIN_BUFFER_SIZE = 1024;
export const BATCH_SIZE = 16;

// === Feature extraction ====================================================
// FeatureExtractor (spec §2.2): produces a 12-dim Float32 vector per frame.

// EMA smoothing for normalized dt (dt / FRAME_BUDGET_60).
export const DT_EMA_FAST_ALPHA = 0.3;
export const DT_EMA_SLOW_ALPHA = 0.05;

// Rolling window sizes over raw dt (ms).
export const DT_WINDOW_SHORT = 8; // used by dt_var and dt_max_8
export const DT_WINDOW_MISS = 32; // used by miss_rate_32; also backs dt_var/max

// Device tier mapping from navigator.hardwareConcurrency (spec §2.2 feature 12).
export const DEVICE_TIER_LOW_MAX = 2; // hw ≤ this → tier 0
export const DEVICE_TIER_MID_MAX = 8; // hw ≤ this → tier 1, else tier 2
export const DEVICE_TIER_DEFAULT = 1; // used when hardwareConcurrency is undefined
