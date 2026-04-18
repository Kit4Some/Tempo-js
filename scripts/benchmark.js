// Phase 5 Part 1 — Sequential + shadow headless benchmark runner.
//
// Spec §4 Phase 5 Part 1. 4 workloads × 3 schedulers × N reps, run order
// shuffled with a seeded Fisher–Yates, 60 s per run, 10 s cooldown between.
// Puppeteer-headless via vite preview. Per-run summary appended to
// docs/PHASE5_PART1_RESULTS.jsonl (committed). Per-frame shadow log
// streamed to shadow.jsonl (gitignored, ephemeral).
//
// Usage:
//   node scripts/benchmark.js --reps=10      # full run, ~2.3 h for reps=10
//   node scripts/benchmark.js --reps=1       # smoke test, ~14 min
//   node scripts/benchmark.js --resume       # skip completed runIndexes
//
// Exit codes:
//   0 — all runs completed (some may be status=error, see failure rate)
//   1 — harness or preview-server failure before any run
//   2 — interrupted (SIGINT)

import { spawn } from "node:child_process";
import {
  createWriteStream,
  existsSync,
  readFileSync,
  mkdirSync,
  appendFileSync,
} from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import puppeteer from "puppeteer";

// --- Config ---------------------------------------------------------------

const RUN_MS = 60_000; // HEADLESS_RUN_MS
const COOLDOWN_MS = 10_000; // HEADLESS_COOLDOWN_MS
const WARMUP_FRAMES = 30; // drop first N frames (JIT + initial paint)
const SHADOW_MAX_FRAMES = 30_000; // upper bound for 60 s at any workload
const SEED = 42;
const PREVIEW_PORT = 4173;
const PREVIEW_URL = `http://localhost:${PREVIEW_PORT}/tempo/`;

const WORKLOADS = ["constant", "sawtooth", "burst", "scroll"];
const SCHEDULERS = ["B0", "B1", "Predictor"];

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");
const OUT_RESULTS = resolve(REPO_ROOT, "docs/PHASE5_PART1_RESULTS.jsonl");
const OUT_SHADOW = resolve(REPO_ROOT, "shadow.jsonl");

const CHROME_FLAGS = [
  "--disable-background-timer-throttling",
  "--disable-renderer-backgrounding",
  "--disable-backgrounding-occluded-windows",
];

// --- Args parse -----------------------------------------------------------

function parseArgs(argv) {
  const args = { reps: 10, resume: false };
  for (const a of argv.slice(2)) {
    if (a.startsWith("--reps=")) {
      const n = parseInt(a.split("=")[1], 10);
      if (!Number.isInteger(n) || n <= 0) {
        throw new Error(`--reps must be a positive integer, got ${a}`);
      }
      args.reps = n;
    } else if (a === "--resume") {
      args.resume = true;
    } else {
      throw new Error(`unknown argument: ${a}`);
    }
  }
  return args;
}

// --- Utilities ------------------------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fisherYates(arr, rng) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function fmtDuration(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h${String(m % 60).padStart(2, "0")}m${String(s % 60).padStart(2, "0")}s`;
  if (m > 0) return `${m}m${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
}

function log(msg) {
  process.stderr.write(`[${new Date().toISOString()}] ${msg}\n`);
}

// --- Run plan -------------------------------------------------------------

function buildRunPlan(reps) {
  const plan = [];
  let runIndex = 0;
  for (const workload of WORKLOADS) {
    for (const active of SCHEDULERS) {
      for (let rep = 0; rep < reps; rep++) {
        plan.push({ runIndex: runIndex++, workload, active, rep });
      }
    }
  }
  // Shuffle with seeded PRNG so execution order is reproducible across
  // --resume invocations. Each entry's executionPosition is assigned AFTER
  // the shuffle so thermal-attribution joins back to wall-clock order.
  const shuffled = fisherYates(plan, mulberry32(SEED));
  return shuffled.map((p, i) => ({ ...p, executionPosition: i }));
}

function loadCompletedRunIndexes(path) {
  if (!existsSync(path)) return new Set();
  const completed = new Set();
  const data = readFileSync(path, "utf-8");
  for (const line of data.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const rec = JSON.parse(trimmed);
      if (rec.runIndex != null && rec.status === "ok") {
        completed.add(rec.runIndex);
      }
    } catch {
      /* ignore malformed line */
    }
  }
  return completed;
}

// --- Preview server -------------------------------------------------------

async function waitForPreview(maxMs = 15_000) {
  const deadline = Date.now() + maxMs;
  let lastErr;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(PREVIEW_URL);
      if (res.ok) return;
      lastErr = new Error(`status ${res.status}`);
    } catch (e) {
      lastErr = e;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(
    `vite preview did not respond at ${PREVIEW_URL} within ${maxMs}ms: ${lastErr?.message ?? "unknown"}`,
  );
}

function startPreview() {
  const child = spawn(
    "npx",
    ["vite", "preview", "--port", String(PREVIEW_PORT)],
    {
      cwd: REPO_ROOT,
      shell: true,
      stdio: ["ignore", "ignore", "inherit"],
    },
  );
  child.on("error", (e) => {
    process.stderr.write(`[preview spawn error] ${e.message}\n`);
  });
  return child;
}

function stopPreview(child) {
  if (!child || child.killed) return;
  try {
    child.kill("SIGTERM");
  } catch {
    /* ignore */
  }
}

// --- Single-run execution -------------------------------------------------

async function runOne(page, plan, shadowStream) {
  // Reload to isolate from previous run's JIT / GC state.
  await page.goto(PREVIEW_URL, { waitUntil: "networkidle0" });
  await page.waitForFunction(() => typeof window.tempo === "object");

  // Init with shadow log. Allocating up-front means per-frame logging is
  // allocation-free (Float32Array + Uint8Array writes only).
  await page.evaluate(
    (seed, workload, active, shadowMax) => {
      window.tempo.stop();
      window.tempo.init(seed, workload, active, {
        shadowMaxFrames: shadowMax,
      });
      window.tempo.start();
    },
    SEED,
    plan.workload,
    plan.active,
    SHADOW_MAX_FRAMES,
  );

  const startedAt = new Date().toISOString();
  const t0 = Date.now();
  await new Promise((r) => setTimeout(r, RUN_MS));
  const durationMs = Date.now() - t0;

  // Snapshot metrics + shadow log, then stop.
  const payload = await page.evaluate(() => {
    const state = window.tempo.getState();
    const loop = state.loop;
    const active = state.activeName;
    const m = loop.getMetrics()[active];
    const shadow = loop.getShadowLog();
    window.tempo.stop();
    return {
      allStats: m.all.getStats(),
      recentStats: m.recent.getStats(),
      shadow,
    };
  });

  // Compute warmup-filtered stats from the shadow log. This is the source
  // of truth for the committed results — page metrics include the first
  // WARMUP_FRAMES frames; the filtered stats do not.
  const filtered = filterAndSummarize(payload.shadow, WARMUP_FRAMES);

  // Stream raw shadow frames to shadow.jsonl (one per line).
  streamShadowFrames(shadowStream, plan, payload.shadow, WARMUP_FRAMES);

  return {
    status: "ok",
    runIndex: plan.runIndex,
    executionPosition: plan.executionPosition,
    workload: plan.workload,
    active: plan.active,
    rep: plan.rep,
    startedAt,
    durationMs,
    warmupFramesDropped: WARMUP_FRAMES,
    frameCount: filtered.frameCount,
    // Metrics computed from shadow log (authoritative). Page-reported
    // metrics included for cross-check.
    jankRate: filtered.jankRate,
    p95: filtered.p95,
    p99: filtered.p99,
    meanDt: filtered.meanDt,
    pageJankRate: payload.allStats.jankRate,
    pageP95: payload.allStats.p95,
    pageMeanDt: payload.allStats.meanDt,
    // Confusion-matrix counts per shadow scheduler. Truth = miss flag
    // observed on the active scheduler's executed dt. Prediction =
    // scheduler's decision ∈ {full=negative, reduce|degrade=positive}.
    confusion: filtered.confusion,
  };
}

function filterAndSummarize(shadow, warmupN) {
  // Drop the first `warmupN` frames. Apply the same rules the Rolling
  // metrics use (jank = dt > FRAME_BUDGET_60 + JANK_TOLERANCE_MS). These
  // values are hard-coded here to keep benchmark.js self-contained rather
  // than importing core/constants.js (which would pull in other modules).
  const FRAME_BUDGET_60 = 16.67;
  const JANK_TOLERANCE_MS = 1.0;
  const threshold = FRAME_BUDGET_60 + JANK_TOLERANCE_MS;

  const n = Math.max(0, shadow.count - warmupN);
  if (n === 0) {
    return {
      frameCount: 0,
      jankRate: 0,
      p95: 0,
      p99: 0,
      meanDt: 0,
      confusion: emptyConfusion(),
    };
  }

  const dt = shadow.dt.slice(warmupN);
  const miss = shadow.miss.slice(warmupN);
  const decisions = shadow.decisions.slice(warmupN * 3);

  let jank = 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (dt[i] > threshold) jank++;
    sum += dt[i];
  }
  const meanDt = sum / n;
  const jankRate = jank / n;

  const sorted = [...dt].sort((a, b) => a - b);
  const p95 = sorted[Math.min(n - 1, Math.floor(n * 0.95))];
  const p99 = sorted[Math.min(n - 1, Math.floor(n * 0.99))];

  const confusion = emptyConfusion();
  for (let i = 0; i < n; i++) {
    const truth = miss[i] === 1;
    for (let s = 0; s < 3; s++) {
      const d = decisions[i * 3 + s];
      const predMiss = d !== 0;
      const who = ["B0", "B1", "Predictor"][s];
      if (truth && predMiss) confusion[who].tp++;
      else if (!truth && predMiss) confusion[who].fp++;
      else if (truth && !predMiss) confusion[who].fn++;
      else confusion[who].tn++;
    }
  }

  return { frameCount: n, jankRate, p95, p99, meanDt, confusion };
}

function emptyConfusion() {
  return {
    B0: { tp: 0, fp: 0, tn: 0, fn: 0 },
    B1: { tp: 0, fp: 0, tn: 0, fn: 0 },
    Predictor: { tp: 0, fp: 0, tn: 0, fn: 0 },
  };
}

function streamShadowFrames(stream, plan, shadow, warmupN) {
  const DECISION_NAMES = ["full", "reduce", "degrade"];
  for (let i = warmupN; i < shadow.count; i++) {
    const rec = {
      runIndex: plan.runIndex,
      workload: plan.workload,
      active: plan.active,
      frameIdx: i - warmupN,
      dt: shadow.dt[i],
      miss: shadow.miss[i] === 1,
      decisions: {
        B0: DECISION_NAMES[shadow.decisions[i * 3]],
        B1: DECISION_NAMES[shadow.decisions[i * 3 + 1]],
        Predictor: DECISION_NAMES[shadow.decisions[i * 3 + 2]],
      },
    };
    stream.write(JSON.stringify(rec) + "\n");
  }
}

// --- Top-level orchestration ----------------------------------------------

let sigintCaught = false;
process.on("SIGINT", () => {
  sigintCaught = true;
  log("SIGINT received — finishing current run, then stopping.");
});

async function main() {
  const args = parseArgs(process.argv);
  const plan = buildRunPlan(args.reps);
  const total = plan.length;

  mkdirSync(dirname(OUT_RESULTS), { recursive: true });

  const completed = args.resume
    ? loadCompletedRunIndexes(OUT_RESULTS)
    : new Set();
  const remaining = plan.filter((p) => !completed.has(p.runIndex));

  log(
    `Plan: ${total} runs total, ${completed.size} already ok, ${remaining.length} to execute.`,
  );
  log(
    `Per-run: ${RUN_MS / 1000}s measurement + ${COOLDOWN_MS / 1000}s cooldown. Seed ${SEED}.`,
  );

  let preview;
  let browser;
  let page;
  let shadowStream;

  try {
    preview = startPreview();
    await waitForPreview();
    log(`Preview ready at ${PREVIEW_URL}`);

    browser = await puppeteer.launch({ headless: true, args: CHROME_FLAGS });
    page = await browser.newPage();

    shadowStream = createWriteStream(OUT_SHADOW, { flags: "a" });

    const startedAt = Date.now();
    let okCount = 0;
    let errCount = 0;

    for (let i = 0; i < remaining.length; i++) {
      if (sigintCaught) break;

      const p = remaining[i];
      const elapsed = Date.now() - startedAt;
      const avgPerRun = i > 0 ? elapsed / i : RUN_MS + COOLDOWN_MS;
      const eta = Math.max(0, (remaining.length - i) * avgPerRun);

      log(
        `[${i + 1}/${remaining.length}] run: ${p.workload} × ${p.active} × rep ${p.rep} (idx ${p.runIndex}, pos ${p.executionPosition}) · elapsed ${fmtDuration(elapsed)} · ETA ${fmtDuration(eta)}`,
      );

      let record;
      try {
        record = await runOne(page, p, shadowStream);
        okCount++;
      } catch (e) {
        record = {
          status: "error",
          runIndex: p.runIndex,
          executionPosition: p.executionPosition,
          workload: p.workload,
          active: p.active,
          rep: p.rep,
          error: e?.message ?? String(e),
          startedAt: new Date().toISOString(),
        };
        errCount++;
        log(`  → ERROR: ${record.error}`);
      }

      appendFileSync(OUT_RESULTS, JSON.stringify(record) + "\n");

      if (i < remaining.length - 1 && !sigintCaught) {
        await new Promise((r) => setTimeout(r, COOLDOWN_MS));
      }
    }

    const totalElapsed = Date.now() - startedAt;
    log(
      `Done. ${okCount} ok, ${errCount} error, ${sigintCaught ? "interrupted" : "complete"}. Total ${fmtDuration(totalElapsed)}.`,
    );
    log(`Results: ${OUT_RESULTS}`);
    log(`Shadow:  ${OUT_SHADOW} (ephemeral, gitignored)`);
    process.exit(sigintCaught ? 2 : 0);
  } finally {
    if (shadowStream) shadowStream.end();
    if (browser) await browser.close().catch(() => {});
    stopPreview(preview);
  }
}

main().catch((e) => {
  process.stderr.write(
    `[benchmark] fatal: ${e?.stack ?? e?.message ?? String(e)}\n`,
  );
  process.exit(1);
});
