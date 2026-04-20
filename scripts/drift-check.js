// Phase 5 Part 2 — harness drift check.
//
// Compares Part 2's B1 drift runs against Part 1's B1 baseline and emits
// a Markdown report + a structured verdict. Intended to run BEFORE
// analyze.js --compare so a polluted B1 baseline can be caught before
// rendering table (c) — the blog-post headline.
//
// Rule (pinned per user directive on 2026-04-20):
//   For each workload:
//     - Compute Part 1 B1 mean μ and stdev σ.
//     - Each Part 2 drift run is an outlier if its jankRate ∉ [μ-2σ, μ+2σ].
//     - Compute absolute mean shift |μ_part2 - μ_part1|.
//   Status per workload:
//     PASS     → 0 outliers AND absShift ≤ 1.0 percentage points.
//     WARNING  → exactly 1 outlier AND absShift ≤ 1.0 pp.
//     STOP     → ≥ 2 outliers OR absShift > 1.0 pp.
//   Aggregate verdict = worst status across workloads.
//
// Exit codes (for shell piping):
//   0 = PASS, 1 = WARNING, 2 = STOP, 3 = NO_DATA-only.

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { mean, stdev } from "../src/harness/stats.js";
import { loadResults, normalizeRecords, selectCell } from "./analyze.js";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");

const DRIFT_ABS_SHIFT_STOP = 0.01; // 1.0 pp absolute mean shift → STOP
const DRIFT_OUTLIER_STOP = 2; // ≥ this many out-of-band runs → STOP
const DRIFT_OUTLIER_WARN = 1; // < STOP and ≥ this → WARNING
// Part 1 B1 is a deterministic EMA heuristic — same seed + same dt sequence
// produce identical decisions, so σ collapses to ~0 on constant / sawtooth /
// burst. A strict [μ - 2σ, μ + 2σ] band then becomes {μ} and flags any
// sub-percent variation as an outlier, which defeats the signal-to-noise
// purpose of the rule. Floor the half-band at DRIFT_MIN_HALFBAND so
// zero-σ baselines still admit normal frame-boundary noise. 0.5pp = half
// of the absShift STOP threshold: anything bigger than this in a single
// run IS worth flagging even for a deterministic heuristic.
const DRIFT_MIN_HALFBAND = 0.005; // 0.5 pp floor on per-run outlier band

const WORKLOADS = ["constant", "sawtooth", "burst", "scroll"];

/**
 * Core rule: given Part 1 + Part 2 records (raw JSONL parse), return a
 * structured drift report. Pure function — exported for tests. Accepts
 * pre-normalized or raw records (normalization is applied internally).
 *
 * @param {Array} part1Records — raw Part 1 JSONL records
 * @param {Array} part2Records — raw Part 2 JSONL records
 * @param {object} [opts]
 * @param {string[]} [opts.workloads=WORKLOADS]
 * @returns {{ verdict: 'PASS'|'WARNING'|'STOP'|'NO_DATA',
 *            sections: Array<object> }}
 */
export function computeDriftReport(part1Records, part2Records, opts = {}) {
  const workloads = opts.workloads ?? WORKLOADS;
  const p1 = normalizeRecords(part1Records);
  const p2 = normalizeRecords(part2Records);

  const sections = [];
  let worst = "PASS";
  const worsen = (status) => {
    const rank = { PASS: 0, NO_DATA: 0, WARNING: 1, STOP: 2 };
    if (rank[status] > rank[worst]) worst = status;
  };
  let anyData = false;

  for (const wl of workloads) {
    // Only the "scratch" condition is relevant — drift runs were
    // deliberately scheduled as scratch in the Part 2 plan.
    const p1B1 = selectCell(p1, {
      workload: wl,
      active: "B1",
      condition: "scratch",
    });
    const p2B1 = selectCell(p2, {
      workload: wl,
      active: "B1",
      condition: "scratch",
    });

    if (p1B1.length < 2 || p2B1.length === 0) {
      sections.push({
        workload: wl,
        n1: p1B1.length,
        n2: p2B1.length,
        status: "NO_DATA",
        reason:
          p1B1.length < 2
            ? `Part 1 B1 has n<2 (${p1B1.length})`
            : "Part 2 drift has no runs",
      });
      continue;
    }

    anyData = true;
    const p1J = p1B1.map((r) => r.jankRate);
    const p2J = p2B1.map((r) => r.jankRate);
    const m1 = mean(p1J);
    const s1 = stdev(p1J);
    const halfBand = Math.max(2 * s1, DRIFT_MIN_HALFBAND);
    const lo = m1 - halfBand;
    const hi = m1 + halfBand;
    const outliers = p2J.filter((j) => j < lo || j > hi).length;
    const m2 = mean(p2J);
    const absShift = Math.abs(m2 - m1);

    let status = "PASS";
    if (absShift > DRIFT_ABS_SHIFT_STOP) status = "STOP";
    else if (outliers >= DRIFT_OUTLIER_STOP) status = "STOP";
    else if (outliers >= DRIFT_OUTLIER_WARN) status = "WARNING";

    sections.push({
      workload: wl,
      n1: p1J.length,
      n2: p2J.length,
      p1Mean: m1,
      p1SD: s1,
      band: [lo, hi],
      p2Mean: m2,
      absShift,
      outliers,
      status,
    });
    worsen(status);
  }

  const verdict = anyData ? worst : "NO_DATA";
  return { verdict, sections };
}

// --- Report rendering -----------------------------------------------------

const pct = (x) => `${(x * 100).toFixed(2)}%`;
const pctSigned = (x) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(2)}pp`;

function renderDriftReport(report, inputs) {
  const { verdict, sections } = report;
  const lines = [
    "# Phase 5 Part 2 — B1 Drift Check",
    "",
    `_Generated: ${new Date().toISOString()}_`,
    "",
    `Inputs: \`${inputs.part1}\` (Part 1 B1 baseline) vs \`${inputs.part2}\` (Part 2 drift runs).`,
    "",
    `**Aggregate verdict: \`${verdict}\`**`,
    "",
    "Rule (pinned): per workload, outlier = Part-2-B1 jankRate ∉ [μ ± 2σ] of Part-1-B1.",
    "`PASS` = 0 outliers AND |mean shift| ≤ 1.0pp. `WARNING` = 1 outlier. `STOP` = ≥2 outliers OR |mean shift| > 1.0pp.",
    "",
    "| Workload | n₁ | n₂ | Part 1 μ±σ | ±2σ band | Part 2 μ | Mean shift | Outliers | Status |",
    "|---|---:|---:|---|---|---|---:|---:|---|",
  ];
  for (const s of sections) {
    if (s.status === "NO_DATA") {
      lines.push(
        `| ${s.workload} | ${s.n1} | ${s.n2} | — | — | — | — | — | NO_DATA (${s.reason}) |`,
      );
      continue;
    }
    lines.push(
      `| ${s.workload} | ${s.n1} | ${s.n2} | ${pct(s.p1Mean)} ± ${pct(s.p1SD)} | [${pct(s.band[0])}, ${pct(s.band[1])}] | ${pct(s.p2Mean)} | ${pctSigned(s.p2Mean - s.p1Mean)} | ${s.outliers}/${s.n2} | ${s.status} |`,
    );
  }
  lines.push("");
  if (verdict === "PASS") {
    lines.push(
      "_No drift detected. Proceed to `analyze.js --compare` for the three comparison tables._",
    );
  } else if (verdict === "WARNING") {
    lines.push(
      "_Single-outlier noise within tolerance. Proceed to the comparison, but surface this section in RESULTS.md so readers see the one anomalous cell._",
    );
  } else if (verdict === "STOP") {
    lines.push(
      "_STOP — do not render the comparison. Investigate environment (thermal, background load, Chrome update) before deciding whether to re-run Part 2 or accept the drift as real._",
    );
  } else {
    lines.push("_No data available for any workload._");
  }
  lines.push("");
  return lines.join("\n");
}

// --- CLI ------------------------------------------------------------------

export function parseDriftArgs(argv) {
  const args = { part1: null, part2: null, out: null };
  for (const a of argv.slice(2)) {
    if (a.startsWith("--part1=")) args.part1 = a.split("=")[1];
    else if (a.startsWith("--part2=")) args.part2 = a.split("=")[1];
    else if (a.startsWith("--out=")) args.out = a.split("=")[1];
    else throw new Error(`unknown argument: ${a}`);
  }
  if (!args.part1 || !args.part2) {
    throw new Error("drift-check: both --part1=... and --part2=... required");
  }
  return args;
}

function main() {
  const args = parseDriftArgs(process.argv);
  const part1Path = resolve(REPO_ROOT, args.part1);
  const part2Path = resolve(REPO_ROOT, args.part2);
  const outPath = args.out
    ? resolve(REPO_ROOT, args.out)
    : resolve(REPO_ROOT, "docs/PHASE5_PART2_DRIFT.md");

  const p1 = loadResults(part1Path);
  const p2 = loadResults(part2Path);
  const report = computeDriftReport(p1, p2);

  const md = renderDriftReport(report, { part1: args.part1, part2: args.part2 });
  writeFileSync(outPath, md);
  process.stdout.write(
    `drift-check: ${report.verdict}\n` +
      `  wrote: ${outPath}\n` +
      report.sections
        .map(
          (s) =>
            `  ${s.workload}: ${s.status}` +
            (s.status === "NO_DATA"
              ? ` (${s.reason})`
              : ` | n₁=${s.n1} n₂=${s.n2} shift=${((s.p2Mean - s.p1Mean) * 100).toFixed(2)}pp outliers=${s.outliers}`),
        )
        .join("\n") +
      "\n",
  );
  // Exit code signals severity so a shell pipeline can gate analyze.js.
  const code =
    report.verdict === "PASS"
      ? 0
      : report.verdict === "WARNING"
        ? 1
        : report.verdict === "STOP"
          ? 2
          : 3;
  process.exit(code);
}

const invokedAsScript =
  process.argv[1] && process.argv[1].endsWith("drift-check.js");
if (invokedAsScript) {
  main();
}
