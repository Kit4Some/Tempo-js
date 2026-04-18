// Phase 5 Part 1 — Go/No-Go analysis + RESULTS.md generator.
//
// Reads docs/PHASE5_PART1_RESULTS.jsonl (one JSON record per run) and
// writes docs/RESULTS.md with:
//   1. Run provenance (N reps, ok rate, failures list).
//   2. Per-workload × scheduler summary tables (mean + 95% bootstrap CI
//      for jankRate, p95, meanDt).
//   3. Mann-Whitney U + Cohen's d tests: Predictor vs B1 (primary) and
//      Predictor vs B0 (secondary), for each workload, on jankRate.
//   4. Confusion-matrix aggregation + precision/recall/F1 per
//      (workload × scheduler).
//   5. Summary verdict per workload and an overall GO/NO-GO table.
//
// Statistical methodology comes from PHASE5_NOTES.md § Statistical
// methodology:
//   - Mann-Whitney U (non-parametric two-sided)
//   - Cohen's d effect size (|d| ≥ 0.5 required alongside p < 0.05)
//   - 1000-resample percentile bootstrap for 95% CIs
//
// Failure policy: if status='error' rate exceeds 5%, the verdict section
// is suppressed and a WARNING section takes its place. The user's
// directive per PHASE5_NOTES.md § Part 0 Go/No-Go + commit-protocol.

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import {
  mean,
  stdev,
  cohensD,
  mannWhitneyU,
  bootstrapMeanCI,
} from "../src/harness/stats.js";

// Seeded RNG for reproducible bootstrap CIs run-to-run.
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

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");
const IN_RESULTS = resolve(REPO_ROOT, "docs/PHASE5_PART1_RESULTS.jsonl");
const OUT_RESULTS_MD = resolve(REPO_ROOT, "docs/RESULTS.md");

const WORKLOADS = ["constant", "sawtooth", "burst", "scroll"];
const SCHEDULERS = ["B0", "B1", "Predictor"];
const BUDGET_EXCEEDING = new Set(["sawtooth", "burst", "scroll"]); // primary

const GO_P = 0.05;
const GO_D = 0.5;
const MAX_FAILURE_RATE = 0.05;
const BOOT_B = 1000;
const BOOT_SEED = 20260419;

// --- Input parsing --------------------------------------------------------

function loadResults(path) {
  const data = readFileSync(path, "utf-8");
  const records = [];
  for (const line of data.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    records.push(JSON.parse(trimmed));
  }
  return records;
}

function groupByCell(records) {
  // Map<workload, Map<active, Array<record>>>
  const out = new Map();
  for (const wl of WORKLOADS) {
    const m = new Map();
    for (const sc of SCHEDULERS) m.set(sc, []);
    out.set(wl, m);
  }
  for (const r of records) {
    if (r.status !== "ok") continue;
    const cell = out.get(r.workload);
    if (!cell) continue;
    const arr = cell.get(r.active);
    if (!arr) continue;
    arr.push(r);
  }
  return out;
}

// --- Aggregation ---------------------------------------------------------

function summarize(cell, rng) {
  const jank = cell.map((r) => r.jankRate);
  const p95 = cell.map((r) => r.p95);
  const meanDt = cell.map((r) => r.meanDt);
  return {
    n: cell.length,
    jankRate: {
      values: jank,
      mean: mean(jank),
      sd: stdev(jank),
      ci: bootstrapMeanCI(jank, { B: BOOT_B, level: 0.95, rng }),
    },
    p95: {
      values: p95,
      mean: mean(p95),
      sd: stdev(p95),
      ci: bootstrapMeanCI(p95, { B: BOOT_B, level: 0.95, rng }),
    },
    meanDt: {
      values: meanDt,
      mean: mean(meanDt),
      sd: stdev(meanDt),
      ci: bootstrapMeanCI(meanDt, { B: BOOT_B, level: 0.95, rng }),
    },
  };
}

function aggregateConfusion(cell) {
  const out = {};
  for (const sc of SCHEDULERS) out[sc] = { tp: 0, fp: 0, tn: 0, fn: 0 };
  for (const r of cell) {
    if (!r.confusion) continue;
    for (const sc of SCHEDULERS) {
      const c = r.confusion[sc];
      if (!c) continue;
      out[sc].tp += c.tp;
      out[sc].fp += c.fp;
      out[sc].tn += c.tn;
      out[sc].fn += c.fn;
    }
  }
  return out;
}

function prf(c) {
  const precision = c.tp + c.fp === 0 ? null : c.tp / (c.tp + c.fp);
  const recall = c.tp + c.fn === 0 ? null : c.tp / (c.tp + c.fn);
  let f1 = null;
  if (precision != null && recall != null && precision + recall > 0) {
    f1 = (2 * precision * recall) / (precision + recall);
  }
  return { precision, recall, f1 };
}

function compareOnJank(cellA, cellB) {
  const a = cellA.map((r) => r.jankRate);
  const b = cellB.map((r) => r.jankRate);
  if (a.length < 2 || b.length < 2) {
    return { n1: a.length, n2: b.length, insufficient: true };
  }
  const mw = mannWhitneyU(a, b);
  const d = cohensD(a, b);
  return { ...mw, d, insufficient: false };
}

// --- Formatting helpers ---------------------------------------------------

const pct = (x) => `${(x * 100).toFixed(2)}%`;
const ms = (x) => `${x.toFixed(2)} ms`;
const f4 = (x) => (Number.isFinite(x) ? x.toFixed(4) : String(x));
const sci = (x) =>
  x === 0
    ? "0"
    : Math.abs(x) < 1e-3
      ? x.toExponential(2)
      : x.toFixed(4);

function fmtCI(ci, fmt) {
  return `[${fmt(ci[0])}, ${fmt(ci[1])}]`;
}

// --- Markdown builders ----------------------------------------------------

function renderProvenance(all, okRecords) {
  const failed = all.filter((r) => r.status !== "ok");
  const repsPerCell = (() => {
    const seen = new Set();
    for (const r of okRecords) seen.add(`${r.workload}|${r.active}|${r.rep}`);
    // reps-per-cell is the max `rep` index + 1 among ok records.
    const maxRep = okRecords.reduce(
      (acc, r) => Math.max(acc, r.rep ?? 0),
      -1,
    );
    return maxRep + 1;
  })();
  const failureRate = all.length === 0 ? 0 : failed.length / all.length;

  const lines = [
    "## Run provenance",
    "",
    `- Total runs recorded: ${all.length}`,
    `- Successful: ${okRecords.length}`,
    `- Failed: ${failed.length} (${(failureRate * 100).toFixed(2)}%)`,
    `- Reps per (workload × scheduler) cell: ${repsPerCell}`,
    "",
  ];

  if (failed.length > 0) {
    lines.push("### Failed runs");
    lines.push("");
    lines.push("| runIndex | execPos | workload | active | rep | error |");
    lines.push("|---:|---:|---|---|---:|---|");
    for (const r of failed) {
      lines.push(
        `| ${r.runIndex} | ${r.executionPosition ?? ""} | ${r.workload} | ${r.active} | ${r.rep} | \`${(r.error || "").replace(/\|/g, "\\|")}\` |`,
      );
    }
    lines.push("");
  }
  return { md: lines.join("\n"), failureRate, repsPerCell };
}

function renderCellTables(grouped, rng) {
  const lines = ["## Per-workload × scheduler summary", ""];
  for (const wl of WORKLOADS) {
    const cells = grouped.get(wl);
    lines.push(`### ${wl}`);
    lines.push("");
    lines.push(
      "| Scheduler | n | Jank mean [95% CI] | P95 mean [95% CI] | Mean dt [95% CI] |",
    );
    lines.push("|---|---:|---|---|---|");
    for (const sc of SCHEDULERS) {
      const runs = cells.get(sc);
      if (runs.length === 0) {
        lines.push(`| ${sc} | 0 | — | — | — |`);
        continue;
      }
      const s = summarize(runs, rng);
      lines.push(
        `| ${sc} | ${s.n} | ${pct(s.jankRate.mean)} ${fmtCI(s.jankRate.ci, pct)} | ${ms(s.p95.mean)} ${fmtCI(s.p95.ci, ms)} | ${ms(s.meanDt.mean)} ${fmtCI(s.meanDt.ci, ms)} |`,
      );
    }
    lines.push("");
  }
  return lines.join("\n");
}

function renderGoNoGo(grouped) {
  const lines = [
    "## Go/No-Go — Predictor vs B1 (primary) / Predictor vs B0 (secondary)",
    "",
    `Decision gate (PHASE5_NOTES.md § Statistical methodology):`,
    `**Go requires \`p < ${GO_P}\` AND \`|d| ≥ ${GO_D}\` on jankRate, versus B1.**`,
    "",
    "Budget-exceeding workloads (sawtooth / burst / scroll) are primary; constant is sanity-only.",
    "",
    "| Workload | Vs B1: U | Vs B1: p | Vs B1: d | Vs B1: verdict | Vs B0: U | Vs B0: p | Vs B0: d | Vs B0: verdict |",
    "|---|---:|---:|---:|---|---:|---:|---:|---|",
  ];
  const perWorkload = {};
  for (const wl of WORKLOADS) {
    const cells = grouped.get(wl);
    const pred = cells.get("Predictor");
    const b1 = cells.get("B1");
    const b0 = cells.get("B0");
    const predVsB1 = compareOnJank(pred, b1);
    const predVsB0 = compareOnJank(pred, b0);
    const row = [wl];
    const fmtCell = (cmp) => {
      if (cmp.insufficient) return ["—", "—", "—", `INSUFFICIENT n=${cmp.n1}/${cmp.n2}`];
      const verdictParts = [];
      verdictParts.push(cmp.p < GO_P ? "p✓" : "p✗");
      verdictParts.push(Math.abs(cmp.d) >= GO_D ? "d✓" : "d✗");
      const go = cmp.p < GO_P && Math.abs(cmp.d) >= GO_D;
      const sign = cmp.d < 0 ? " (Pred lower)" : cmp.d > 0 ? " (Pred higher)" : "";
      const verdict = `${go ? "GO" : "NO-GO"} (${verdictParts.join(" ")})${sign}`;
      return [`${cmp.U}`, sci(cmp.p), f4(cmp.d), verdict];
    };
    row.push(...fmtCell(predVsB1), ...fmtCell(predVsB0));
    lines.push(`| ${row.join(" | ")} |`);
    perWorkload[wl] = { predVsB1, predVsB0 };
  }
  lines.push("");
  return { md: lines.join("\n"), perWorkload };
}

function renderConfusion(grouped) {
  const lines = ["## Secondary — Shadow prediction quality", ""];
  lines.push(
    "Truth: the active scheduler's executed dt crossed the jank threshold.",
  );
  lines.push(
    "Prediction (per shadow scheduler): decision ∈ {reduce, degrade} counts as positive; full counts as negative.",
  );
  lines.push("");

  for (const wl of WORKLOADS) {
    const cells = grouped.get(wl);
    lines.push(`### ${wl}`);
    lines.push("");
    lines.push(
      "| Scheduler | TP | FP | TN | FN | Precision | Recall | F1 |",
    );
    lines.push("|---|---:|---:|---:|---:|---:|---:|---:|");
    // Use the union confusion from any active cell — shadows are the same
    // regardless of which scheduler was active for that particular run
    // (shadow decisions happen identically). Safer: aggregate across ALL
    // runs of this workload.
    const aggCell = [];
    for (const sc of SCHEDULERS) aggCell.push(...cells.get(sc));
    const conf = aggregateConfusion(aggCell);
    for (const sc of SCHEDULERS) {
      const c = conf[sc];
      const { precision, recall, f1 } = prf(c);
      lines.push(
        `| ${sc} | ${c.tp} | ${c.fp} | ${c.tn} | ${c.fn} | ${precision == null ? "—" : precision.toFixed(3)} | ${recall == null ? "—" : recall.toFixed(3)} | ${f1 == null ? "—" : f1.toFixed(3)} |`,
      );
    }
    lines.push("");
  }
  return lines.join("\n");
}

function renderVerdictSummary(perWorkload, failureRate) {
  if (failureRate > MAX_FAILURE_RATE) {
    return [
      "## ⚠ Verdict withheld",
      "",
      `Failure rate ${(failureRate * 100).toFixed(2)}% exceeds the ${(MAX_FAILURE_RATE * 100).toFixed(0)}% ceiling set in PHASE5_NOTES.md § Part 0 Go/No-Go.`,
      "",
      "Investigate the Failed runs section before re-running this analysis.",
      "",
    ].join("\n");
  }

  const lines = ["## Summary verdict", ""];
  lines.push("| Workload | Category | Predictor vs B1 |");
  lines.push("|---|---|---|");
  let goCount = 0;
  let primaryCount = 0;
  for (const wl of WORKLOADS) {
    const cmp = perWorkload[wl]?.predVsB1;
    const cat = BUDGET_EXCEEDING.has(wl) ? "primary" : "sanity";
    if (!cmp || cmp.insufficient) {
      lines.push(`| ${wl} | ${cat} | insufficient |`);
      continue;
    }
    const go = cmp.p < GO_P && Math.abs(cmp.d) >= GO_D;
    const verdict = go
      ? cmp.d < 0
        ? "GO (Predictor reduces jank)"
        : "GO (opposite direction!)"
      : "NO-GO";
    lines.push(`| ${wl} | ${cat} | ${verdict} |`);
    if (BUDGET_EXCEEDING.has(wl)) {
      primaryCount++;
      if (go && cmp.d < 0) goCount++;
    }
  }
  lines.push("");
  lines.push(`**Primary workloads with GO verdict (Predictor lower than B1):** ${goCount}/${primaryCount}`);
  lines.push("");
  if (primaryCount > 0) {
    const fraction = goCount / primaryCount;
    if (fraction >= 0.5) {
      lines.push(
        `Overall: **${fraction >= 0.66 ? "STRONG GO" : "MARGINAL GO"}** — Predictor wins majority of primary workloads with both statistical and effect-size significance.`,
      );
    } else {
      lines.push(
        `Overall: **NO-GO** — Predictor does not consistently outperform B1 on primary workloads.`,
      );
    }
  }
  lines.push("");
  return lines.join("\n");
}

// --- Main -----------------------------------------------------------------

function main() {
  const all = loadResults(IN_RESULTS);
  const ok = all.filter((r) => r.status === "ok");
  const grouped = groupByCell(ok);
  const rng = mulberry32(BOOT_SEED);

  const { md: provenanceMd, failureRate } = renderProvenance(all, ok);
  const tablesMd = renderCellTables(grouped, rng);
  const { md: gonogoMd, perWorkload } = renderGoNoGo(grouped);
  const confusionMd = renderConfusion(grouped);
  const verdictMd = renderVerdictSummary(perWorkload, failureRate);

  const header = [
    "# Phase 5 Results",
    "",
    `_Generated: ${new Date().toISOString()}_`,
    "",
    "Auto-generated by `scripts/analyze.js` from `docs/PHASE5_PART1_RESULTS.jsonl`.",
    "Statistical methodology pinned in `docs/PHASE5_NOTES.md § Statistical methodology`.",
    "",
  ].join("\n");

  const out = [
    header,
    provenanceMd,
    tablesMd,
    gonogoMd,
    verdictMd,
    confusionMd,
  ].join("\n");

  writeFileSync(OUT_RESULTS_MD, out);
  process.stdout.write(`
Analysis complete.
  ok: ${ok.length} / ${all.length}
  failure rate: ${(failureRate * 100).toFixed(2)}%
  wrote: ${OUT_RESULTS_MD}
`);
}

main();
