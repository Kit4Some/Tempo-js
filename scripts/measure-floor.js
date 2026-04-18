// Phase 5 Part 0 — Headless benchmark floor measurement.
//
// Answers: "is the headless harness quiet enough that scheduler deltas
// can be measured above the noise floor?"
//
// Protocol (PHASE5_NOTES.md § Part 0):
//   1. npm run build → dist/
//   2. vite preview serves dist/ at http://localhost:4173/tempo/
//   3. Puppeteer launches headless Chrome with the spec §5 flags
//      (--disable-background-timer-throttling, --disable-renderer-
//      backgrounding, --disable-backgrounding-occluded-windows)
//   4. Navigate to the preview URL, stop the auto-started loop, init
//      with (seed=42, workload=constant, active=B0), start, wait 60s,
//      read B0.all.getStats() — that's the harness's ambient jank floor
//      (B0 never reduces/degrades, so any jank is pure harness noise).
//
// Gates (PHASE5_NOTES.md § Part 0):
//   - floor > live (~10%)   → DEBUG (harness worse than live, broken)
//   - floor ≥ 8%             → DEBUG (suspect Puppeteer/flags/page)
//   - floor 3–8%             → PROCEED with new baseline
//   - floor 0–2%             → PROCEED (tighter Go/No-Go threshold)
//
// Output:
//   - docs/PHASE5_PART0_FLOOR.json — machine-readable
//   - docs/PHASE5_PART0_FLOOR.md   — human-readable
//   - stdout summary
//   - exit code: 0 (PROCEED) or 1 (DEBUG)

import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import puppeteer from "puppeteer";

const RUN_MS = 60_000; // HEADLESS_RUN_MS — PHASE5_NOTES calibration
const SEED = 42;
const WORKLOAD = "constant";
const ACTIVE = "B0";
const PREVIEW_PORT = 4173;
const PREVIEW_URL = `http://localhost:${PREVIEW_PORT}/tempo/`;
const LIVE_FLOOR_PERCENT = 10; // from Phase 4 verification

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");
const OUT_JSON = resolve(REPO_ROOT, "docs/PHASE5_PART0_FLOOR.json");
const OUT_MD = resolve(REPO_ROOT, "docs/PHASE5_PART0_FLOOR.md");

function decideGate(floorPercent) {
  if (floorPercent > LIVE_FLOOR_PERCENT) {
    return {
      gate: "DEBUG",
      reason: `Headless floor ${floorPercent.toFixed(2)}% > live floor ${LIVE_FLOOR_PERCENT}%. Harness is noisier than the live page — flags, Puppeteer config, or page overhead is broken.`,
    };
  }
  if (floorPercent >= 8) {
    return {
      gate: "DEBUG",
      reason: `Headless floor ${floorPercent.toFixed(2)}% ≥ 8%. Suspect Puppeteer flags / page overhead; investigate before full benchmark.`,
    };
  }
  if (floorPercent >= 3) {
    return {
      gate: "PROCEED",
      reason: `Headless floor ${floorPercent.toFixed(2)}% in 3–8% band. Acknowledge in RESULTS.md as the new baseline and continue.`,
    };
  }
  return {
    gate: "PROCEED",
    reason: `Headless floor ${floorPercent.toFixed(2)}% < 3%. Go/No-Go threshold becomes relatively more discriminating.`,
  };
}

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
  // shell:true for Windows npm shim compatibility. Inherit stderr so
  // preview errors show up in our stdout; silence stdout to keep the
  // measurement output clean.
  const child = spawn("npx", ["vite", "preview", "--port", String(PREVIEW_PORT)], {
    cwd: REPO_ROOT,
    shell: true,
    stdio: ["ignore", "ignore", "inherit"],
  });
  child.on("error", (e) => {
    process.stderr.write(`[preview spawn error] ${e.message}\n`);
  });
  return child;
}

function stopPreview(child) {
  if (!child || child.killed) return;
  // On Windows, simple .kill() may not propagate to vite's child renderer
  // process — fine for this one-off, the OS cleans up when Node exits.
  try {
    child.kill("SIGTERM");
  } catch {
    /* ignore */
  }
}

async function measureFloor() {
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      "--disable-background-timer-throttling",
      "--disable-renderer-backgrounding",
      "--disable-backgrounding-occluded-windows",
    ],
  });

  try {
    const page = await browser.newPage();
    const consoleErrors = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(err.message));

    await page.goto(PREVIEW_URL, { waitUntil: "networkidle0" });

    // Wait for window.tempo to exist (DOMContentLoaded handler completes).
    await page.waitForFunction(() => typeof window.tempo === "object");

    // Stop the page's auto-started loop and re-init with our exact
    // parameters. Reset the loop so metrics start clean.
    await page.evaluate(
      (seed, workload, active) => {
        window.tempo.stop();
        window.tempo.init(seed, workload, active);
        window.tempo.start();
      },
      SEED,
      WORKLOAD,
      ACTIVE,
    );

    // Run the measurement window.
    await new Promise((r) => setTimeout(r, RUN_MS));

    // Read stats, then stop.
    const stats = await page.evaluate((active) => {
      const m = window.tempo.getState().loop.getMetrics();
      window.tempo.stop();
      return {
        all: m[active].all.getStats(),
        recent: m[active].recent.getStats(),
      };
    }, ACTIVE);

    return { stats, consoleErrors };
  } finally {
    await browser.close();
  }
}

async function main() {
  let preview;
  try {
    preview = startPreview();
    await waitForPreview();

    const { stats, consoleErrors } = await measureFloor();
    const floorPercent = stats.all.jankRate * 100;
    const { gate, reason } = decideGate(floorPercent);

    const result = {
      timestamp: new Date().toISOString(),
      seed: SEED,
      workload: WORKLOAD,
      active: ACTIVE,
      run_ms: RUN_MS,
      chrome_flags: [
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
      ],
      stats: {
        all: stats.all,
        recent: stats.recent,
      },
      floor_percent: floorPercent,
      live_floor_percent: LIVE_FLOOR_PERCENT,
      gate,
      gate_reason: reason,
      console_errors: consoleErrors,
      node_version: process.version,
      platform: `${process.platform} ${process.arch}`,
    };

    mkdirSync(dirname(OUT_JSON), { recursive: true });
    writeFileSync(OUT_JSON, JSON.stringify(result, null, 2));

    const md = [
      "# Phase 5 Part 0 — Headless Floor",
      "",
      `_Generated: ${result.timestamp} · Node ${result.node_version} · ${result.platform}_`,
      "",
      `60 s run, workload \`${WORKLOAD}\`, active \`${ACTIVE}\`, seed ${SEED}.`,
      "B0 never reduces or degrades, so any jank observed is pure harness",
      "noise — scheduler behaviour cannot contribute. This is the floor.",
      "",
      "## Chrome flags",
      "",
      ...result.chrome_flags.map((f) => `- \`${f}\``),
      "",
      "## Measurement",
      "",
      "| Window | Frames | Jank rate | P95 (ms) | Mean (ms) |",
      "|---|---:|---:|---:|---:|",
      `| all | ${stats.all.frameCount} | ${(stats.all.jankRate * 100).toFixed(2)}% | ${stats.all.p95.toFixed(2)} | ${stats.all.meanDt.toFixed(2)} |`,
      `| recent (300) | ${stats.recent.frameCount} | ${(stats.recent.jankRate * 100).toFixed(2)}% | ${stats.recent.p95.toFixed(2)} | ${stats.recent.meanDt.toFixed(2)} |`,
      "",
      "## Gate decision",
      "",
      `- Live floor reference: ${LIVE_FLOOR_PERCENT}% (Phase 4 verification).`,
      `- Headless floor (this run): **${floorPercent.toFixed(2)}%**`,
      `- Outcome: **${gate}**`,
      `- Reason: ${reason}`,
      "",
      ...(consoleErrors.length
        ? [
            "## Browser console errors",
            "",
            ...consoleErrors.map((e) => `- \`${e}\``),
            "",
          ]
        : []),
    ].join("\n");

    writeFileSync(OUT_MD, md);

    process.stdout.write(`
Phase 5 Part 0 — Floor
  frames: ${stats.all.frameCount}
  jank:   ${floorPercent.toFixed(2)}%
  P95:    ${stats.all.p95.toFixed(2)} ms
  mean:   ${stats.all.meanDt.toFixed(2)} ms
  ─────────────────────────────────────
  gate: ${gate}
  → ${reason}

  wrote: ${OUT_JSON}
  wrote: ${OUT_MD}
`);

    process.exit(gate === "DEBUG" ? 1 : 0);
  } finally {
    stopPreview(preview);
  }
}

main().catch((e) => {
  process.stderr.write(`[measure-floor] fatal: ${e.stack ?? e.message}\n`);
  process.exit(2);
});
