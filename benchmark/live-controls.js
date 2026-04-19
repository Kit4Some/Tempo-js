/**
 * Pure UI helpers for the Phase 4 live benchmark page. Separated from
 * app.js so the DOM-free logic (stat formatting, Run-all-3 orchestration)
 * can be unit-tested without jsdom.
 */

/**
 * Parse the Phase 5 Part 2 init-mode URL query into a plain options object.
 * Pure string → object — no `location` access — so it unit-tests without
 * jsdom and reuses cleanly from Puppeteer (build a query, call the parser,
 * hand off to init()).
 *
 * Accepted params (anything else is ignored):
 *   ?init=scratch       → { pretrained: false, freeze: false }  (default)
 *   ?init=pretrained    → { pretrained: true, freeze: false }
 *   &freeze=true        → overrides freeze to true (literal 'true' only)
 *
 * Strict whitelist on `init`: an unknown value falls back to 'scratch' to
 * keep surprises out of the benchmark. A permissive parser would let a
 * typo silently run the wrong condition and taint the results.
 *
 * @param {string} search — typically location.search ("?k=v&..."), may be empty
 * @returns {{pretrained: boolean, freeze: boolean}}
 */
export function parseInitOpts(search) {
  const params = new URLSearchParams(search || "");
  const initVal = params.get("init");
  const pretrained = initVal === "pretrained";
  const freeze = params.get("freeze") === "true";
  return { pretrained, freeze };
}

export function formatJank(rate) {
  return `${(rate * 100).toFixed(1)}%`;
}

export function formatMs(ms) {
  return ms.toFixed(1);
}

const EMPTY = "—";

/**
 * Build 3 row descriptors (B0, B1, Predictor) from a SequentialLoop's
 * metrics map. Returns plain objects — the caller translates to DOM. This
 * keeps the formatting logic testable and the DOM write a trivial
 * string-template.
 *
 * A scheduler with frameCount=0 in the `all` track shows "—" everywhere
 * in its row; a scheduler with non-empty `all` but empty `recent` (e.g.,
 * just after a workload change) still shows the lifetime numbers and "—"
 * for the rolling ones.
 *
 * @param {Record<'B0'|'B1'|'Predictor', {all:{getStats():object}, recent:{getStats():object}}>} metrics
 * @param {'B0'|'B1'|'Predictor'} activeName
 */
export function buildStatsRows(metrics, activeName) {
  const names = ["B0", "B1", "Predictor"];
  return names.map((name) => {
    const m = metrics[name];
    const all = m.all.getStats();
    const recent = m.recent.getStats();
    const allEmpty = all.frameCount === 0;
    const recentEmpty = recent.frameCount === 0;
    return {
      name,
      active: name === activeName,
      jankAll: allEmpty ? EMPTY : formatJank(all.jankRate),
      jankRecent: recentEmpty ? EMPTY : formatJank(recent.jankRate),
      p95All: allEmpty ? EMPTY : formatMs(all.p95),
      p95Recent: recentEmpty ? EMPTY : formatMs(recent.p95),
      meanDt: allEmpty ? EMPTY : formatMs(all.meanDt),
    };
  });
}

/**
 * Run-all-3 sequence (B0 → cooldown → B1 → cooldown → Predictor). Each
 * scheduler gets a fresh loop state (loop.reset()) for isolation, matching
 * the §5 Phase 5 Sequential methodology at a shortened live-demo cadence.
 *
 * Phase callbacks:
 *   { kind: 'started',  name, index, total }
 *   { kind: 'finished', name, index, total }
 *   { kind: 'cooldown', name, index, total }   // fires only between schedulers
 *   { kind: 'complete' }
 *
 * Aborting (AbortSignal) throws an AbortError the moment the next phase
 * boundary is reached. A pre-aborted signal throws before any loop
 * mutation, so callers can wire a Cancel button without worrying about
 * partial state.
 *
 * @param {object} args
 * @param {{reset():void, setActive(name:string):void}} args.loop
 * @param {AbortSignal} args.signal
 * @param {(phase:object) => void} args.onPhase
 * @param {number} args.runMs
 * @param {number} args.cooldownMs
 * @param {(ms:number, signal:AbortSignal) => Promise<void>} args.sleep
 */
export async function runSequence({
  loop,
  signal,
  onPhase,
  runMs,
  cooldownMs,
  sleep,
}) {
  const names = ["B0", "B1", "Predictor"];
  const total = names.length;

  const bail = () => {
    if (signal?.aborted) {
      throw new DOMException("Run-all-3 aborted", "AbortError");
    }
  };

  bail();

  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    loop.reset();
    loop.setActive(name);
    onPhase({ kind: "started", name, index: i, total });
    await sleep(runMs, signal);
    bail();
    onPhase({ kind: "finished", name, index: i, total });
    if (i < names.length - 1) {
      onPhase({ kind: "cooldown", name, index: i, total });
      await sleep(cooldownMs, signal);
      bail();
    }
  }
  onPhase({ kind: "complete" });
}
