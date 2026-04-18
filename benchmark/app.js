// Phase 4 commit 3: skeleton only. The frame loop, scheduler wiring, and
// stats/chart/heatmap updates arrive in commit 4 (frame loop + shared
// FeatureExtractor + Sequential execution) and commit 5 (integration).
//
// Responsibilities in this file right now:
//   - Feature-detect requestIdleCallback. If missing, show the unsupported-
//     browser banner and disable Run / Run all 3 buttons. Phase 4 relies on
//     rIC for low-priority paint + training cadence, and we'd rather fail
//     loudly than silently misbehave on Safari.
//   - Nothing else. Do NOT wire up the frame loop or schedulers here — the
//     next commit owns that responsibility, and splitting keeps commit 3's
//     blast radius tiny (HTML + CSS skeleton plus one feature check).

function detectSupport() {
  return typeof window.requestIdleCallback === "function";
}

function showUnsupportedWarning() {
  const banner = document.getElementById("unsupported-browser");
  if (banner) banner.hidden = false;
  for (const id of ["run-btn", "run-all-btn"]) {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (!detectSupport()) {
    showUnsupportedWarning();
    return;
  }
  // Intentionally empty. Commit 4 fills in the frame loop.
});
