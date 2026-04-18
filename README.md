# Tempo

A 353-parameter online-learning MLP that predicts browser frame-budget
overruns. Research artifact — not an npm library, not a production tool.

See [TEMPO_CLAUDE_CODE_SPEC.md](TEMPO_CLAUDE_CODE_SPEC.md) for the full scope,
methodology, and phase plan.

## Development

```bash
npm install         # one-time
npm run dev         # live benchmark at http://localhost:5173/
npm test            # full vitest suite (155 tests)
npm run build       # production build to dist/
npm run preview     # preview the production build locally
```

The live benchmark page (`benchmark/index.html`) is the project's sole web
entry. There is no separate dev harness — the benchmark page itself is
what runs in development, production, and on GitHub Pages.

## Browser support

- **Chrome 47+**, **Firefox 55+**, **Edge 79+** — fully supported.
- **Safari** — not supported. The benchmark relies on `requestIdleCallback`
  for training cadence and paint deferral; polyfilling it via `setTimeout`
  would distort training semantics. The page shows an explicit unsupported
  banner on Safari rather than silently misbehave.

## Benchmark protocol divergence

The live demo uses shorter runs (30 s per scheduler in Run-all-3) for UX.
The Phase 5 headless benchmark uses 60 s per scheduler per the §5
methodology — the statistical power required for Mann-Whitney U differs
from the "don't freeze the tab" UX budget.
