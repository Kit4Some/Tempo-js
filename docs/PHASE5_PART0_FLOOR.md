# Phase 5 Part 0 — Headless Floor

_Generated: 2026-04-18T12:22:02.077Z · Node v22.18.0 · win32 x64_

60 s run, workload `constant`, active `B0`, seed 42.
B0 never reduces or degrades, so any jank observed is pure harness
noise — scheduler behaviour cannot contribute. This is the floor.

## Chrome flags

- `--disable-background-timer-throttling`
- `--disable-renderer-backgrounding`
- `--disable-backgrounding-occluded-windows`

## Measurement

| Window | Frames | Jank rate | P95 (ms) | Mean (ms) |
|---|---:|---:|---:|---:|
| all | 1024 | 0.00% | 5.10 | 5.03 |
| recent (300) | 300 | 0.00% | 5.10 | 5.03 |

## Gate decision

- Live floor reference: 10% (Phase 4 verification).
- Headless floor (this run): **0.00%**
- Outcome: **PROCEED**
- Reason: Headless floor 0.00% < 3%. Go/No-Go threshold becomes relatively more discriminating.

## Browser console errors

- `Failed to load resource: the server responded with a status of 404 (Not Found)`
