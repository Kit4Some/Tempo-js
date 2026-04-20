# Cohen's d validation

Purpose: confirm that the extraordinarily large effect sizes reported in
[PHASE5_PART2_COMPARE.md](PHASE5_PART2_COMPARE.md) (|d| up to 649) are a
correct consequence of this benchmark's controlled-environment minimal
variance, not a formula or unit bug in `src/harness/stats.js`.

## Method

Re-compute Cohen's d by hand from the raw `jankRate` values in the
committed JSONL files:

- `docs/PHASE5_PART1_RESULTS.jsonl` (120 runs)
- `docs/PHASE5_PART2_RESULTS.jsonl` (88 runs)

Formula (Cohen 1988, standard pooled-SD form):

```
pooled_sd = sqrt(((n_A - 1) * var_A + (n_B - 1) * var_B) / (n_A + n_B - 2))
d         = (mean_A - mean_B) / pooled_sd
```

where `var_*` is the sample variance (denominator `n - 1`). Implementation
in `src/harness/stats.js::cohensD` uses this exact form.

## Raw computations

All three pre-registered comparison tables. Cell key: `A vs B`.

### Table (a) — Scratch (Part 1) vs Pretrained + Online (Part 2)

| Workload | n_A | mean_A | sd_A | n_B | mean_B | sd_B | pooled_sd | d_manual | d_analyze | match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sawtooth | 10 | 6.623 % | 0.2786 pp | 10 | 4.593 % | 0.1229 pp | 0.2153 pp | +9.4281 | +9.4281 | ✅ |
| burst | 10 | 5.455 % | 0.0213 pp | 10 | 5.515 % | 0.0487 pp | 0.0376 pp | −1.6064 | −1.6064 | ✅ |
| scroll | 10 | 7.076 % | 0.1860 pp | 10 | 6.784 % | 0.2919 pp | 0.2447 pp | +1.1943 | +1.1943 | ✅ |

### Table (b) — Pretrained + Online vs Pretrained + Frozen

| Workload | n_A | mean_A | sd_A | n_B | mean_B | sd_B | pooled_sd | d_manual | d_analyze | match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sawtooth | 10 | 4.593 % | 0.1229 pp | 10 | 11.679 % | 0.0224 pp | 0.0883 pp | −80.2267 | −80.2267 | ✅ |
| burst | 10 | 5.515 % | 0.0487 pp | 10 | 5.523 % | 0.0388 pp | 0.0440 pp | −0.1724 | −0.1724 | ✅ |
| scroll | 10 | 6.784 % | 0.2919 pp | 10 | 14.606 % | 0.1431 pp | 0.2299 pp | −34.0306 | −34.0306 | ✅ |

### Table (c) — B1 vs Pretrained + Frozen (blog headline)

| Workload | n_A | mean_A | sd_A | n_B | mean_B | sd_B | pooled_sd | d_manual | d_analyze | match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sawtooth | 12 | 1.664 % | 0.0048 pp | 10 | 11.679 % | 0.0224 pp | 0.0154 pp | −649.3136 | −649.3136 | ✅ |
| burst | 12 | 5.559 % | 0.0009 pp | 10 | 5.523 % | 0.0388 pp | 0.0260 pp | +1.4149 | +1.4149 | ✅ |
| scroll | 12 | 3.383 % | 0.0277 pp | 10 | 14.606 % | 0.1431 pp | 0.0981 pp | −114.3681 | −114.3681 | ✅ |

## Verdict

**Scenario (a) — computation is correct; standard deviations are
legitimately tiny.**

Every manual value matches the corresponding `analyze.js` output bit-for-
bit. The extraordinary effect sizes follow directly from standard
deviations that are orders of magnitude smaller than what Cohen's
"small / medium / large" conventions (d = 0.2 / 0.5 / 0.8) were
calibrated against. That calibration came from naturally-noisy
human-subjects research; this benchmark is a controlled synthetic
environment.

Why variance is minimal:

1. **B1 is fully deterministic.** Seed = 42 + identical `dt` sequences
   through an EMA threshold produces byte-identical decisions every run.
   Its residual SD (~0.005 pp) is pure Float32 quantization of the
   summary metrics — no real variance.
2. **Pretrained + Frozen is also fully deterministic.** No weight
   updates during measurement, so the Predictor is a pure function of
   its (frozen) 353 parameters and the feature stream. Same seed → same
   decisions → same metrics modulo float noise.
3. **Headless no-vsync regime.** `scripts/benchmark.js` runs with
   `--disable-background-timer-throttling` and
   `--disable-backgrounding-occluded-windows`; the rAF loop is
   dispatched by CPU work completion, not by a 60 Hz refresh signal.
   This removes wall-clock jitter that would otherwise show up as
   variance.

When both cells in a comparison are deterministic (as in table (c)'s
sawtooth cell — B1 and Pretrained+Frozen), pooled SD collapses to
order 1e-4, and any real mean shift produces a `d` of comparable
magnitude. `d = −649` on sawtooth's (c) cell decomposes to
`|Δμ| = 10.01 pp` ÷ `pooled_sd = 0.015 pp` — both numerators and
denominators are correct.

## Practical interpretation

Under these measurement conditions, Cohen's d primarily encodes "this
mean shift exceeds the measurement noise floor." At a 0.015 pp noise
floor, that is a very weak statement. The blog post and RESULTS.md
should read effect direction + absolute percentage-point shift as the
practical magnitude, and treat `d` only as a "not-noise" qualifier.

Accordingly, RESULTS.md's Part 2 section now reports `|Δ|` in pp
alongside every `d`. The existing d values stay — they are correct —
but the accompanying prose directs readers to the pp figure for
practical interpretation, with a one-paragraph note explaining why
d inflates in controlled measurement regimes.

## Action

No code change required. This document is the record that the d values
were validated and the inflated magnitudes are a property of the
measurement environment, not a bug.
