import {
  WL_BURST_BASE_MS,
  WL_BURST_SPIKE_DURATION,
  WL_BURST_SPIKE_EVERY,
  WL_BURST_SPIKE_MS,
  WL_CONSTANT_MS,
  WL_SAWTOOTH_MAX,
  WL_SAWTOOTH_MIN,
  WL_SAWTOOTH_PERIOD,
  WL_SCROLL_K,
} from "../core/constants.js";

function busyWait(ms) {
  if (ms <= 0) return;
  const start = performance.now();
  while (performance.now() - start < ms) {
    /* burn CPU */
  }
}

export function constant(ms = WL_CONSTANT_MS) {
  return function constantLoad(_frameIdx) {
    busyWait(ms);
  };
}

export function sawtooth(
  min = WL_SAWTOOTH_MIN,
  max = WL_SAWTOOTH_MAX,
  periodFrames = WL_SAWTOOTH_PERIOD,
) {
  const span = max - min;
  return function sawtoothLoad(frameIdx) {
    const phase = ((frameIdx % periodFrames) + periodFrames) % periodFrames;
    const ms = min + span * (phase / periodFrames);
    busyWait(ms);
  };
}

export function burst(
  baseMs = WL_BURST_BASE_MS,
  spikeMs = WL_BURST_SPIKE_MS,
  spikeEveryFrames = WL_BURST_SPIKE_EVERY,
  spikeDurationFrames = WL_BURST_SPIKE_DURATION,
) {
  return function burstLoad(frameIdx) {
    const inSpike =
      frameIdx >= spikeEveryFrames &&
      ((frameIdx % spikeEveryFrames) + spikeEveryFrames) % spikeEveryFrames <
        spikeDurationFrames;
    busyWait(inSpike ? spikeMs : baseMs);
  };
}

export function scrollCorrelated(scrollVelocityFn, k = WL_SCROLL_K) {
  return function scrollCorrelatedLoad(_frameIdx) {
    const v = Math.abs(scrollVelocityFn());
    busyWait(v * k);
  };
}
