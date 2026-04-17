function busyWait(ms) {
  if (ms <= 0) return;
  const start = performance.now();
  while (performance.now() - start < ms) {
    /* burn CPU */
  }
}

export function constant(ms = 5) {
  return function constantLoad(_frameIdx) {
    busyWait(ms);
  };
}

export function sawtooth(min = 0, max = 20, periodFrames = 60) {
  const span = max - min;
  return function sawtoothLoad(frameIdx) {
    const phase = ((frameIdx % periodFrames) + periodFrames) % periodFrames;
    const ms = min + span * (phase / periodFrames);
    busyWait(ms);
  };
}

export function burst(
  baseMs = 3,
  spikeMs = 30,
  spikeEveryFrames = 90,
  spikeDurationFrames = 5,
) {
  return function burstLoad(frameIdx) {
    const inSpike =
      frameIdx >= spikeEveryFrames &&
      ((frameIdx % spikeEveryFrames) + spikeEveryFrames) % spikeEveryFrames <
        spikeDurationFrames;
    busyWait(inSpike ? spikeMs : baseMs);
  };
}

export function scrollCorrelated(scrollVelocityFn, k = 0.3) {
  return function scrollCorrelatedLoad(_frameIdx) {
    const v = Math.abs(scrollVelocityFn());
    busyWait(v * k);
  };
}
