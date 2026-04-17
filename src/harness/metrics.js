const DEFAULT_RAF =
  typeof requestAnimationFrame !== "undefined" ? requestAnimationFrame : null;
const DEFAULT_CAF =
  typeof cancelAnimationFrame !== "undefined" ? cancelAnimationFrame : null;
const BUFFER_SIZE = 1024;

export class FrameMetrics {
  constructor(
    budgetMs = 16.67,
    { raf = DEFAULT_RAF, caf = DEFAULT_CAF } = {},
  ) {
    this._budget = budgetMs;
    this._raf = raf;
    this._caf = caf;
    this._buf = new Float32Array(BUFFER_SIZE);
    this._frameCount = 0;
    this._prevTs = null;
    this._running = false;
    this._rafId = null;
    this._callbacks = [];
  }

  start() {
    if (this._running) return;
    this._running = true;
    this._prevTs = null;
    this._schedule();
  }

  stop() {
    this._running = false;
    if (this._rafId != null && this._caf) {
      this._caf(this._rafId);
    }
    this._rafId = null;
  }

  reset() {
    this._frameCount = 0;
    this._prevTs = null;
  }

  onFrame(cb) {
    this._callbacks.push(cb);
  }

  getStats() {
    const n = Math.min(this._frameCount, BUFFER_SIZE);
    if (n === 0) {
      return {
        jankRate: 0,
        p95: 0,
        p99: 0,
        meanDt: 0,
        frameCount: 0,
      };
    }
    const snap = new Float32Array(n);
    snap.set(
      this._frameCount < BUFFER_SIZE
        ? this._buf.subarray(0, n)
        : this._buf,
    );
    snap.sort();
    let jank = 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      if (snap[i] > this._budget) jank++;
      sum += snap[i];
    }
    return {
      jankRate: jank / n,
      p95: snap[Math.min(n - 1, Math.floor(n * 0.95))],
      p99: snap[Math.min(n - 1, Math.floor(n * 0.99))],
      meanDt: sum / n,
      frameCount: this._frameCount,
    };
  }

  _schedule() {
    if (!this._raf) return;
    this._rafId = this._raf((ts) => this._tick(ts));
  }

  _tick(ts) {
    if (!this._running) return;
    if (this._prevTs != null) {
      const dt = ts - this._prevTs;
      const frameIdx = this._frameCount;
      this._buf[frameIdx % BUFFER_SIZE] = dt;
      this._frameCount++;
      for (const cb of this._callbacks) cb(frameIdx, dt);
    }
    this._prevTs = ts;
    this._schedule();
  }
}
