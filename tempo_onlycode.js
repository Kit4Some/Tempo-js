'use strict';

const BUDGET   = 16.67;
const JANK_TOL = 1.0;
const COST     = { full: 1.0, reduce: 0.7, degrade: 0.35 };

const B1_ALPHA = 0.3, B1_REDUCE = 0.8, B1_DEGRADE = 1.2;
const PRED_REDUCE = 0.1, PRED_DEGRADE = 0.3;

const D_IN = 12, D_H1 = 16, D_H2 = 8, N_PARAMS = 353;
const W1_OFF = 0, BIAS1 = 192, W2_OFF = 208, BIAS2 = 336, W3_OFF = 344, BIAS3 = 352;

const LR = 1e-3, MOMENTUM = 0.9, GRAD_CLIP = 1.0, BATCH = 16, BUF = 1024;
const EMA_FAST = 0.3, EMA_SLOW = 0.05, WIN_SHORT = 8, WIN_MISS = 32;
const LOSS_EPS = 1e-7;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

const WORKLOADS = {
  sawtooth: (i) => ((i % 60) / 60) * 20,
  burst:    (i) => (i >= 90 && (i % 90) < 5 ? 30 : 3),
  scroll:   (i) => {
    const t = i + 1;
    return 60 * Math.abs(Math.sin(Math.PI * ((t % 120) / 120))) * 0.3;
  },
  constant: (_) => 5,
};

class Features {
  constructor() {
    this.emaFast = 0; this.emaSlow = 0;
    this.buf = new Float32Array(WIN_MISS);
    this.idx = 0; this.n = 0;
    this.out = new Float32Array(D_IN);
  }
  observe(dt) {
    const norm = dt / BUDGET;
    this.emaFast = EMA_FAST * norm + (1 - EMA_FAST) * this.emaFast;
    this.emaSlow = EMA_SLOW * norm + (1 - EMA_SLOW) * this.emaSlow;
    this.buf[this.idx] = dt;
    this.idx = (this.idx + 1) % WIN_MISS;
    if (this.n < WIN_MISS) this.n++;
  }
  extract() {
    const o = this.out;
    o[0] = this.emaFast; o[1] = this.emaSlow;
    const w = Math.min(this.n, WIN_SHORT);
    let sum = 0;
    for (let i = 0; i < w; i++) sum += this.buf[(this.idx - 1 - i + WIN_MISS) % WIN_MISS];
    const m = w > 0 ? sum / w : 0;
    let v = 0, mx = 0;
    for (let i = 0; i < w; i++) {
      const d = this.buf[(this.idx - 1 - i + WIN_MISS) % WIN_MISS];
      v += (d - m) * (d - m);
      if (d > mx) mx = d;
    }
    o[2] = w > 0 ? v / w / (BUDGET * BUDGET) : 0;
    o[3] = w > 0 ? mx / BUDGET : 0;
    let miss = 0;
    for (let i = 0; i < this.n; i++) if (this.buf[i] > BUDGET + JANK_TOL) miss++;
    o[4] = this.n > 0 ? miss / this.n : 0;
    o[5] = o[6] = o[7] = o[8] = o[9] = o[10] = 0;
    o[11] = 1;
    return o;
  }
}

class MLP {
  constructor(rng) {
    this.w  = new Float32Array(N_PARAMS);
    this.z1 = new Float64Array(D_H1); this.a1 = new Float64Array(D_H1);
    this.z2 = new Float64Array(D_H2); this.a2 = new Float64Array(D_H2);
    this.g  = new Float32Array(N_PARAMS);
    this.dz1 = new Float64Array(D_H1); this.dz2 = new Float64Array(D_H2);
    this.pMiss = 0;
    const s1 = Math.sqrt(2 / D_IN), s2 = Math.sqrt(2 / D_H1), s3 = Math.sqrt(2 / D_H2);
    for (let i = W1_OFF; i < BIAS1;  i++) this.w[i] = gauss(rng) * s1;
    for (let i = W2_OFF; i < BIAS2;  i++) this.w[i] = gauss(rng) * s2;
    for (let i = W3_OFF; i < BIAS3;  i++) this.w[i] = gauss(rng) * s3;
  }
  forward(x) {
    const w = this.w, z1 = this.z1, a1 = this.a1, z2 = this.z2, a2 = this.a2;
    for (let i = 0; i < D_H1; i++) {
      let s = w[BIAS1 + i];
      for (let j = 0; j < D_IN; j++) s += w[W1_OFF + i * D_IN + j] * x[j];
      z1[i] = s; a1[i] = s > 0 ? s : 0;
    }
    for (let i = 0; i < D_H2; i++) {
      let s = w[BIAS2 + i];
      for (let j = 0; j < D_H1; j++) s += w[W2_OFF + i * D_H1 + j] * a1[j];
      z2[i] = s; a2[i] = s > 0 ? s : 0;
    }
    let z3 = w[BIAS3];
    for (let j = 0; j < D_H2; j++) z3 += w[W3_OFF + j] * a2[j];
    this.pMiss = 1 / (1 + Math.exp(-z3));
    return this.pMiss;
  }
  backward(x, y) {
    this.forward(x);
    const w = this.w, z1 = this.z1, a1 = this.a1, z2 = this.z2, a2 = this.a2;
    const g = this.g, dz1 = this.dz1, dz2 = this.dz2;
    const p = this.pMiss;
    const pc = p < LOSS_EPS ? LOSS_EPS : p > 1 - LOSS_EPS ? 1 - LOSS_EPS : p;
    const dz3 = pc - y;
    for (let j = 0; j < D_H2; j++) g[W3_OFF + j] = dz3 * a2[j];
    g[BIAS3] = dz3;
    for (let i = 0; i < D_H2; i++) dz2[i] = z2[i] > 0 ? dz3 * w[W3_OFF + i] : 0;
    for (let i = 0; i < D_H2; i++) {
      const d = dz2[i];
      for (let j = 0; j < D_H1; j++) g[W2_OFF + i * D_H1 + j] = d * a1[j];
      g[BIAS2 + i] = d;
    }
    for (let i = 0; i < D_H1; i++) {
      if (z1[i] > 0) {
        let s = 0;
        for (let k = 0; k < D_H2; k++) s += dz2[k] * w[W2_OFF + k * D_H1 + i];
        dz1[i] = s;
      } else dz1[i] = 0;
    }
    for (let i = 0; i < D_H1; i++) {
      const d = dz1[i];
      for (let j = 0; j < D_IN; j++) g[W1_OFF + i * D_IN + j] = d * x[j];
      g[BIAS1 + i] = d;
    }
    return g;
  }
}

class Trainer {
  constructor(mlp, rng) {
    this.mlp = mlp; this.rng = rng;
    this.x   = new Float32Array(BUF * D_IN); this.y = new Uint8Array(BUF);
    this.wi  = 0; this.count = 0;
    this.acc = new Float32Array(N_PARAMS); this.v = new Float32Array(N_PARAMS);
  }
  push(features, target) {
    this.x.set(features, this.wi * D_IN); this.y[this.wi] = target;
    this.wi = (this.wi + 1) % BUF; this.count++;
  }
  step() {
    const n = Math.min(this.count, BUF);
    if (n === 0) return;
    const B = Math.min(BATCH, n);
    this.acc.fill(0);
    for (let s = 0; s < B; s++) {
      const idx = Math.floor(this.rng() * n);
      const xs  = this.x.subarray(idx * D_IN, (idx + 1) * D_IN);
      const g   = this.mlp.backward(xs, this.y[idx]);
      for (let j = 0; j < N_PARAMS; j++) this.acc[j] += g[j];
    }
    const inv = 1 / B;
    let sq = 0;
    for (let j = 0; j < N_PARAMS; j++) { this.acc[j] *= inv; sq += this.acc[j] * this.acc[j]; }
    const norm = Math.sqrt(sq);
    const scale = norm > GRAD_CLIP ? GRAD_CLIP / norm : 1;
    const w = this.mlp.w, v = this.v;
    for (let j = 0; j < N_PARAMS; j++) {
      v[j]  = MOMENTUM * v[j] - LR * this.acc[j] * scale;
      w[j] += v[j];
    }
  }
}

const B0 = { decide: () => 'full', learn: () => {} };
const B1 = {
  decide: (f) => f[0] > B1_DEGRADE ? 'degrade' : f[0] > B1_REDUCE ? 'reduce' : 'full',
  learn:  () => {},
};
function makePredictor(mlp, trainer) {
  const last = new Float32Array(D_IN);
  return {
    decide: (f) => {
      last.set(f);
      const p = mlp.forward(f);
      return p > PRED_DEGRADE ? 'degrade' : p > PRED_REDUCE ? 'reduce' : 'full';
    },
    learn: (wasMiss) => { trainer.push(last, wasMiss ? 1 : 0); trainer.step(); },
  };
}
function freezePredictor(mlp) {
  return {
    decide: (f) => {
      const p = mlp.forward(f);
      return p > PRED_DEGRADE ? 'degrade' : p > PRED_REDUCE ? 'reduce' : 'full';
    },
    learn: () => {},
  };
}

function runOne(sched, workload, nFrames) {
  const f = new Features();
  let lastBase = 0, lastCost = COST.full, jank = 0, dropped = 0;
  const WARMUP = 30;
  for (let i = 0; i < nFrames; i++) {
    const dt = lastBase * lastCost;
    const wasMiss = dt > BUDGET + JANK_TOL;
    f.observe(dt);
    sched.learn(wasMiss);
    const decision = sched.decide(f.extract());
    if (i >= WARMUP) { if (wasMiss) jank++; } else dropped++;
    lastBase = workload(i);
    lastCost = COST[decision];
  }
  return jank / (nFrames - dropped);
}

function rank(xs) {
  const ord = xs.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(xs.length);
  for (let i = 0; i < xs.length;) {
    let j = i;
    while (j + 1 < xs.length && ord[j + 1][0] === ord[i][0]) j++;
    const avg = (i + j + 2) / 2;
    for (let k = i; k <= j; k++) r[ord[k][1]] = avg;
    i = j + 1;
  }
  return r;
}
function normCdf(z) {
  const [A1, A2, A3, A4, A5, P] =
    [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911];
  const sgn = z < 0 ? -1 : 1, x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + P * x);
  const y = 1 - ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t * Math.exp(-x * x);
  return 0.5 * (1 + sgn * y);
}
function mannWhitneyU(a, b) {
  const n1 = a.length, n2 = b.length, N = n1 + n2;
  const r = rank(a.concat(b));
  let R1 = 0;
  for (let i = 0; i < n1; i++) R1 += r[i];
  const U1 = R1 - n1 * (n1 + 1) / 2;
  const U  = Math.min(U1, n1 * n2 - U1);
  const sorted = [...a, ...b].sort((x, y) => x - y);
  let tc = 0;
  for (let i = 0; i < N;) {
    let j = i;
    while (j + 1 < N && sorted[j + 1] === sorted[i]) j++;
    const t = j - i + 1;
    if (t > 1) tc += t * t * t - t;
    i = j + 1;
  }
  const mu = n1 * n2 / 2;
  const vu = (n1 * n2 / 12) * (N + 1 - tc / (N * (N - 1)));
  if (vu <= 0) return { U, p: 1 };
  const z = Math.max(0, Math.abs(U - mu) - 0.5) / Math.sqrt(vu);
  return { U, p: 2 * (1 - normCdf(z)) };
}
function cohensD(a, b) {
  const mA = a.reduce((s, x) => s + x, 0) / a.length;
  const mB = b.reduce((s, x) => s + x, 0) / b.length;
  const vA = a.reduce((s, x) => s + (x - mA) * (x - mA), 0) / (a.length - 1);
  const vB = b.reduce((s, x) => s + (x - mB) * (x - mB), 0) / (b.length - 1);
  const sp = Math.sqrt(((a.length - 1) * vA + (b.length - 1) * vB) / (a.length + b.length - 2));
  if (sp === 0) return mA === mB ? 0 : Math.sign(mA - mB) * Infinity;
  return (mA - mB) / sp;
}
function bootCI(xs, rng, B = 1000) {
  const n = xs.length, means = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += xs[Math.floor(rng() * n)];
    means[b] = s / n;
  }
  const srt = Array.from(means).sort((x, y) => x - y);
  return [srt[Math.floor(0.025 * B)], srt[Math.min(B - 1, Math.ceil(0.975 * B) - 1)]];
}

function experimentBenchmark(wlName, reps, baseSeed) {
  const wl = WORKLOADS[wlName];
  const FRAMES = 3600;
  const out = { B0: [], B1: [], Predictor: [] };
  for (let rep = 0; rep < reps; rep++) {
    out.B0.push(runOne(B0, wl, FRAMES));
    out.B1.push(runOne(B1, wl, FRAMES));
    const seed = baseSeed + rep * 7919;
    const rng = mulberry32(seed);
    const mlp = new MLP(rng);
    const trn = new Trainer(mlp, rng);
    out.Predictor.push(runOne(makePredictor(mlp, trn), wl, FRAMES));
  }
  return { out, wlName, reps, FRAMES };
}

function buildDistillDataset() {
  const xs = [], ys = [];
  const f = new Features();
  let lastBase = 0, lastCost = COST.full;
  for (let i = 0; i < 3600; i++) {
    const dt = lastBase * lastCost;
    f.observe(dt);
    const feat = f.extract();
    const d = B1.decide(feat);
    xs.push(new Float32Array(feat));
    ys.push(d === 'full' ? 0 : 1);
    lastBase = WORKLOADS.sawtooth(i);
    lastCost = COST[d];
  }
  return { xs, ys };
}

function experimentDistillation(seed) {
  const rng = mulberry32(seed);
  const { xs, ys } = buildDistillDataset();

  const mlp = new MLP(rng);
  const acc = new Float32Array(N_PARAMS);
  const vel = new Float32Array(N_PARAMS);
  const idx = Array.from({ length: xs.length }, (_, i) => i);
  const BATCH_D = 16, EPOCHS = 100, LR_D = 5e-3, CLIP_D = 5.0;
  let finalLoss = 0;
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    let epochLoss = 0, nBatches = 0;
    for (let b = 0; b < idx.length; b += BATCH_D) {
      const B = Math.min(BATCH_D, idx.length - b);
      acc.fill(0);
      let batchLoss = 0;
      for (let k = 0; k < B; k++) {
        const y = ys[idx[b + k]];
        const g = mlp.backward(xs[idx[b + k]], y);
        for (let j = 0; j < N_PARAMS; j++) acc[j] += g[j];
        const pc = Math.min(1 - LOSS_EPS, Math.max(LOSS_EPS, mlp.pMiss));
        batchLoss += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
      }
      const inv = 1 / B;
      let sq = 0;
      for (let j = 0; j < N_PARAMS; j++) { acc[j] *= inv; sq += acc[j] * acc[j]; }
      const norm = Math.sqrt(sq);
      const scale = norm > CLIP_D ? CLIP_D / norm : 1;
      for (let j = 0; j < N_PARAMS; j++) {
        vel[j] = MOMENTUM * vel[j] - LR_D * acc[j] * scale;
        mlp.w[j] += vel[j];
      }
      epochLoss += batchLoss * inv;
      nBatches++;
    }
    finalLoss = epochLoss / nBatches;
  }

  let trainAgree = 0;
  for (let i = 0; i < xs.length; i++) {
    mlp.forward(xs[i]);
    const hyp = mlp.pMiss > PRED_REDUCE ? 1 : 0;
    if (hyp === ys[i]) trainAgree++;
  }

  const distilled = freezePredictor(mlp);
  const janks = [];
  for (let rep = 0; rep < 10; rep++) {
    janks.push(runOne(distilled, WORKLOADS.sawtooth, 3600));
  }
  const jankMean = janks.reduce((s, x) => s + x, 0) / janks.length;

  return { mlp, jankMean, finalLoss, trainAgreement: trainAgree / xs.length };
}

function experimentOptimizerDivergence(distilledMLP, seed) {
  const rng = mulberry32(seed);
  const mlp = new MLP(rng);
  const thRandom  = new Float32Array(mlp.w);
  const trn = new Trainer(mlp, rng);
  runOne(makePredictor(mlp, trn), WORKLOADS.sawtooth, 3600);
  const thOnline  = new Float32Array(mlp.w);
  const thDistill = new Float32Array(distilledMLP.w);

  const { xs, ys } = buildDistillDataset();
  const lossAt = (weights) => {
    mlp.w.set(weights);
    let total = 0;
    for (let i = 0; i < xs.length; i++) {
      mlp.forward(xs[i]);
      const pc = Math.min(1 - LOSS_EPS, Math.max(LOSS_EPS, mlp.pMiss));
      total += -(ys[i] * Math.log(pc) + (1 - ys[i]) * Math.log(1 - pc));
    }
    return total / xs.length;
  };
  const lossRandom  = lossAt(thRandom);
  const lossOnline  = lossAt(thOnline);
  const lossDistill = lossAt(thDistill);

  let dot = 0, nO = 0, nD = 0, distSq = 0;
  for (let j = 0; j < N_PARAMS; j++) {
    const dO = thOnline[j]  - thRandom[j];
    const dD = thDistill[j] - thRandom[j];
    dot += dO * dD;
    nO  += dO * dO;
    nD  += dD * dD;
    const dOD = thOnline[j] - thDistill[j];
    distSq += dOD * dOD;
  }
  const normOnline  = Math.sqrt(nO);
  const normDistill = Math.sqrt(nD);
  const cosine = dot / (normOnline * normDistill + 1e-12);
  const distOnlineDistill = Math.sqrt(distSq);

  return {
    lossRandom, lossOnline, lossDistill,
    normOnline, normDistill, cosine,
    distOnlineDistill,
    lossRecovered: (lossRandom - lossOnline) / Math.max(lossRandom - lossDistill, 1e-12),
  };
}

function experimentVisualization(distilledMLP, seed) {
  const GRID_X = 48, GRID_Y = 12;
  const X_MAX = 1.5, Y_MAX = 1.5;
  const SYMBOL = { full: '·', reduce: 'o', degrade: '#' };

  const scratchRng = mulberry32(seed);
  const mlpScratch = new MLP(scratchRng);
  const scratchSched = freezePredictor(mlpScratch);

  const onlineRng = mulberry32(seed + 1);
  const mlpOnline = new MLP(onlineRng);
  const onlineTrn = new Trainer(mlpOnline, onlineRng);
  runOne(makePredictor(mlpOnline, onlineTrn), WORKLOADS.sawtooth, 3600);
  const onlineSched = freezePredictor(mlpOnline);

  const distilledSched = freezePredictor(distilledMLP);

  const buf = new Float32Array(D_IN);
  buf[11] = 1;

  function render(scheduler, label) {
    const lines = [`  ${label}`];
    for (let yi = GRID_Y - 1; yi >= 0; yi--) {
      let row = '    ';
      for (let xi = 0; xi < GRID_X; xi++) {
        buf[0] = X_MAX * xi / (GRID_X - 1);
        buf[1] = Y_MAX * yi / (GRID_Y - 1);
        row += SYMBOL[scheduler.decide(buf)];
      }
      lines.push(row);
    }
    return lines.join('\n');
  }
  function agreement(scheduler) {
    let a = 0, t = 0;
    for (let xi = 0; xi < GRID_X; xi++) {
      buf[0] = X_MAX * xi / (GRID_X - 1);
      for (let yi = 0; yi < GRID_Y; yi++) {
        buf[1] = Y_MAX * yi / (GRID_Y - 1);
        if (scheduler.decide(buf) === B1.decide(buf)) a++;
        t++;
      }
    }
    return a / t;
  }

  return {
    panels: [
      render(B1,             'B1  —  3-parameter EMA threshold (reference)'),
      render(scratchSched,   'MLP random init  —  353 params, untrained'),
      render(onlineSched,    'MLP online  —  353 params, 1 sawtooth rep of SGD'),
      render(distilledSched, 'MLP distilled  —  353 params, 30 epochs of B1 supervision'),
    ],
    agreement: {
      scratch:   agreement(scratchSched),
      online:    agreement(onlineSched),
      distilled: agreement(distilledSched),
    },
  };
}
