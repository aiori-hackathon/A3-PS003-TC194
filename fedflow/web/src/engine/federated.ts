// In-browser federated learning simulator: real training of the TemporalMLP on real PeMS08
// readings, one client per traffic zone. Mirrors ml/fedflow/fl.py (FedAvg / FedProx, client-level
// DP with clipping + distributed Gaussian noise, pairwise-mask secure aggregation, dropout).
import { HORIZONS, LAGS, STEPS_PER_DAY, timeFeats, type TrainSubset, type Series } from './data'
import {
  Adam, cloneParams, flatten, huber, mlpBackward, mlpForward, mlpInit, mlpRow, rng, unflatten,
  type Params, type RNG,
} from './models'

export type FLSettings = {
  strategy: 'fedavg' | 'fedprox'
  rounds: number
  localSteps: number
  batch: number
  lr: number
  mu: number
  dp: boolean
  clip: number
  sigma: number
  secureAgg: boolean
  dropout: number
  historyDays: number
  seed: number
}

export const DEFAULT_SETTINGS: FLSettings = {
  strategy: 'fedavg', rounds: 25, localSteps: 20, batch: 96, lr: 0.003, mu: 0.01,
  dp: false, clip: 1.0, sigma: 0.5, secureAgg: true, dropout: 0, historyDays: 14, seed: 7,
}

export type ClientState = {
  zone: number
  sensors: number[]
  samples: number
  status: 'idle' | 'training' | 'uploading' | 'dropped' | 'done'
  loss: number | null
  norm: number | null
  clipped: boolean
  maskedPreview: number[] | null
  truePreview: number[] | null
}

export type RoundLog = {
  round: number
  fedMae: number
  localMae: number
  perHorizon: number[]
  participants: number
  bytesUp: number
  epsilon: number | null
  serverView: number[] | null
}

/** epsilon for `rounds` Gaussian-mechanism compositions (RDP accountant, δ = 1e-5) */
export function rdpEpsilon(sigma: number, rounds: number, delta = 1e-5) {
  if (sigma <= 0) return Infinity
  let best = Infinity
  for (let a = 1.01; a < 512; a += a < 10 ? 0.02 : 1) {
    const e = (rounds * a) / (2 * sigma * sigma) + Math.log(1 / delta) / (a - 1)
    if (e < best) best = e
  }
  return best
}

type Window = { row: number; t: number } // row = index into subset rows, t = first target index (local)

export class FederatedSim {
  s: FLSettings
  data: TrainSubset
  test: Series
  zones: number[]
  clients: ClientState[]
  windows: Window[][]
  testWindows: Window[][]
  global: Params
  locals: Params[]
  localOpts: Adam[]
  mu = 0
  sd = 1
  round = 0
  logs: RoundLog[] = []
  R: RNG
  bytesUp = 0
  dim: number

  constructor(data: TrainSubset, test: Series, settings: FLSettings) {
    this.s = settings
    this.data = data
    this.test = test
    this.R = rng(settings.seed)
    this.zones = [...new Set(data.zones)].sort((a, b) => a - b)
    const H = HORIZONS[HORIZONS.length - 1]
    this.windows = this.zones.map((z) => {
      const w: Window[] = []
      data.zones.forEach((zz, row) => {
        if (zz !== z) return
        const start = Math.max(LAGS, data.steps - H - settings.historyDays * STEPS_PER_DAY)
        for (let t = start; t <= data.steps - H; t++) w.push({ row, t })
      })
      return w
    })
    // evaluation windows on the test period (only for the subset sensors)
    this.testWindows = this.zones.map((z) => {
      const w: Window[] = []
      data.zones.forEach((zz, row) => {
        if (zz !== z) return
        for (let t = LAGS; t <= test.steps - H; t += 6) w.push({ row, t })
      })
      return w
    })
    this.clients = this.zones.map((z, i) => ({
      zone: z,
      sensors: data.sensors.filter((_, r) => data.zones[r] === z),
      samples: this.windows[i].length,
      status: 'idle', loss: null, norm: null, clipped: false, maskedPreview: null, truePreview: null,
    }))
    // federated normalisation: clients share only sums
    let S = 0, Q = 0, C = 0
    for (const row of data.rows) for (const v of row) { S += v; Q += v * v; C++ }
    this.mu = S / C
    this.sd = Math.sqrt(Q / C - this.mu * this.mu)
    this.global = mlpInit(settings.seed)
    this.dim = flatten(this.global).length
    this.locals = this.zones.map(() => cloneParams(this.global))
    this.localOpts = this.locals.map((p) => new Adam(p, settings.lr))
  }

  get rawBytes() {
    return this.data.rows.length * this.data.steps * 3 * 4 * 3 // ~3 features, float32, x3 for full 43-day history
  }

  private sensorRow(row: number) {
    return this.data.rows[row]
  }

  private makeBatch(wins: Window[], idx: number[], src: 'train' | 'test') {
    const B = idx.length
    const x = new Float64Array(B * (LAGS + 3))
    const y = new Float64Array(B * HORIZONS.length)
    const lag = new Float64Array(LAGS)
    const t0 = src === 'train' ? this.data.t0 : this.test.t0
    for (let b = 0; b < B; b++) {
      const w = wins[idx[b]]
      const series = src === 'train' ? this.sensorRow(w.row) : this.test.rows[this.data.sensors[w.row]]
      for (let k = 0; k < LAGS; k++) lag[k] = (series[w.t - LAGS + k] - this.mu) / this.sd
      const { tod, weekend } = timeFeats(t0 + w.t - 1)
      mlpRow(x, b * (LAGS + 3), lag, tod, weekend)
      HORIZONS.forEach((h, j) => { y[b * HORIZONS.length + j] = (series[w.t + h - 1] - this.mu) / this.sd })
    }
    return { x, y, B }
  }

  private localTrain(start: Params, ci: number, opt?: Adam) {
    const p = cloneParams(start)
    const o = opt ?? new Adam(p, this.s.lr)
    if (opt) o.lr = this.s.lr
    const wins = this.windows[ci]
    let lossSum = 0
    for (let step = 0; step < this.s.localSteps; step++) {
      const idx = Array.from({ length: this.s.batch }, () => this.R.int(wins.length))
      const { x, y, B } = this.makeBatch(wins, idx, 'train')
      const { y: pred, cache } = mlpForward(p, x, B)
      const { loss, grad } = huber(pred, y)
      const g = mlpBackward(p, cache, grad)
      if (this.s.strategy === 'fedprox') {
        for (const k in g) for (let i = 0; i < g[k].d.length; i++) g[k].d[i] += this.s.mu * (p[k].d[i] - start[k].d[i])
      }
      if (opt) {
        // local-only baseline keeps its own optimiser and parameters across rounds
        o.step(p, g)
      } else o.step(p, g)
      lossSum += loss
    }
    return { p, loss: lossSum / this.s.localSteps }
  }

  /** MAE in vehicles / 5 min over the test period for one model on one zone (or all zones) */
  evaluate(p: Params, zoneIdx?: number) {
    const zs = zoneIdx === undefined ? this.zones.map((_, i) => i) : [zoneIdx]
    const perH = new Array(HORIZONS.length).fill(0)
    let n = 0
    for (const zi of zs) {
      const wins = this.testWindows[zi]
      for (let i = 0; i < wins.length; i += 256) {
        const idx = Array.from({ length: Math.min(256, wins.length - i) }, (_, k) => i + k)
        const { x, y, B } = this.makeBatch(wins, idx, 'test')
        const { y: pred } = mlpForward(p, x, B)
        for (let b = 0; b < B; b++) for (let j = 0; j < HORIZONS.length; j++) {
          const pv = Math.max(0, pred[b * 4 + j] * this.sd + this.mu)
          const tv = y[b * 4 + j] * this.sd + this.mu
          perH[j] += Math.abs(pv - tv)
        }
        n += B
      }
    }
    const ph = perH.map((v) => v / n)
    return { mae: ph.reduce((a, b) => a + b, 0) / ph.length, perHorizon: ph }
  }

  /** One full federated round. `onPhase` lets the UI animate the protocol. */
  async step(onPhase: (phase: 'broadcast' | 'train' | 'upload' | 'aggregate', ci?: number) => Promise<void> | void) {
    if (this.round >= this.s.rounds) return null
    this.round++
    const r = this.round
    await onPhase('broadcast')
    const gvec = flatten(this.global)
    const alive: number[] = []
    this.clients.forEach((c, i) => {
      if (this.R.next() < this.s.dropout) c.status = 'dropped'
      else { c.status = 'training'; alive.push(i) }
    })
    if (alive.length === 0) { const i = this.R.int(this.clients.length); alive.push(i); this.clients[i].status = 'training' }
    const updates = new Map<number, Float64Array>()
    for (const ci of alive) {
      const { p, loss } = this.localTrain(this.global, ci)
      const v = flatten(p)
      const delta = new Float64Array(v.length)
      let nrm = 0
      for (let i = 0; i < v.length; i++) { delta[i] = v[i] - gvec[i]; nrm += delta[i] * delta[i] }
      nrm = Math.sqrt(nrm)
      const c = this.clients[ci]
      c.loss = loss; c.norm = nrm; c.clipped = false
      if (this.s.dp && this.s.clip > 0 && nrm > this.s.clip) {
        const f = this.s.clip / nrm
        for (let i = 0; i < delta.length; i++) delta[i] *= f
        c.clipped = true
      }
      if (this.s.dp) {
        const std = (this.s.sigma * this.s.clip) / Math.sqrt(alive.length)
        for (let i = 0; i < delta.length; i++) delta[i] += this.R.normal() * std
      }
      updates.set(ci, delta)
      c.status = 'uploading'
      await onPhase('train', ci)
    }
    // weights: proportional to samples (uniform under DP to bound sensitivity)
    const total = alive.reduce((s, ci) => s + (this.s.dp ? 1 : this.clients[ci].samples), 0)
    const contrib = new Map<number, Float64Array>()
    for (const ci of alive) {
      const w = (this.s.dp ? 1 : this.clients[ci].samples) / total
      contrib.set(ci, updates.get(ci)!.map((v) => v * w))
    }
    // secure aggregation: pairwise masks that cancel in the sum
    const agg = new Float64Array(this.dim)
    if (this.s.secureAgg) {
      const masks = new Map<number, Float64Array>(alive.map((ci) => [ci, new Float64Array(this.dim)]))
      for (let a = 0; a < alive.length; a++) for (let b = a + 1; b < alive.length; b++) {
        const pr = rng(r * 7919 + alive[a] * 131 + alive[b])
        const ma = masks.get(alive[a])!, mb = masks.get(alive[b])!
        for (let i = 0; i < this.dim; i++) { const m = pr.normal() * 10; ma[i] += m; mb[i] -= m }
      }
      for (const ci of alive) {
        const masked = contrib.get(ci)!.map((v, i) => v + masks.get(ci)![i])
        this.clients[ci].maskedPreview = Array.from(masked.slice(0, 6))
        this.clients[ci].truePreview = Array.from(contrib.get(ci)!.slice(0, 6))
        for (let i = 0; i < this.dim; i++) agg[i] += masked[i]
      }
    } else {
      for (const ci of alive) {
        this.clients[ci].maskedPreview = null
        this.clients[ci].truePreview = Array.from(contrib.get(ci)!.slice(0, 6))
        const c = contrib.get(ci)!
        for (let i = 0; i < this.dim; i++) agg[i] += c[i]
      }
    }
    await onPhase('upload')
    this.bytesUp += alive.length * this.dim * 4
    for (let i = 0; i < this.dim; i++) gvec[i] += agg[i]
    this.global = unflatten(gvec, this.global)
    // local-only baseline: every zone keeps training alone for the same number of steps
    this.zones.forEach((_, ci) => {
      const res = this.localTrain(this.locals[ci], ci, this.localOpts[ci])
      this.locals[ci] = res.p
    })
    await onPhase('aggregate')
    const fed = this.evaluate(this.global)
    let localMae = 0, n = 0
    this.zones.forEach((_, ci) => { const w = this.testWindows[ci].length; localMae += this.evaluate(this.locals[ci], ci).mae * w; n += w })
    localMae /= n
    this.clients.forEach((c) => { if (c.status !== 'dropped') c.status = 'done' })
    const log: RoundLog = {
      round: r,
      fedMae: fed.mae,
      localMae,
      perHorizon: fed.perHorizon,
      participants: alive.length,
      bytesUp: this.bytesUp,
      epsilon: this.s.dp ? rdpEpsilon(this.s.sigma, r) : null,
      serverView: Array.from(agg.slice(0, 6)),
    }
    this.logs.push(log)
    return log
  }

  /** predictions of the current global model for one subset sensor over a test-time window */
  predictSeries(row: number, from: number, to: number, horizon: number) {
    const series = this.test.rows[this.data.sensors[row]]
    const out: number[] = []
    const x = new Float64Array(LAGS + 3)
    const lag = new Float64Array(LAGS)
    for (let t = from; t < to; t++) {
      const last = t - horizon
      if (last - LAGS + 1 < 0) { out.push(NaN); continue }
      for (let k = 0; k < LAGS; k++) lag[k] = (series[last - LAGS + 1 + k] - this.mu) / this.sd
      const { tod, weekend } = timeFeats(this.test.t0 + last)
      mlpRow(x, 0, lag, tod, weekend)
      const { y } = mlpForward(this.global, x, 1)
      out.push(Math.max(0, y[HORIZONS.indexOf(horizon as 1)] * this.sd + this.mu))
    }
    return out
  }
}

export const DAY = STEPS_PER_DAY
