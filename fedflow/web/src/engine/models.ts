// Browser implementations of the two forecasters. They mirror ml/fedflow/models.py exactly.
import { LAGS } from './data'

export type Mat = { r: number; c: number; d: Float64Array }
export type Params = Record<string, Mat>

const mat = (r: number, c: number, d?: Float64Array): Mat => ({ r, c, d: d ?? new Float64Array(r * c) })

// ------------------------------------------------------------------ seeded RNG (mulberry32)
export function rng(seed: number) {
  let a = seed >>> 0
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const normal = () => {
    const u = Math.max(next(), 1e-12)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next())
  }
  return { next, normal, int: (n: number) => Math.floor(next() * n) }
}
export type RNG = ReturnType<typeof rng>

// ------------------------------------------------------------------ Temporal MLP (edge model)
export const MLP_SHAPE = [LAGS + 3, 64, 32, 4] as const

export function mlpInit(seed: number): Params {
  const R = rng(seed)
  const glorot = (a: number, b: number) => {
    const lim = Math.sqrt(6 / (a + b))
    const m = mat(a, b)
    for (let i = 0; i < m.d.length; i++) m.d[i] = (R.next() * 2 - 1) * lim
    return m
  }
  const [n0, n1, n2, n3] = MLP_SHAPE
  return { W1: glorot(n0, n1), b1: mat(1, n1), W2: glorot(n1, n2), b2: mat(1, n2), W3: glorot(n2, n3), b3: mat(1, n3) }
}

export function mlpParamCount() {
  const [a, b, c, d] = MLP_SHAPE
  return a * b + b + b * c + c + c * d + d
}

/** x: B×15 row-major. Returns y (B×4) and the cache for backprop. */
function dense(x: Float64Array, B: number, W: Mat, b: Mat, relu: boolean) {
  const out = new Float64Array(B * W.c)
  for (let i = 0; i < B; i++) {
    const xo = i * W.r, oo = i * W.c
    for (let j = 0; j < W.c; j++) out[oo + j] = b.d[j]
    for (let k = 0; k < W.r; k++) {
      const xv = x[xo + k]
      if (xv === 0) continue
      const wo = k * W.c
      for (let j = 0; j < W.c; j++) out[oo + j] += xv * W.d[wo + j]
    }
  }
  if (relu) {
    const z = out.slice()
    for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0
    return { z, a: out }
  }
  return { z: out, a: out }
}

export function mlpForward(p: Params, x: Float64Array, B: number) {
  const l1 = dense(x, B, p.W1, p.b1, true)
  const l2 = dense(l1.a, B, p.W2, p.b2, true)
  const l3 = dense(l2.a, B, p.W3, p.b3, false)
  return { y: l3.a, cache: { x, l1, l2, B } }
}

function denseBack(xin: Float64Array, B: number, W: Mat, dz: Float64Array, gW: Mat, gb: Mat, needDx: boolean) {
  for (let i = 0; i < B; i++) {
    const xo = i * W.r, zo = i * W.c
    for (let j = 0; j < W.c; j++) gb.d[j] += dz[zo + j]
    for (let k = 0; k < W.r; k++) {
      const xv = xin[xo + k]
      if (xv === 0) continue
      const wo = k * W.c
      for (let j = 0; j < W.c; j++) gW.d[wo + j] += xv * dz[zo + j]
    }
  }
  if (!needDx) return null
  const dx = new Float64Array(B * W.r)
  for (let i = 0; i < B; i++) {
    const zo = i * W.c, xo = i * W.r
    for (let k = 0; k < W.r; k++) {
      let s = 0
      const wo = k * W.c
      for (let j = 0; j < W.c; j++) s += W.d[wo + j] * dz[zo + j]
      dx[xo + k] = s
    }
  }
  return dx
}

export function zerosLike(p: Params): Params {
  const g: Params = {}
  for (const k in p) g[k] = mat(p[k].r, p[k].c)
  return g
}

export function mlpBackward(p: Params, cache: ReturnType<typeof mlpForward>['cache'], dy: Float64Array): Params {
  const g = zerosLike(p)
  const { x, l1, l2, B } = cache
  const da2 = denseBack(l2.a, B, p.W3, dy, g.W3, g.b3, true)!
  for (let i = 0; i < da2.length; i++) if (l2.z[i] <= 0) da2[i] = 0
  const da1 = denseBack(l1.a, B, p.W2, da2, g.W2, g.b2, true)!
  for (let i = 0; i < da1.length; i++) if (l1.z[i] <= 0) da1[i] = 0
  denseBack(x, B, p.W1, da1, g.W1, g.b1, false)
  return g
}

/** Builds one MLP input row: 12 normalised lags + sin/cos(tod) + weekend */
export function mlpRow(out: Float64Array, off: number, lagsNorm: ArrayLike<number>, tod: number, weekend: number) {
  for (let k = 0; k < LAGS; k++) out[off + k] = lagsNorm[k]
  out[off + LAGS] = Math.sin(2 * Math.PI * tod)
  out[off + LAGS + 1] = Math.cos(2 * Math.PI * tod)
  out[off + LAGS + 2] = weekend
}

// ------------------------------------------------------------------ Huber loss + Adam
export function huber(pred: Float64Array, target: Float64Array, delta = 1) {
  const g = new Float64Array(pred.length)
  let loss = 0
  for (let i = 0; i < pred.length; i++) {
    const e = pred[i] - target[i]
    const a = Math.abs(e)
    if (a <= delta) { loss += 0.5 * e * e; g[i] = e } else { loss += delta * (a - 0.5 * delta); g[i] = delta * Math.sign(e) }
  }
  for (let i = 0; i < g.length; i++) g[i] /= pred.length
  return { loss: loss / pred.length, grad: g }
}

export class Adam {
  m: Params; v: Params; t = 0
  lr: number; b1: number; b2: number; eps: number
  constructor(p: Params, lr = 3e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps
    this.m = zerosLike(p); this.v = zerosLike(p)
  }
  step(p: Params, g: Params) {
    this.t++
    const c1 = 1 - this.b1 ** this.t, c2 = 1 - this.b2 ** this.t
    for (const k in p) {
      const P = p[k].d, G = g[k].d, M = this.m[k].d, V = this.v[k].d
      for (let i = 0; i < P.length; i++) {
        M[i] = this.b1 * M[i] + (1 - this.b1) * G[i]
        V[i] = this.b2 * V[i] + (1 - this.b2) * G[i] * G[i]
        P[i] -= (this.lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + this.eps)
      }
    }
  }
}

// ------------------------------------------------------------------ flatten helpers
export function flatten(p: Params): Float64Array {
  const keys = Object.keys(p).sort()
  let n = 0
  for (const k of keys) n += p[k].d.length
  const out = new Float64Array(n)
  let i = 0
  for (const k of keys) { out.set(p[k].d, i); i += p[k].d.length }
  return out
}

export function unflatten(v: Float64Array, like: Params): Params {
  const out: Params = {}
  let i = 0
  for (const k of Object.keys(like).sort()) {
    const n = like[k].d.length
    out[k] = mat(like[k].r, like[k].c, v.slice(i, i + n))
    i += n
  }
  return out
}

export function cloneParams(p: Params): Params {
  const o: Params = {}
  for (const k in p) o[k] = mat(p[k].r, p[k].c, p[k].d.slice())
  return o
}

// ------------------------------------------------------------------ GRU (inference only, weights trained in Python)
type W = { shape: number[]; data: number[] }
export type GRU = { h: number; n: number; P: Record<string, Float64Array>; norm: [number, number] }

export function loadGRU(weights: Record<string, W>, norm: [number, number]): GRU {
  const P: Record<string, Float64Array> = {}
  for (const k in weights) P[k] = Float64Array.from(weights[k].data)
  return { h: weights.Uz.shape[0], n: weights.Wz.shape[0], P, norm }
}

const sig = (x: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))))

/** lags: raw flow, 12 values (oldest first). lastStep: absolute step index of the last lag.
 *  Returns raw-unit predictions for the 4 horizons. */
export function gruPredict(m: GRU, lags: ArrayLike<number>, lastStep: number, weekendOf: (s: number) => number): number[] {
  const { h: H, n: N, P, norm } = m
  const [mu, sd] = norm
  let h = new Float64Array(H)
  const z = new Float64Array(H), r = new Float64Array(H), nn = new Float64Array(H)
  const x = new Float64Array(N)
  for (let t = 0; t < LAGS; t++) {
    const step = lastStep - (LAGS - 1) + t
    const tod = (((step % 288) + 288) % 288) / 288
    x[0] = (lags[t] - mu) / sd
    x[1] = Math.sin(2 * Math.PI * tod)
    x[2] = Math.cos(2 * Math.PI * tod)
    x[3] = weekendOf(lastStep)
    for (let j = 0; j < H; j++) {
      let az = P.bz[j], ar = P.br[j]
      for (let i = 0; i < N; i++) { az += x[i] * P.Wz[i * H + j]; ar += x[i] * P.Wr[i * H + j] }
      for (let i = 0; i < H; i++) { az += h[i] * P.Uz[i * H + j]; ar += h[i] * P.Ur[i * H + j] }
      z[j] = sig(az); r[j] = sig(ar)
    }
    for (let j = 0; j < H; j++) {
      let an = P.bn[j]
      for (let i = 0; i < N; i++) an += x[i] * P.Wn[i * H + j]
      for (let i = 0; i < H; i++) an += r[i] * h[i] * P.Un[i * H + j]
      nn[j] = Math.tanh(an)
    }
    const hn = new Float64Array(H)
    for (let j = 0; j < H; j++) hn[j] = (1 - z[j]) * nn[j] + z[j] * h[j]
    h = hn
  }
  const out: number[] = []
  const O = P.bo.length
  for (let o = 0; o < O; o++) {
    let s = P.bo[o]
    for (let i = 0; i < H; i++) s += h[i] * P.Wo[i * O + o]
    out.push(Math.max(0, s * sd + mu))
  }
  return out
}
