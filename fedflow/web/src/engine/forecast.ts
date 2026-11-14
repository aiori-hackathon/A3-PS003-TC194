import { LAGS, timeFeats, type Series } from './data'
import { gruPredict, type GRU } from './models'

const weekendOf = (s: number) => timeFeats(s).weekend

/** Forecast made at local test index `t` (the 12 lags are t-12 … t-1).
 *  Returns flows for [t, t+2, t+5, t+11] (= +5, +15, +30, +60 min). */
export function forecastAt(model: GRU, test: Series, sensor: number, t: number) {
  const row = test.rows[sensor]
  if (t < LAGS) return null
  return gruPredict(model, row.subarray(t - LAGS, t), test.t0 + t - 1, weekendOf)
}

/** Series of h-step-ahead forecasts aligned with the target index, for t in [from, to). */
export function forecastSeries(model: GRU, test: Series, sensor: number, from: number, to: number, hIdx: number, hSteps: number) {
  const out: (number | null)[] = []
  for (let t = from; t < to; t++) {
    const made = t - (hSteps - 1)
    const f = made >= LAGS ? forecastAt(model, test, sensor, made) : null
    out.push(f ? f[hIdx] : null)
  }
  return out
}
