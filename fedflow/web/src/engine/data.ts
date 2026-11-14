// Loading and decoding the assets exported by ml/run_experiments.py
export const STEPS_PER_DAY = 288
export const LAGS = 12
export const HORIZONS = [1, 3, 6, 12] as const // ×5 min
export const HORIZON_LABELS = ['5 min', '15 min', '30 min', '60 min']
export const FIRST_WEEKDAY = 4 // 1 Jul 2016 = Friday (Mon = 0)

export type Sensor = { id: number; x: number; y: number; zone: number; p95: number }
export type Network = {
  sensors: Sensor[]
  zones: { zone: number; sensors: number[]; cx: number; cy: number }[]
  edges: [number, number][]
  start: string
  test_start_step: number
}
export type Series = { t0: number; steps: number; rows: Float32Array[] } // rows[i] = flow of sensor i
export type TrainSubset = Series & { sensors: number[]; zones: number[] }
export type Weights = Record<string, { shape: number[]; data: number[] }>
export type GRUFile = {
  model: string
  hidden: number
  federated: { norm: [number, number]; weights: Weights }
  local: Record<string, { norm: [number, number]; weights: Weights }>
}

function decodeU16(b64: string, rows: number, cols: number): Float32Array[] {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  const u16 = new Uint16Array(bytes.buffer)
  const out: Float32Array[] = []
  for (let r = 0; r < rows; r++) out.push(Float32Array.from(u16.subarray(r * cols, (r + 1) * cols)))
  return out
}

async function json<T>(name: string): Promise<T> {
  const res = await fetch(`./data/${name}`)
  if (!res.ok) throw new Error(`Failed to load ${name}`)
  return res.json() as Promise<T>
}

export type Assets = {
  network: Network
  test: Series
  train: TrainSubset
  gru: GRUFile
  results: Results
}

export async function loadAssets(): Promise<Assets> {
  const [network, testRaw, trainRaw, gru, results] = await Promise.all([
    json<Network>('network.json'),
    json<{ t0: number; steps: number; sensors: number; flow_u16: string }>('test_flow.json'),
    json<{ t0: number; steps: number; sensors: number[]; zones: number[]; flow_u16: string }>('train_subset.json'),
    json<GRUFile>('gru_weights.json'),
    json<Results>('results.json'),
  ])
  return {
    network,
    gru,
    results,
    test: { t0: testRaw.t0, steps: testRaw.steps, rows: decodeU16(testRaw.flow_u16, testRaw.sensors, testRaw.steps) },
    train: {
      t0: trainRaw.t0,
      steps: trainRaw.steps,
      sensors: trainRaw.sensors,
      zones: trainRaw.zones,
      rows: decodeU16(trainRaw.flow_u16, trainRaw.sensors.length, trainRaw.steps),
    },
  }
}

/** absolute step index -> time features of that step */
export function timeFeats(step: number) {
  const tod = (step % STEPS_PER_DAY) / STEPS_PER_DAY
  const dow = (FIRST_WEEKDAY + Math.floor(step / STEPS_PER_DAY)) % 7
  return { tod, weekend: dow >= 5 ? 1 : 0, dow }
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
export function stepLabel(step: number, withDay = true) {
  const d = new Date(Date.UTC(2016, 6, 1) + Math.floor(step / STEPS_PER_DAY) * 86400000)
  const m = (step % STEPS_PER_DAY) * 5
  const hh = String(Math.floor(m / 60)).padStart(2, '0')
  const mm = String(m % 60).padStart(2, '0')
  const { dow } = timeFeats(step)
  return withDay
    ? `${DAYS[dow]} ${d.getUTCDate()} ${d.toLocaleString('en', { month: 'short', timeZone: 'UTC' })} · ${hh}:${mm}`
    : `${hh}:${mm}`
}

// ---------------------------------------------------------------- results.json typing
export type Metric = { mae: number; rmse: number; mape: number }
export type MetricSet = Record<string, Metric> & { avg: Metric }
export type Results = {
  dataset: { name: string; sensors: number; days: number; interval_min: number; train_days: number; val_days: number; test_days: number; zones: number; zone_sizes: number[] }
  horizons_min: number[]
  runs: Record<string, { model: string; metrics: MetricSet; private: boolean; rounds?: number; bytes_up?: number }>
  curves: Record<string, { round: number; val_mae: number; loss: number }[]>
  per_zone: Record<string, { zone: number; sensors: number; samples: number; local_mae: number; fed_mae: number; gain_pct: number }[]>
  privacy: { sigma: number; epsilon: number | null; delta: number | null; mae: number; note?: string }[]
  privacy_clip: number
  dropout: { dropout: number; mae: number; avg_clients: number }[]
  communication: { params_gru: number; params_mlp: number; model_bytes: number; total_upload_bytes: number; raw_data_bytes: number; rounds: number }
  runtime_sec: number
  scarce?: { days: number; local_mae: number; fed_mae: number; gain_pct: number; zones: { zone: number; local_mae: number; fed_mae: number; gain_pct: number }[] }[]
  unseen_zone?: { zone: number; sensors: number; fed_unseen_mae: number; local_7day_mae: number }[]
  privacy_scale?: { clients: number; sigma: number; epsilon: number | null; mae: number }[]
}
