import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Assets } from '../engine/data'
import { LAGS, STEPS_PER_DAY, stepLabel } from '../engine/data'
import { loadGRU, type GRU } from '../engine/models'
import { forecastAt } from '../engine/forecast'
import { Badge, Card, Stat, ZONE_COLORS, zoneName } from '../ui'

const APPROACHES = ['North', 'South', 'East', 'West'] as const
const LOST = 10 // s lost time per cycle (2 phases × 5 s)
const SAT_PER_LANE = 1800 // veh/h/lane

type Plan = { C: number; green: [number, number]; delay: number; Y: number }

function webster(q: number[], s: number[]): Plan {
  // phase A = N/S, phase B = E/W ; critical flow ratio per phase.
  // Webster's split (green ∝ critical flow ratio); cycle chosen by minimising HCM control delay on the forecast demand,
  // starting from Webster's closed form C0 = (1.5L + 5)/(1 − Y).
  const y = q.map((v, i) => v / s[i])
  const yA = Math.max(y[0], y[1]), yB = Math.max(y[2], y[3])
  const Y = Math.min(0.92, yA + yB)
  let best: Plan | null = null
  for (let C = 40; C <= 150; C += 5) {
    const gA = ((C - LOST) * yA) / (yA + yB), gB = C - LOST - gA
    const d = avgDelay(q, s, C, [gA, gA, gB, gB])
    if (!best || d < best.delay) best = { C, green: [gA, gB], delay: d, Y }
  }
  return best!
}
function fixed(q: number[], s: number[], C = 120): Plan {
  const g = (C - LOST) / 2
  return { C, green: [g, g], delay: avgDelay(q, s, C, [g, g, g, g]), Y: 0 }
}
/** Average control delay per vehicle (s), flow-weighted over approaches.
 *  HCM 2000 signalised-intersection delay: uniform delay d1 (Webster's first term) +
 *  incremental/oversaturation delay d2 over a 15-min analysis period, valid also when x > 1. */
const T_ANALYSIS = 0.25 // h
function avgDelay(qh: number[], s: number[], C: number, g: number[]) {
  let tot = 0, n = 0
  qh.forEach((qv, i) => {
    const lam = g[i] / C
    const cap = s[i] * lam // veh/h
    const x = qv / cap
    const d1 = (0.5 * C * (1 - lam) ** 2) / (1 - Math.min(1, x) * lam)
    const d2 = 900 * T_ANALYSIS * ((x - 1) + Math.sqrt((x - 1) ** 2 + (8 * 0.5 * x) / (cap * T_ANALYSIS)))
    tot += (d1 + d2) * qv; n += qv
  })
  return tot / n
}

function plansAt(model: GRU, a: Assets, sensors: number[], lanes: number[], t: number) {
  // plan on the FORECAST of the next 15 min, then score both plans on the demand that actually arrived
  const toVph = (v: number) => (v * 12) / 4 // veh/5min -> veh/h; one direction ≈ 1/4 of the detector's lanes
  const q = sensors.map((id) => {
    const f = forecastAt(model, a.test, id, t)
    return toVph(f ? (f[0] + f[1]) / 2 : a.test.rows[id][t - 1])
  })
  const actual = sensors.map((id) => {
    const r = a.test.rows[id]
    return toVph((r[t] + r[t + 1] + r[t + 2]) / 3 || r[t - 1])
  })
  const s = lanes.map((l) => l * SAT_PER_LANE)
  const plan = webster(q, s)
  const g = plan.green
  const adaptive = { ...plan, delay: avgDelay(actual, s, plan.C, [g[0], g[0], g[1], g[1]]) }
  return { q, s, adaptive, fixedPlan: fixed(actual, s) }
}

export default function SignalPlanner({ a }: { a: Assets }) {
  const { network: net, test } = a
  const model = useMemo(() => loadGRU(a.gru.federated.weights, a.gru.federated.norm), [a])
  const [zone, setZone] = useState(0)
  const [day, setDay] = useState(3)
  const [tod, setTod] = useState(8 * 12 + 6) // 08:30
  const sensors = useMemo(() => {
    const z = net.zones[zone].sensors
    return [...z].sort((x, y) => net.sensors[y].p95 - net.sensors[x].p95).slice(0, 4)
  }, [net, zone])
  const lanes = sensors.map((id) => Math.max(2, Math.ceil((net.sensors[id].p95 * 12) / 4 / 1500)))
  const t = LAGS + day * STEPS_PER_DAY + tod
  const cur = plansAt(model, a, sensors, lanes, t)
  const saving = (1 - cur.adaptive.delay / cur.fixedPlan.delay) * 100

  const daily = useMemo(() => {
    const pts = []
    let sumA = 0, sumF = 0
    for (let k = 6 * 12; k < 22 * 12; k += 6) {
      const p = plansAt(model, a, sensors, lanes, LAGS + day * STEPS_PER_DAY + k)
      sumA += p.adaptive.delay * p.q.reduce((x, y) => x + y, 0); sumF += p.fixedPlan.delay * p.q.reduce((x, y) => x + y, 0)
      pts.push({ label: stepLabel(test.t0 + LAGS + day * STEPS_PER_DAY + k, false), adaptiveC: Math.round(p.adaptive.C), adaptive: +p.adaptive.delay.toFixed(1), fixed: +p.fixedPlan.delay.toFixed(1) })
    }
    return { pts, saving: (1 - sumA / sumF) * 100 }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, a, sensors, day])

  const maxQ = Math.max(...cur.q)
  const W = 360, cx = W / 2
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-[28px] font-semibold tracking-tight text-white">Predictive signal planner</h1>
          <p className="mt-1 max-w-3xl text-[14px] text-ink-300">
            Forecasts become decisions: the next-15-minute demand from the federated model sets the cycle length and green split
            (Webster's method) before the queue builds, instead of a fixed 120-second plan.
          </p>
        </div>
        <Badge tone="warn">Illustrative junction: the 4 busiest detectors of a zone act as its approaches</Badge>
      </div>

      <Card>
        <div className="flex flex-wrap items-center gap-5 text-[13px] text-ink-300">
          <label className="flex items-center gap-2">Zone
            <select value={zone} onChange={(e) => setZone(Number(e.target.value))}>{net.zones.map((z) => <option key={z.zone} value={z.zone}>{zoneName(z.zone)}</option>)}</select>
          </label>
          <label className="flex items-center gap-2">Day
            <select value={day} onChange={(e) => setDay(Number(e.target.value))}>{Array.from({ length: 11 }, (_, d) => <option key={d} value={d}>{stepLabel(test.t0 + LAGS + d * STEPS_PER_DAY).split('·')[0]}</option>)}</select>
          </label>
          <label className="flex flex-1 items-center gap-3">Time <span className="num w-12 text-white">{stepLabel(test.t0 + t, false)}</span>
            <input type="range" min={0} max={STEPS_PER_DAY - 13} value={tod} onChange={(e) => setTod(Number(e.target.value))} />
          </label>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Recommended cycle" value={Math.round(cur.adaptive.C)} unit="s" tone="brand" hint={`fixed plan: ${cur.fixedPlan.C} s`} />
        <Stat label="Green N/S · E/W" value={`${Math.round(cur.adaptive.green[0])} · ${Math.round(cur.adaptive.green[1])}`} unit="s" hint="fixed plan: 55 · 55 s" />
        <Stat label="Avg delay per vehicle" value={cur.adaptive.delay.toFixed(1)} unit="s" tone="good" hint={`fixed plan: ${cur.fixedPlan.delay.toFixed(1)} s`} />
        <Stat label="Delay saved" value={`${saving.toFixed(0)}%`} tone={saving > 0 ? 'good' : 'warn'} hint={`whole day (06–22 h): ${daily.saving.toFixed(0)} %`} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[400px_1fr]">
        <Card title="Junction view" subtitle="Arrow width = forecast demand (veh/h) per approach for the next 15 min.">
          <svg viewBox={`0 0 ${W} ${W}`} className="w-full">
            <rect x={cx - 40} y={0} width={80} height={W} fill="#131c3d" />
            <rect x={0} y={cx - 40} width={W} height={80} fill="#131c3d" />
            <rect x={cx - 40} y={cx - 40} width={80} height={80} fill="#1d2852" />
            {APPROACHES.map((name, i) => {
              const w = 6 + (cur.q[i] / maxQ) * 22
              const green = i < 2 ? cur.adaptive.green[0] : cur.adaptive.green[1]
              const pts = [
                { x: cx - 18, y: 30, dx: 0, dy: 1 }, { x: cx + 18, y: W - 30, dx: 0, dy: -1 },
                { x: W - 30, y: cx - 18, dx: -1, dy: 0 }, { x: 30, y: cx + 18, dx: 1, dy: 0 },
              ][i]
              const x2 = pts.x + pts.dx * 110, y2 = pts.y + pts.dy * 110
              return (
                <g key={name}>
                  <line x1={pts.x} y1={pts.y} x2={x2} y2={y2} stroke={ZONE_COLORS[zone]} strokeWidth={w} strokeLinecap="round" opacity={0.85} />
                  <text x={i === 2 ? W - 60 : i === 3 ? 6 : pts.x + (i === 0 ? -100 : 26)} y={i < 2 ? (i === 0 ? 20 : W - 12) : pts.y - 22} fill="#cdd5f3" fontSize={11}>
                    {name} · S{sensors[i]}
                  </text>
                  <text x={i === 2 ? W - 60 : i === 3 ? 6 : pts.x + (i === 0 ? -100 : 26)} y={i < 2 ? (i === 0 ? 34 : W - 26) : pts.y - 9} fill="#7a87b8" fontSize={10}>
                    {Math.round(cur.q[i])} veh/h · green {Math.round(green)} s
                  </text>
                </g>
              )
            })}
          </svg>
          <div className="mt-3 space-y-1.5">
            {[{ label: 'Adaptive', p: cur.adaptive, c: '#38bdf8' }, { label: 'Fixed 120 s', p: cur.fixedPlan, c: '#7a87b8' }].map(({ label, p, c }) => (
              <div key={label}>
                <div className="mb-0.5 text-[11px] text-ink-400">{label} cycle ({Math.round(p.C)} s)</div>
                <div className="flex h-4 overflow-hidden rounded-md text-[9px] text-ink-950">
                  <div style={{ width: `${(p.green[0] / 150) * 100}%`, background: c }} className="flex items-center justify-center font-semibold">N/S</div>
                  <div style={{ width: `${(LOST / 2 / 150) * 100}%` }} className="bg-amber-300" />
                  <div style={{ width: `${(p.green[1] / 150) * 100}%`, background: c, opacity: 0.7 }} className="flex items-center justify-center font-semibold">E/W</div>
                  <div style={{ width: `${(LOST / 2 / 150) * 100}%` }} className="bg-amber-300" />
                </div>
              </div>
            ))}
          </div>
        </Card>

        <div className="space-y-5">
          <Card title="Average delay through the day" subtitle="Seconds per vehicle, HCM control-delay model, every 30 min from 06:00 to 22:00.">
            <div className="h-60">
              <ResponsiveContainer>
                <BarChart data={daily.pts} margin={{ left: -10, right: 8 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="label" interval={3} tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="fixed" name="Fixed 120 s plan" fill="#475569" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  <Bar dataKey="adaptive" name="Forecast-driven plan" fill="#38bdf8" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title="Recommended cycle length" subtitle="Longer cycles at the peaks, shorter cycles when demand is light.">
            <div className="h-40">
              <ResponsiveContainer>
                <LineChart data={daily.pts} margin={{ left: -10, right: 8 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="label" interval={3} tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} domain={[30, 160]} />
                  <Tooltip />
                  <Line dataKey="adaptiveC" name="Cycle (s)" stroke="#a78bfa" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </div>
      <p className="text-[12px] text-ink-400">
        Assumptions: two-phase signal, {LOST} s lost time per cycle, saturation flow {SAT_PER_LANE} veh/h/lane, one direction carries a quarter of the
        detector's flow. Green split by Webster (1958); cycle length (40–150 s) picked to minimise HCM 2000 control delay for the forecast demand. Both plans are then scored on the traffic that actually arrived. These numbers show the decision logic; field deployment needs junction geometry and turning counts.
      </p>
    </div>
  )
}
