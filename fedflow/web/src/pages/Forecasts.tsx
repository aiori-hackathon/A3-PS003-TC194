import { useMemo, useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Assets } from '../engine/data'
import { HORIZONS, HORIZON_LABELS, LAGS, STEPS_PER_DAY, stepLabel } from '../engine/data'
import { loadGRU } from '../engine/models'
import { forecastSeries } from '../engine/forecast'
import { Badge, Card, Segmented, Stat, zoneName } from '../ui'

export default function Forecasts({ a }: { a: Assets }) {
  const { network: net, test } = a
  const fed = useMemo(() => loadGRU(a.gru.federated.weights, a.gru.federated.norm), [a])
  const locals = useMemo(() => Object.fromEntries(Object.entries(a.gru.local).map(([z, v]) => [Number(z), loadGRU(v.weights, v.norm)])), [a])
  const [sensor, setSensor] = useState(net.zones[0].sensors[0])
  const [day, setDay] = useState(3)
  const [h, setH] = useState(2) // horizon index → 30 min
  const zone = net.sensors[sensor].zone
  const days = Math.floor((test.steps - LAGS - 12) / STEPS_PER_DAY)

  const data = useMemo(() => {
    const from = LAGS + day * STEPS_PER_DAY, to = from + STEPS_PER_DAY
    const hs = HORIZONS[h]
    const pf = forecastSeries(fed, test, sensor, from, to, h, hs)
    const pl = forecastSeries(locals[zone], test, sensor, from, to, h, hs)
    const row = test.rows[sensor]
    let ef = 0, el = 0, n = 0
    const pts = []
    for (let k = 0; k < to - from; k++) {
      const act = row[from + k]
      if (pf[k] !== null && pl[k] !== null) { ef += Math.abs(pf[k]! - act); el += Math.abs(pl[k]! - act); n++ }
      if (k % 2 === 0) pts.push({ label: stepLabel(test.t0 + from + k, false), actual: act, fed: pf[k] !== null ? Math.round(pf[k]!) : null, local: pl[k] !== null ? Math.round(pl[k]!) : null, band: pf[k] !== null ? [Math.round(pf[k]! * 0.88), Math.round(pf[k]! * 1.12)] : null })
    }
    return { pts, maeF: ef / n, maeL: el / n, title: stepLabel(test.t0 + from, true).split('·')[0] }
  }, [fed, locals, zone, test, sensor, day, h])

  const best = data.maeF <= data.maeL
  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-[28px] font-semibold tracking-tight text-white">Forecast explorer</h1>
        <p className="mt-1 max-w-3xl text-[14px] text-ink-300">
          Pick any of the 170 sensors and any of the 12 test days the model has never seen. Compare the federated GRU with a
          model the zone would have trained on its own.
        </p>
      </div>
      <Card>
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-[13px] text-ink-300">Sensor
            <select value={sensor} onChange={(e) => setSensor(Number(e.target.value))}>
              {net.zones.map((z) => (
                <optgroup key={z.zone} label={`${zoneName(z.zone)} (${z.sensors.length})`}>
                  {z.sensors.map((id) => <option key={id} value={id}>Sensor {id}</option>)}
                </optgroup>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-[13px] text-ink-300">Test day
            <select value={day} onChange={(e) => setDay(Number(e.target.value))}>
              {Array.from({ length: days }, (_, d) => <option key={d} value={d}>{stepLabel(test.t0 + LAGS + d * STEPS_PER_DAY).split('·')[0]}</option>)}
            </select>
          </label>
          <div className="flex items-center gap-2 text-[13px] text-ink-300">Horizon
            <Segmented<number> value={h} onChange={setH} options={HORIZON_LABELS.map((l, i) => ({ value: i, label: l }))} />
          </div>
          <Badge tone="brand">{zoneName(zone)}</Badge>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <Stat label="Federated GRU · MAE" value={data.maeF.toFixed(1)} unit="veh/5min" tone="brand" hint={`${HORIZON_LABELS[h]} ahead, ${data.title}`} />
        <Stat label="Zone-only GRU · MAE" value={data.maeL.toFixed(1)} unit="veh/5min" hint="trained on this zone's data only" />
        <Stat label="Difference" value={`${best ? '−' : '+'}${Math.abs(((data.maeF - data.maeL) / data.maeL) * 100).toFixed(1)}%`} tone={best ? 'good' : 'warn'} hint={best ? 'federated is more accurate here' : 'zone model slightly better here'} />
        <Stat label="Normal peak (p95)" value={Math.round(net.sensors[sensor].p95)} unit="veh/5min" hint="used for congestion alerts" />
      </div>

      <Card title={`Sensor ${sensor} · ${data.title} · ${HORIZON_LABELS[h]}-ahead forecast`} subtitle="Shaded band = ±12 % around the federated forecast (typical error range).">
        <div className="h-[380px]">
          <ResponsiveContainer>
            <ComposedChart data={data.pts} margin={{ left: -6, right: 10, top: 10 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="label" interval={11} tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} />
              <Tooltip />
              <Legend />
              <Area dataKey="band" name="Forecast band" stroke="none" fill="#38bdf8" fillOpacity={0.12} isAnimationActive={false} />
              <Line dataKey="actual" name="Actual" stroke="#e2e8f0" dot={false} strokeWidth={1.6} isAnimationActive={false} />
              <Line dataKey="fed" name="Federated GRU" stroke="#38bdf8" dot={false} strokeWidth={2.4} isAnimationActive={false} />
              <Line dataKey="local" name="Zone-only GRU" stroke="#f472b6" dot={false} strokeWidth={1.4} strokeDasharray="5 4" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  )
}
