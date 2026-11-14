import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Database, Gauge, Lock, Pause, Play, Radio, SkipForward, Timer } from 'lucide-react'
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Assets } from '../engine/data'
import { LAGS, STEPS_PER_DAY, stepLabel } from '../engine/data'
import { loadGRU } from '../engine/models'
import { forecastAt } from '../engine/forecast'
import { NetworkMap, type NodeValue } from '../components/NetworkMap'
import { Badge, Button, Card, Segmented, Stat, ZONE_COLORS, congestionColor, congestionLabel, zoneName } from '../ui'

type Mode = 'now' | 15 | 30 | 60

export default function CommandCentre({ a }: { a: Assets }) {
  const { network: net, test, results } = a
  const model = useMemo(() => loadGRU(a.gru.federated.weights, a.gru.federated.norm), [a])
  const [t, setT] = useState(LAGS + STEPS_PER_DAY * 3 + 7 * 12 + 6) // Monday ~07:30 of the test period
  const [playing, setPlaying] = useState(false)
  const [mode, setMode] = useState<Mode>(30)
  const [zone, setZone] = useState<number | null>(null)
  const maxT = test.steps - 12

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => setT((v) => (v + 1 >= maxT ? LAGS : v + 1)), 350)
    return () => clearInterval(id)
  }, [playing, maxT])

  // forecasts for every sensor at time t (GRU, federated global model)
  const fc = useMemo(() => net.sensors.map((s) => forecastAt(model, test, s.id, t)), [net, model, test, t])
  const hIdx = mode === 'now' ? -1 : [5, 15, 30, 60].indexOf(mode)
  const values: (NodeValue | null)[] = net.sensors.map((s, i) => {
    const now = test.rows[s.id][t - 1]
    const f = fc[i]
    const v = hIdx < 0 || !f ? now : f[hIdx]
    return { flow: v, ratio: v / s.p95, forecast: f ? f[Math.max(hIdx, 2)] : undefined }
  })

  // alerts: congestion expected in 30 min at sensors that are not congested now
  const alerts = net.sensors
    .map((s, i) => ({ s, now: test.rows[s.id][t - 1] / s.p95, f30: fc[i] ? fc[i]![2] / s.p95 : 0, f30v: fc[i]?.[2] ?? 0 }))
    .filter((x) => x.f30 >= 0.95 && x.now < x.f30)
    .sort((x, y) => y.f30 - x.f30)
    .slice(0, 7)

  const zoneRows = net.zones.map((z) => {
    const nowR = z.sensors.reduce((s, id) => s + test.rows[id][t - 1] / net.sensors[id].p95, 0) / z.sensors.length
    const fR = z.sensors.reduce((s, id, k) => s + (fc[id]?.[2] ?? 0) / net.sensors[id].p95 + 0 * k, 0) / z.sensors.length
    return { z: z.zone, n: z.sensors.length, nowR, fR }
  })

  // city-wide total flow (actual) for the current day + forecast
  const dayStart = Math.max(LAGS, t - (t % STEPS_PER_DAY))
  const cityDay = useMemo(() => {
    const pts = []
    for (let k = dayStart; k < Math.min(dayStart + STEPS_PER_DAY, test.steps); k += 3) {
      let s = 0
      for (const row of test.rows) s += row[k]
      pts.push({ k, label: stepLabel(test.t0 + k, false), actual: k <= t ? s : null, future: k > t ? s : null })
    }
    return pts
  }, [dayStart, t, test])

  const fed = results.runs['FedAvg · GRU']?.metrics
  const severe = values.filter((v) => v && v.ratio >= 1.05).length
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-[28px] font-semibold tracking-tight text-white">City traffic command centre</h1>
          <p className="mt-1 max-w-3xl text-[14px] text-ink-300">
            Every zone keeps its sensor data on its own server. A single forecasting model, trained collaboratively with
            federated learning, predicts flow up to an hour ahead for all 170 sensors.
          </p>
        </div>
        <Badge tone="good"><Lock size={12} /> 0 bytes of raw sensor data left any zone</Badge>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Sensors online" value={net.sensors.length} hint={`${net.zones.length} zones, one federated client each`} icon={<Radio size={16} />} />
        <Stat label="30-min forecast error" value={fed ? fed['30min'].mae.toFixed(1) : '–'} unit="veh/5min" tone="brand" hint={fed ? `MAPE ${fed['30min'].mape.toFixed(1)} % on 12 unseen days` : ''} icon={<Gauge size={16} />} />
        <Stat label="Global model" value={`v${results.communication.rounds}`} hint={`${results.communication.params_gru.toLocaleString()} parameters · GRU`} icon={<Database size={16} />} />
        <Stat label="Severe right now" value={severe} tone={severe ? 'bad' : 'good'} hint={mode === 'now' ? 'sensors above normal peak' : `forecast at +${mode} min`} icon={<AlertTriangle size={16} />} />
        <Stat label="Early warnings" value={alerts.length} tone={alerts.length ? 'warn' : 'good'} hint="jams predicted 30 min ahead" icon={<Timer size={16} />} />
        <Stat label="Data shared" value={(results.communication.model_bytes / 1024).toFixed(1)} unit="KB / round / zone" tone="good" hint="model update only, encrypted" icon={<Lock size={16} />} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1fr_380px]">
        <Card
          title={<span className="flex items-center gap-2">Live network <span className="num text-[13px] font-normal text-brand-400">{stepLabel(test.t0 + t)}</span></span>}
          subtitle="Sensor layout reconstructed from road-network distances. Node colour = flow relative to that sensor's normal peak."
          right={<Segmented<Mode> value={mode} onChange={setMode} options={[{ value: 'now', label: 'Now' }, { value: 15, label: '+15 min' }, { value: 30, label: '+30 min' }, { value: 60, label: '+60 min' }]} />}
        >
          <NetworkMap net={net} values={values} highlightZone={zone} height={470} />
          <div className="mt-3 flex items-center gap-3">
            <Button variant="ghost" onClick={() => setPlaying((p) => !p)}>{playing ? <Pause size={15} /> : <Play size={15} />}{playing ? 'Pause' : 'Play'}</Button>
            <Button variant="ghost" onClick={() => setT((v) => Math.min(maxT, v + 12))}><SkipForward size={15} />+1 h</Button>
            <input type="range" min={LAGS} max={maxT} value={t} onChange={(e) => setT(Number(e.target.value))} className="flex-1" />
          </div>
        </Card>

        <div className="space-y-5">
          <Card title="Early-warning feed" subtitle="Sensors not jammed now, but predicted to exceed their normal peak in 30 min.">
            {alerts.length === 0 && <p className="text-[13px] text-ink-400">No congestion expected in the next 30 minutes.</p>}
            <ul className="space-y-2">
              {alerts.map((x) => (
                <li key={x.s.id} className="flex items-center justify-between rounded-xl bg-white/[0.03] px-3 py-2 ring-1 ring-white/5">
                  <div>
                    <div className="text-[13px] font-medium text-white">Sensor {x.s.id} <span className="text-ink-400">· {zoneName(x.s.zone)}</span></div>
                    <div className="text-[11.5px] text-ink-400">now {Math.round(x.now * 100)} % → <span style={{ color: congestionColor(x.f30) }}>{Math.round(x.f30 * 100)} % of peak</span> ({Math.round(x.f30v)} veh/5 min)</div>
                  </div>
                  <Badge tone={x.f30 >= 1.05 ? 'bad' : 'warn'}>{congestionLabel(x.f30)}</Badge>
                </li>
              ))}
            </ul>
          </Card>
          <Card title="Zones" subtitle="Average load now → in 30 min. Click to highlight.">
            <div className="space-y-1.5">
              {zoneRows.map((r) => (
                <button key={r.z} onClick={() => setZone(zone === r.z ? null : r.z)} className={`flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left transition hover:bg-white/5 ${zone === r.z ? 'bg-white/5 ring-1 ring-white/10' : ''}`}>
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: ZONE_COLORS[r.z] }} />
                  <span className="w-16 text-[13px] text-ink-200">{zoneName(r.z)}</span>
                  <span className="w-16 text-[11px] text-ink-400">{r.n} sensors</span>
                  <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-white/5">
                    <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.min(100, r.fR * 100)}%`, background: congestionColor(r.fR), opacity: 0.35 }} />
                    <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.min(100, r.nowR * 100)}%`, background: congestionColor(r.nowR) }} />
                  </div>
                  <span className="num w-20 text-right text-[12px] text-ink-300">{Math.round(r.nowR * 100)}→{Math.round(r.fR * 100)}%</span>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <Card title="City-wide demand today" subtitle="Total flow across all 170 sensors (vehicles per 5 minutes). Solid = observed so far, faded = rest of the day.">
        <div className="h-48">
          <ResponsiveContainer>
            <AreaChart data={cityDay} margin={{ left: 0, right: 8, top: 5, bottom: 0 }}>
              <defs>
                <linearGradient id="ga" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#38bdf8" stopOpacity={0.5} /><stop offset="1" stopColor="#38bdf8" stopOpacity={0} /></linearGradient>
              </defs>
              <XAxis dataKey="label" interval={11} tickLine={false} axisLine={false} />
              <YAxis width={50} tickLine={false} axisLine={false} />
              <Tooltip />
              <Area type="monotone" dataKey="future" stroke="#7a87b8" strokeDasharray="4 4" fill="transparent" name="Later today" />
              <Area type="monotone" dataKey="actual" stroke="#38bdf8" strokeWidth={2} fill="url(#ga)" name="Observed" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  )
}
