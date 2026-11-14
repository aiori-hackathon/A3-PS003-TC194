import { useCallback, useMemo, useRef, useState } from 'react'
import { Pause, Play, RotateCcw, StepForward, Lock, EyeOff, Cpu } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Assets } from '../engine/data'
import { STEPS_PER_DAY, stepLabel } from '../engine/data'
import { DEFAULT_SETTINGS, FederatedSim, rdpEpsilon, type FLSettings, type RoundLog } from '../engine/federated'
import { mlpParamCount } from '../engine/models'
import { Topology, type Phase } from '../components/Topology'
import { Badge, Button, Card, Segmented, Slider, Stat, Toggle, ZONE_COLORS, fmtBytes, zoneName } from '../ui'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export default function Studio({ a }: { a: Assets }) {
  const [s, setS] = useState<FLSettings>(DEFAULT_SETTINGS)
  const [speed, setSpeed] = useState<1 | 2 | 4>(2)
  const simRef = useRef<FederatedSim | null>(null)
  const [, force] = useState(0)
  const [phase, setPhase] = useState<Phase>('idle')
  const [running, setRunning] = useState(false)
  const runRef = useRef(false)
  const [logs, setLogs] = useState<RoundLog[]>([])
  const [inspect, setInspect] = useState(0)
  const [sensorRow, setSensorRow] = useState(0)

  const getSim = useCallback(() => {
    if (!simRef.current) simRef.current = new FederatedSim(a.train, a.test, s)
    return simRef.current
  }, [a, s])
  const sim = simRef.current ?? getSim()

  const ha = a.results.runs['Historical average']?.metrics.avg.mae

  const doRound = useCallback(async () => {
    const sm = getSim()
    const delay = 650 / speed
    const log = await sm.step(async (ph) => {
      if (ph === 'train') { force((x) => x + 1); await sleep(delay / 3); return }
      setPhase(ph); force((x) => x + 1); await sleep(delay)
    })
    if (log) setLogs([...sm.logs])
    force((x) => x + 1)
    return log
  }, [getSim, speed])

  const start = async () => {
    if (runRef.current) return
    runRef.current = true; setRunning(true)
    while (runRef.current) {
      const log = await doRound()
      if (!log || getSim().round >= s.rounds) break
    }
    runRef.current = false; setRunning(false); setPhase('idle')
  }
  const pause = () => { runRef.current = false }
  const reset = (next = s) => {
    runRef.current = false; setRunning(false)
    simRef.current = new FederatedSim(a.train, a.test, next)
    setLogs([]); setPhase('idle'); force((x) => x + 1)
  }
  const update = (patch: Partial<FLSettings>) => { const next = { ...s, ...patch }; setS(next); reset(next) }

  const last = logs[logs.length - 1]
  const eps = s.dp ? rdpEpsilon(s.sigma, s.rounds) : Infinity

  // live forecast of the current global model for one sensor on one test day
  const day = 3
  const series = useMemo(() => {
    const from = day * STEPS_PER_DAY + 12, to = from + STEPS_PER_DAY
    const pred = sim.predictSeries(sensorRow, from, to, 3)
    const row = a.test.rows[a.train.sensors[sensorRow]]
    const pts = []
    for (let k = 0; k < to - from; k += 2) pts.push({ label: stepLabel(a.test.t0 + from + k, false), actual: row[from + k], forecast: Number.isFinite(pred[k]) ? Math.round(pred[k]) : null })
    return pts
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sim, sensorRow, logs.length])

  const chart = logs.map((l) => ({ round: l.round, fed: +l.fedMae.toFixed(2), local: +l.localMae.toFixed(2) }))
  const ci = sim.clients[inspect]

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-[28px] font-semibold tracking-tight text-white">Federated Studio</h1>
          <p className="mt-1 max-w-3xl text-[14px] text-ink-300">
            Real federated training running in this browser: {sim.clients.length} zone clients, real PeMS08 readings,
            a {mlpParamCount().toLocaleString()}-parameter edge model. Change the protocol and watch accuracy, privacy and bandwidth react.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Segmented<1 | 2 | 4> value={speed} onChange={setSpeed} options={[{ value: 1, label: '1×' }, { value: 2, label: '2×' }, { value: 4, label: '4×' }]} />
          {!running ? <Button onClick={start} disabled={sim.round >= s.rounds}><Play size={15} />{sim.round ? 'Resume' : 'Start training'}</Button> : <Button variant="ghost" onClick={pause}><Pause size={15} />Pause</Button>}
          <Button variant="ghost" onClick={() => doRound()} disabled={running || sim.round >= s.rounds}><StepForward size={15} />1 round</Button>
          <Button variant="ghost" onClick={() => reset()}><RotateCcw size={15} />Reset</Button>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[300px_1fr_380px]">
        <Card title="Protocol" subtitle="Changing a setting restarts training.">
          <div className="space-y-4">
            <div>
              <div className="label mb-1.5">Aggregation</div>
              <Segmented<'fedavg' | 'fedprox'> value={s.strategy} onChange={(v) => update({ strategy: v })} options={[{ value: 'fedavg', label: 'FedAvg' }, { value: 'fedprox', label: 'FedProx' }]} />
            </div>
            {s.strategy === 'fedprox' && <Slider label="Proximal term μ" value={s.mu} min={0.001} max={0.1} step={0.001} onChange={(v) => update({ mu: v })} />}
            <Slider label="Rounds" value={s.rounds} min={5} max={60} step={1} onChange={(v) => update({ rounds: v })} />
            <Slider label="Local steps / round" value={s.localSteps} min={5} max={60} step={5} onChange={(v) => update({ localSteps: v })} />
            <Slider label="Learning rate" value={s.lr} min={0.0005} max={0.01} step={0.0005} onChange={(v) => update({ lr: v })} format={(v) => v.toFixed(4)} />
            <Slider label="History per zone" value={s.historyDays} min={1} max={14} step={1} onChange={(v) => update({ historyDays: v })} format={(v) => `${v} day${v > 1 ? 's' : ''}`} />
            <Slider label="Client dropout" value={s.dropout} min={0} max={0.6} step={0.05} onChange={(v) => update({ dropout: v })} format={(v) => `${Math.round(v * 100)} %`} />
            <div className="h-px bg-white/5" />
            <Toggle label="Secure aggregation" hint="Pairwise masks: server can't read any single zone's update" checked={s.secureAgg} onChange={(v) => update({ secureAgg: v })} />
            <Toggle label="Differential privacy" hint="Clip each update + add Gaussian noise" checked={s.dp} onChange={(v) => update({ dp: v })} />
            {s.dp && (
              <>
                <Slider label="Clip norm C" value={s.clip} min={0.1} max={3} step={0.1} onChange={(v) => update({ clip: v })} />
                <Slider label="Noise multiplier σ" value={s.sigma} min={0.1} max={3} step={0.1} onChange={(v) => update({ sigma: v })} />
                <div className="rounded-xl bg-violet-400/10 px-3 py-2 text-[12px] text-violet-200 ring-1 ring-violet-400/20">
                  Privacy budget after {s.rounds} rounds: <span className="num font-semibold">ε = {Number.isFinite(eps) ? eps.toFixed(1) : '∞'}</span> (δ = 10⁻⁵)
                </div>
              </>
            )}
          </div>
        </Card>

        <Card title={<span className="flex items-center gap-2">Training protocol <Badge tone="brand">round {sim.round}/{s.rounds}</Badge></span>}
          right={<div className="flex gap-1.5">{s.secureAgg && <Badge tone="good"><Lock size={11} />SecAgg</Badge>}{s.dp && <Badge tone="violet"><EyeOff size={11} />DP</Badge>}</div>}>
          <Topology clients={sim.clients} phase={phase} round={Math.max(1, sim.round)} secure={s.secureAgg} dp={s.dp} />
        </Card>

        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Global MAE" value={last ? last.fedMae.toFixed(2) : '–'} unit="veh" tone="brand" hint="test days, all horizons" />
            <Stat label="Local-only MAE" value={last ? last.localMae.toFixed(2) : '–'} unit="veh" hint="each zone alone" />
            <Stat label="Uploaded" value={fmtBytes(sim.bytesUp)} tone="good" hint={`vs ${fmtBytes(sim.rawBytes)} raw data`} />
            <Stat label="Privacy budget" value={s.dp && last?.epsilon ? last.epsilon.toFixed(1) : '∞'} tone={s.dp ? 'brand' : 'warn'} hint={s.dp ? 'ε spent so far (lower = more private)' : 'ε = ∞ · DP off'} />
          </div>
          <Card title="Accuracy per round" subtitle="Mean absolute error on the 12 held-out test days (lower is better).">
            <div className="h-52">
              <ResponsiveContainer>
                <LineChart data={chart} margin={{ left: -10, right: 8, top: 5 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="round" tickLine={false} axisLine={false} />
                  <YAxis domain={['auto', 'auto']} tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Legend />
                  {ha && <ReferenceLine y={ha} stroke="#f87171" strokeDasharray="4 4" label={{ value: 'historical avg', fill: '#f87171', fontSize: 10, position: 'insideTopRight' }} />}
                  <Line type="monotone" dataKey="fed" name="Federated" stroke="#38bdf8" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="local" name="Local-only" stroke="#a3aed6" strokeDasharray="4 4" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[1fr_1fr]">
        <Card title="Zones (clients)" subtitle="Each row is a separate organisation. Click a zone to inspect what it sends.">
          <table className="w-full text-[12.5px]">
            <thead><tr className="text-left text-ink-400"><th className="pb-2 font-medium">Zone</th><th className="font-medium">Sensors</th><th className="font-medium">Private samples</th><th className="font-medium">Status</th><th className="font-medium">Local loss</th><th className="font-medium">Update ‖Δ‖</th></tr></thead>
            <tbody>
              {sim.clients.map((c, i) => (
                <tr key={c.zone} onClick={() => setInspect(i)} className={`cursor-pointer border-t border-white/5 hover:bg-white/[0.03] ${inspect === i ? 'bg-white/[0.04]' : ''}`}>
                  <td className="py-1.5"><span className="inline-flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: ZONE_COLORS[c.zone] }} />{zoneName(c.zone)}</span></td>
                  <td className="num">{c.sensors.length}</td>
                  <td className="num">{c.samples.toLocaleString()}</td>
                  <td><Badge tone={c.status === 'dropped' ? 'bad' : c.status === 'done' ? 'good' : c.status === 'idle' ? 'default' : 'brand'}>{c.status}</Badge></td>
                  <td className="num">{c.loss?.toFixed(4) ?? '–'}</td>
                  <td className="num">{c.norm?.toFixed(3) ?? '–'}{c.clipped && <span className="ml-1 text-violet-300">✂</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title={<span className="flex items-center gap-2"><Cpu size={15} />What the aggregator actually sees · {ci ? zoneName(ci.zone) : ''}</span>}
          subtitle="First 6 of the update values. With secure aggregation the server only sees masked numbers; the masks cancel exactly when all zones are summed.">
          {!ci?.truePreview ? <p className="text-[13px] text-ink-400">Run at least one round.</p> : (
            <div className="space-y-3 font-mono text-[12px]">
              <div>
                <div className="label mb-1">Zone's real update (never leaves the zone)</div>
                <div className="flex flex-wrap gap-1.5">{ci.truePreview.map((v, k) => <span key={k} className="rounded-md bg-emerald-400/10 px-2 py-1 text-emerald-200">{v.toFixed(5)}</span>)}</div>
              </div>
              <div>
                <div className="label mb-1">Received by server</div>
                <div className="flex flex-wrap gap-1.5">{(ci.maskedPreview ?? ci.truePreview).map((v, k) => <span key={k} className={`rounded-md px-2 py-1 ${ci.maskedPreview ? 'bg-amber-400/10 text-amber-200' : 'bg-red-400/10 text-red-200'}`}>{v.toFixed(5)}</span>)}</div>
                {!ci.maskedPreview && <div className="mt-1 text-[11.5px] text-red-300">Secure aggregation is off: the server can read this zone's update.</div>}
              </div>
              <div>
                <div className="label mb-1">Sum of all masked updates = true aggregate</div>
                <div className="flex flex-wrap gap-1.5">{(last?.serverView ?? []).map((v, k) => <span key={k} className="rounded-md bg-sky-400/10 px-2 py-1 text-sky-200">{v.toFixed(5)}</span>)}</div>
              </div>
            </div>
          )}
        </Card>
      </div>

      <Card title="The global model getting smarter" subtitle="15-minute-ahead forecast of the current global model for one sensor on an unseen test day. It updates after every round."
        right={<select value={sensorRow} onChange={(e) => setSensorRow(Number(e.target.value))}>{a.train.sensors.map((id, r) => <option key={id} value={r}>Sensor {id} · {zoneName(a.train.zones[r])}</option>)}</select>}>
        <div className="h-56">
          <ResponsiveContainer>
            <LineChart data={series} margin={{ left: -10, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="label" interval={17} tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="actual" name="Actual flow" stroke="#a3aed6" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              <Line type="monotone" dataKey="forecast" name={`Global model v${sim.round}`} stroke="#38bdf8" dot={false} strokeWidth={2.2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  )
}
