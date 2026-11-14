import { ArrowDown, ArrowUp, Cloud, Cpu, Database, Globe, Layers, Lock, Radio, Server, TrafficCone } from 'lucide-react'
import type { Assets } from '../engine/data'
import { Badge, Card } from '../ui'

function Box({ icon: Icon, title, body, tone = 'sky' }: { icon: typeof Cpu; title: string; body: string; tone?: 'sky' | 'violet' | 'emerald' | 'amber' }) {
  const c = { sky: 'text-sky-300 ring-sky-400/20 bg-sky-400/5', violet: 'text-violet-300 ring-violet-400/20 bg-violet-400/5', emerald: 'text-emerald-300 ring-emerald-400/20 bg-emerald-400/5', amber: 'text-amber-300 ring-amber-400/20 bg-amber-400/5' }[tone]
  return (
    <div className={`rounded-xl p-3.5 ring-1 ${c}`}>
      <div className="flex items-center gap-2 text-[13px] font-semibold"><Icon size={16} />{title}</div>
      <div className="mt-1 text-[12px] text-ink-300">{body}</div>
    </div>
  )
}

export default function Architecture({ a }: { a: Assets }) {
  const c = a.results.communication
  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-[28px] font-semibold tracking-tight text-white">How FedFlow works</h1>
        <p className="mt-1 max-w-3xl text-[14px] text-ink-300">Three layers: sensing at the edge, private training per zone, and a coordinator that only ever handles encrypted model updates.</p>
      </div>

      <Card title="System architecture">
        <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr_auto_1fr] lg:items-center">
          <div className="space-y-3">
            <div className="label">1 · Edge (per junction)</div>
            <Box icon={Radio} title="Loop detectors / cameras" body="Flow, occupancy, speed every 5 min" tone="emerald" />
            <Box icon={Cpu} title="Edge model (TemporalMLP)" body={`${c.params_mlp.toLocaleString()} params, runs on a Raspberry Pi`} tone="emerald" />
          </div>
          <div className="hidden flex-col items-center text-ink-400 lg:flex"><ArrowUp className="rotate-90" /><span className="text-[10px]">readings</span></div>
          <div className="space-y-3">
            <div className="label">2 · Zone server (federated client)</div>
            <Box icon={Database} title="Private data store" body="Raw readings never leave this box" tone="sky" />
            <Box icon={Layers} title="Local trainer" body={`GRU, ${c.params_gru.toLocaleString()} params · clip + noise · masks`} tone="sky" />
            <Box icon={TrafficCone} title="Signal & alert service" body="Uses the global model for 5–60 min forecasts" tone="sky" />
          </div>
          <div className="hidden flex-col items-center text-ink-400 lg:flex"><Lock size={16} /><span className="text-[10px]">Δ weights only</span><ArrowDown className="-rotate-90" /></div>
          <div className="space-y-3">
            <div className="label">3 · Coordinator</div>
            <Box icon={Server} title="Secure aggregator" body="Sums masked updates, FedAvg / FedProx" tone="violet" />
            <Box icon={Cloud} title="Model registry" body="Versioned global models, rollback" tone="violet" />
            <Box icon={Globe} title="City dashboard" body="This web app: forecasts, alerts, signal plans" tone="violet" />
          </div>
        </div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="One federated round">
          <ol className="space-y-2.5 text-[13px] text-ink-300">
            {[
              ['Broadcast', 'Coordinator sends the current global model (≈14 KB) to every zone.'],
              ['Local training', 'Each zone trains for a few steps on its own last 43 days of readings.'],
              ['Protect', 'Each zone clips its update (bounds influence), optionally adds Gaussian noise (DP), then adds pairwise masks.'],
              ['Upload', 'Only the masked update goes up, over TLS.'],
              ['Aggregate', 'Masks cancel in the sum; the coordinator applies the weighted average → global model v+1.'],
              ['Deploy', 'New model is pushed back; zones use it for forecasts, alerts and signal timing.'],
            ].map(([t, d], i) => (
              <li key={t} className="flex gap-3"><span className="num flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-sky-400/15 text-[12px] text-sky-300">{i + 1}</span><span><span className="font-semibold text-white">{t}. </span>{d}</span></li>
            ))}
          </ol>
        </Card>
        <Card title="Technology">
          <div className="grid grid-cols-2 gap-3 text-[12.5px]">
            {[
              ['Learning core', 'Python · NumPy (hand-written GRU & MLP with back-propagation, gradient-checked)'],
              ['Federated engine', 'FedAvg, FedProx, client-level DP with RDP accountant, pairwise-mask secure aggregation, dropout'],
              ['Network mode', 'HTTP coordinator + zone clients (ml/server.py, ml/client.py) – runs across laptops'],
              ['Dashboard', 'React 19 · TypeScript · Vite · Tailwind · Recharts'],
              ['In-browser training', 'Same MLP + FedAvg re-implemented in TypeScript, runs offline'],
              ['Data', 'Caltrans PeMS08: 170 sensors, 62 days, 5-min resolution'],
            ].map(([k, v]) => (
              <div key={k} className="rounded-xl bg-white/[0.03] p-3 ring-1 ring-white/5"><div className="font-semibold text-white">{k}</div><div className="mt-0.5 text-ink-300">{v}</div></div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Roadmap to a city pilot">
        <div className="grid gap-3 md:grid-cols-4">
          {[
            ['Now', 'Working prototype on real PeMS08 data; dashboard, network mode, measured benchmarks.', 'good'],
            ['Pilot (3 months)', 'Pune / Mumbai: 3–5 agencies as clients, ingest from ATCS/ITMS controllers via MQTT.', 'brand'],
            ['Model upgrades', 'Graph neural networks (spatio-temporal), weather & event features, personalised FL.', 'violet'],
            ['Scale', 'Hundreds of junction clients with DP on; adaptive signal control in closed loop.', 'warn'],
          ].map(([t, d, tone]) => (
            <div key={t} className="rounded-xl bg-white/[0.03] p-4 ring-1 ring-white/5">
              <Badge tone={tone as 'good'}>{t}</Badge>
              <p className="mt-2 text-[12.5px] text-ink-300">{d}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
