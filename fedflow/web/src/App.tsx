import { useEffect, useState } from 'react'
import { Activity, BarChart3, Boxes, BrainCircuit, LineChart, ShieldCheck, TrafficCone, Loader2 } from 'lucide-react'
import clsx from 'clsx'
import { loadAssets, type Assets } from './engine/data'
import CommandCentre from './pages/CommandCentre'
import Studio from './pages/Studio'
import Forecasts from './pages/Forecasts'
import SignalPlanner from './pages/SignalPlanner'
import Privacy from './pages/Privacy'
import Benchmarks from './pages/Benchmarks'
import Architecture from './pages/Architecture'

const TABS = [
  { id: 'command', label: 'Command Centre', icon: Activity },
  { id: 'studio', label: 'Federated Studio', icon: BrainCircuit },
  { id: 'forecast', label: 'Forecasts', icon: LineChart },
  { id: 'signals', label: 'Signal Planner', icon: TrafficCone },
  { id: 'privacy', label: 'Privacy & Security', icon: ShieldCheck },
  { id: 'bench', label: 'Benchmarks', icon: BarChart3 },
  { id: 'arch', label: 'Architecture', icon: Boxes },
] as const
type Tab = (typeof TABS)[number]['id']

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden>
      <defs><linearGradient id="lg" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stopColor="#7dd3fc" /><stop offset="1" stopColor="#a78bfa" /></linearGradient></defs>
      <circle cx="16" cy="16" r="4.5" fill="url(#lg)" />
      {[0, 72, 144, 216, 288].map((a) => {
        const x = 16 + 11 * Math.cos((a * Math.PI) / 180), y = 16 + 11 * Math.sin((a * Math.PI) / 180)
        return <g key={a}><line x1="16" y1="16" x2={x} y2={y} stroke="url(#lg)" strokeWidth="1.6" opacity=".7" /><circle cx={x} cy={y} r="2.6" fill="#0a1023" stroke="url(#lg)" strokeWidth="1.6" /></g>
      })}
    </svg>
  )
}

export default function App() {
  const [assets, setAssets] = useState<Assets | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>(() => (location.hash.slice(1) as Tab) || 'command')
  useEffect(() => { loadAssets().then(setAssets).catch((e) => setErr(String(e))) }, [])
  useEffect(() => { location.hash = tab }, [tab])

  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-30 border-b border-white/5 bg-ink-950/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1500px] items-center gap-6 px-5 py-3">
          <div className="flex items-center gap-2.5">
            <Logo />
            <div className="leading-tight">
              <div className="font-[family-name:var(--font-display)] text-[17px] font-semibold tracking-tight text-white">FedFlow</div>
              <div className="text-[11px] text-ink-400">Traffic intelligence, learned privately</div>
            </div>
          </div>
          <nav className="flex flex-1 items-center gap-1 overflow-x-auto">
            {TABS.map((t) => (
              <button key={t.id} onClick={() => setTab(t.id)}
                className={clsx('flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium transition',
                  tab === t.id ? 'bg-white/10 text-white ring-1 ring-white/10' : 'text-ink-400 hover:bg-white/5 hover:text-ink-200')}>
                <t.icon size={15} />{t.label}
              </button>
            ))}
          </nav>
          <div className="hidden items-center gap-2 text-[12px] text-ink-400 xl:flex">
            <span className="relative flex h-2 w-2"><span className="absolute h-2 w-2 animate-ping rounded-full bg-emerald-400 opacity-60" /><span className="h-2 w-2 rounded-full bg-emerald-400" /></span>
            8 zones · 170 sensors
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1500px] px-5 py-6">
        {err && <div className="card p-6 text-red-300">Could not load data: {err}</div>}
        {!assets && !err && (
          <div className="flex h-[70vh] flex-col items-center justify-center gap-3 text-ink-400"><Loader2 className="animate-spin" /> Loading network, sensors and trained models…</div>
        )}
        {assets && (
          <>
            {tab === 'command' && <CommandCentre a={assets} />}
            {tab === 'studio' && <Studio a={assets} />}
            {tab === 'forecast' && <Forecasts a={assets} />}
            {tab === 'signals' && <SignalPlanner a={assets} />}
            {tab === 'privacy' && <Privacy a={assets} />}
            {tab === 'bench' && <Benchmarks a={assets} />}
            {tab === 'arch' && <Architecture a={assets} />}
          </>
        )}
      </main>
      <footer className="mx-auto max-w-[1500px] px-5 pb-8 pt-2 text-[11.5px] text-ink-400">
        FedFlow · built by Team NeuroForge · Data: Caltrans PeMS District 8 (PeMS08), 170 loop detectors, Jul–Aug 2016 · All models run in your browser
      </footer>
    </div>
  )
}
