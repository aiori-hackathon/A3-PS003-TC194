import clsx from 'clsx'
import type { ReactNode } from 'react'

export function Card({ className, children, title, subtitle, right }: { className?: string; children: ReactNode; title?: ReactNode; subtitle?: ReactNode; right?: ReactNode }) {
  return (
    <section className={clsx('card p-5', className)}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h3 className="font-[family-name:var(--font-display)] text-[15px] font-semibold text-white">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-[12.5px] leading-snug text-ink-400">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, unit, hint, tone = 'default', icon }: { label: string; value: ReactNode; unit?: string; hint?: ReactNode; tone?: 'default' | 'good' | 'warn' | 'bad' | 'brand'; icon?: ReactNode }) {
  const color = { default: 'text-white', good: 'text-flow-low', warn: 'text-flow-mid', bad: 'text-flow-high', brand: 'text-brand-400' }[tone]
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between"><span className="label">{label}</span><span className="text-ink-400">{icon}</span></div>
      <div className={clsx('mt-2 num text-[26px] font-semibold leading-none', color)}>
        {value}{unit && <span className="ml-1 text-[13px] font-medium text-ink-400">{unit}</span>}
      </div>
      {hint && <div className="mt-2 text-[12px] text-ink-400">{hint}</div>}
    </div>
  )
}

export function Badge({ children, tone = 'default' }: { children: ReactNode; tone?: 'default' | 'good' | 'warn' | 'bad' | 'brand' | 'violet' }) {
  const cls = {
    default: 'bg-white/5 text-ink-300 ring-white/10',
    good: 'bg-emerald-400/10 text-emerald-300 ring-emerald-400/20',
    warn: 'bg-amber-400/10 text-amber-300 ring-amber-400/20',
    bad: 'bg-red-400/10 text-red-300 ring-red-400/20',
    brand: 'bg-sky-400/10 text-sky-300 ring-sky-400/20',
    violet: 'bg-violet-400/10 text-violet-300 ring-violet-400/20',
  }[tone]
  return <span className={clsx('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1', cls)}>{children}</span>
}

export function Button({ children, onClick, variant = 'primary', disabled, className }: { children: ReactNode; onClick?: () => void; variant?: 'primary' | 'ghost' | 'danger'; disabled?: boolean; className?: string }) {
  const cls = {
    primary: 'bg-gradient-to-b from-sky-400 to-sky-600 text-ink-950 shadow-[0_8px_24px_-8px_#38bdf8] hover:brightness-110',
    ghost: 'bg-white/5 text-ink-200 ring-1 ring-white/10 hover:bg-white/10',
    danger: 'bg-red-500/15 text-red-300 ring-1 ring-red-400/20 hover:bg-red-500/25',
  }[variant]
  return (
    <button onClick={onClick} disabled={disabled} className={clsx('inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-[13px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-40', cls, className)}>
      {children}
    </button>
  )
}

export function Toggle({ checked, onChange, label, hint, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string; disabled?: boolean }) {
  return (
    <label className={clsx('flex cursor-pointer items-start justify-between gap-3', disabled && 'opacity-50')}>
      <span>
        <span className="block text-[13px] font-medium text-ink-200">{label}</span>
        {hint && <span className="block text-[11.5px] text-ink-400">{hint}</span>}
      </span>
      <button type="button" disabled={disabled} onClick={() => onChange(!checked)} className={clsx('relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition', checked ? 'bg-sky-500' : 'bg-white/15')}>
        <span className={clsx('absolute top-0.5 h-4 w-4 rounded-full bg-white transition', checked ? 'left-[18px]' : 'left-0.5')} />
      </button>
    </label>
  )
}

export function Slider({ label, value, min, max, step, onChange, format, disabled }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string; disabled?: boolean }) {
  return (
    <div className={clsx(disabled && 'opacity-50')}>
      <div className="mb-1 flex justify-between text-[12.5px]"><span className="text-ink-300">{label}</span><span className="num text-white">{format ? format(value) : value}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  )
}

export function Segmented<T extends string | number>({ options, value, onChange }: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-xl bg-white/5 p-1 ring-1 ring-white/10">
      {options.map((o) => (
        <button key={String(o.value)} onClick={() => onChange(o.value)} className={clsx('rounded-lg px-3 py-1 text-[12px] font-medium transition', o.value === value ? 'bg-sky-500 text-ink-950' : 'text-ink-300 hover:text-white')}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export const fmtBytes = (b: number) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b > 1e3 ? `${(b / 1e3).toFixed(1)} KB` : `${b} B`)

export const ZONE_COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#f472b6', '#fbbf24', '#fb923c', '#60a5fa', '#2dd4bf']
export const zoneName = (z: number) => `Zone ${String.fromCharCode(65 + z)}`

/** congestion index = flow / normal peak (95th pct) */
export function congestionColor(ratio: number) {
  if (ratio < 0.55) return '#34d399'
  if (ratio < 0.8) return '#a3e635'
  if (ratio < 0.95) return '#fbbf24'
  if (ratio < 1.05) return '#fb923c'
  return '#f87171'
}
export function congestionLabel(ratio: number) {
  if (ratio < 0.55) return 'Free flow'
  if (ratio < 0.8) return 'Moderate'
  if (ratio < 0.95) return 'Busy'
  if (ratio < 1.05) return 'Heavy'
  return 'Severe'
}
