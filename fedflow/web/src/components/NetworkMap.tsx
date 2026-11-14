import { useMemo, useState } from 'react'
import type { Network } from '../engine/data'
import { ZONE_COLORS, congestionColor, congestionLabel, zoneName } from '../ui'

export type NodeValue = { ratio: number; flow: number; forecast?: number }

export function NetworkMap({ net, values, highlightZone, height = 460, onSelect, selected }: {
  net: Network
  values: (NodeValue | null)[]
  highlightZone?: number | null
  height?: number
  onSelect?: (id: number) => void
  selected?: number | null
}) {
  const W = 1000, H = 600, pad = 40
  const pos = useMemo(() => net.sensors.map((s) => [pad + s.x * (W - 2 * pad), pad + s.y * (H - 2 * pad)]), [net])
  const [hover, setHover] = useState<number | null>(null)
  const hv = hover !== null ? values[hover] : null
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ height, width: '100%' }} className="block">
        <defs>
          <radialGradient id="nodeGlow"><stop offset="0%" stopColor="#fff" stopOpacity="0.5" /><stop offset="100%" stopColor="#fff" stopOpacity="0" /></radialGradient>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#ffffff08" /></pattern>
        </defs>
        <rect width={W} height={H} fill="url(#grid)" />
        {net.zones.map((z) => (
          <circle key={z.zone} cx={pad + z.cx * (W - 2 * pad)} cy={pad + z.cy * (H - 2 * pad)} r={70 + Math.sqrt(z.sensors.length) * 16}
            fill={ZONE_COLORS[z.zone]} opacity={highlightZone === null || highlightZone === undefined ? 0.05 : highlightZone === z.zone ? 0.14 : 0.02} />
        ))}
        {net.edges.map(([a, b], i) => {
          const v = values[a], w = values[b]
          const r = Math.max(v?.ratio ?? 0, w?.ratio ?? 0)
          return <line key={i} x1={pos[a][0]} y1={pos[a][1]} x2={pos[b][0]} y2={pos[b][1]} stroke={congestionColor(r)} strokeOpacity={0.35} strokeWidth={2.2} />
        })}
        {net.sensors.map((s) => {
          const v = values[s.id]
          const dim = highlightZone !== null && highlightZone !== undefined && s.zone !== highlightZone
          const big = selected === s.id || hover === s.id
          return (
            <g key={s.id} opacity={dim ? 0.25 : 1} onMouseEnter={() => setHover(s.id)} onMouseLeave={() => setHover(null)} onClick={() => onSelect?.(s.id)} style={{ cursor: onSelect ? 'pointer' : 'default' }}>
              {v && v.ratio >= 1.05 && <circle cx={pos[s.id][0]} cy={pos[s.id][1]} r={14} fill="none" stroke="#f87171" className="pulse" />}
              <circle cx={pos[s.id][0]} cy={pos[s.id][1]} r={big ? 9 : 6} fill={v ? congestionColor(v.ratio) : '#334155'} stroke={ZONE_COLORS[s.zone]} strokeWidth={big ? 3 : 1.6} />
            </g>
          )
        })}
        {net.zones.map((z) => (
          <text key={z.zone} x={pad + z.cx * (W - 2 * pad)} y={pad + z.cy * (H - 2 * pad) - 54 - Math.sqrt(z.sensors.length) * 6}
            textAnchor="middle" fill={ZONE_COLORS[z.zone]} fontSize={15} fontWeight={700} opacity={0.9} style={{ fontFamily: 'Space Grotesk' }}>
            {zoneName(z.zone)}
          </text>
        ))}
      </svg>
      {hover !== null && hv && (
        <div className="pointer-events-none absolute left-3 top-3 rounded-xl border border-white/10 bg-ink-900/95 px-3 py-2 text-[12px] shadow-xl">
          <div className="font-semibold text-white">Sensor {hover} · {zoneName(net.sensors[hover].zone)}</div>
          <div className="text-ink-300">Flow <span className="num text-white">{Math.round(hv.flow)}</span> veh/5 min · <span style={{ color: congestionColor(hv.ratio) }}>{congestionLabel(hv.ratio)}</span></div>
          {hv.forecast !== undefined && <div className="text-ink-300">Forecast <span className="num text-white">{Math.round(hv.forecast)}</span> veh/5 min</div>}
          <div className="text-ink-400">Normal peak (p95): <span className="num">{Math.round(net.sensors[hover].p95)}</span></div>
        </div>
      )}
      <div className="absolute bottom-2 right-3 flex items-center gap-3 rounded-lg bg-ink-900/80 px-3 py-1.5 text-[11px] text-ink-300">
        {[0.4, 0.7, 0.9, 1.0, 1.1].map((r) => (
          <span key={r} className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full" style={{ background: congestionColor(r) }} />{congestionLabel(r)}</span>
        ))}
      </div>
    </div>
  )
}
