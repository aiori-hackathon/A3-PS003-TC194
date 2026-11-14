import { Server } from 'lucide-react'
import type { ClientState } from '../engine/federated'
import { ZONE_COLORS, zoneName } from '../ui'

export type Phase = 'idle' | 'broadcast' | 'train' | 'upload' | 'aggregate'

export function Topology({ clients, phase, round, secure, dp }: { clients: ClientState[]; phase: Phase; round: number; secure: boolean; dp: boolean }) {
  const W = 620, H = 470, cx = W / 2, cy = H / 2, R = 180
  const pos = clients.map((_, i) => {
    const a = (i / clients.length) * Math.PI * 2 - Math.PI / 2
    return [cx + R * Math.cos(a), cy + R * Math.sin(a)]
  })
  const caption: Record<Phase, string> = {
    idle: 'Press Start to begin federated training.',
    broadcast: `Round ${round}: server broadcasts the current global model to every zone.`,
    train: `Round ${round}: each zone trains on its own private sensor data. Nothing leaves the zone.`,
    upload: `Round ${round}: zones upload only model updates${dp ? ' (clipped + noised)' : ''}${secure ? ', masked by secure aggregation' : ''}.`,
    aggregate: `Round ${round}: server averages the updates into a new global model (FedAvg).`,
  }
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 470 }}>
        <defs>
          <radialGradient id="srv"><stop offset="0" stopColor="#38bdf8" stopOpacity=".35" /><stop offset="1" stopColor="#38bdf8" stopOpacity="0" /></radialGradient>
        </defs>
        <circle cx={cx} cy={cy} r={110} fill="url(#srv)" />
        {clients.map((c, i) => {
          const [x, y] = pos[i]
          const dropped = c.status === 'dropped'
          const active = !dropped && (phase === 'broadcast' || phase === 'upload')
          return (
            <g key={c.zone}>
              <line x1={cx} y1={cy} x2={x} y2={y} stroke={dropped ? '#334155' : ZONE_COLORS[c.zone]} strokeOpacity={dropped ? 0.4 : 0.45} strokeWidth={2}
                className={active ? 'flowline' : ''} strokeDasharray={dropped ? '2 6' : active ? undefined : '0'} />
              {phase === 'broadcast' && !dropped && (
                <circle r={5} fill="#7dd3fc">
                  <animateMotion dur="0.9s" repeatCount="1" fill="freeze" path={`M${cx},${cy} L${x},${y}`} />
                </circle>
              )}
              {phase === 'upload' && !dropped && (
                <g>
                  <circle r={6} fill={ZONE_COLORS[c.zone]}>
                    <animateMotion dur="1s" repeatCount="1" fill="freeze" path={`M${x},${y} L${cx},${cy}`} />
                  </circle>
                </g>
              )}
            </g>
          )
        })}
        {/* server */}
        <g>
          <circle cx={cx} cy={cy} r={46} fill="#0e1530" stroke="#38bdf8" strokeWidth={2} />
          {phase === 'aggregate' && <circle cx={cx} cy={cy} r={54} fill="none" stroke="#a78bfa" strokeWidth={3} strokeDasharray="30 200" className="spin" />}
          <foreignObject x={cx - 14} y={cy - 24} width={28} height={28}><Server size={28} color="#7dd3fc" /></foreignObject>
          <text x={cx} y={cy + 20} textAnchor="middle" fill="#cdd5f3" fontSize={11} fontWeight={600}>Aggregator</text>
          <text x={cx} y={cy + 33} textAnchor="middle" fill="#7a87b8" fontSize={10}>global v{round}</text>
        </g>
        {clients.map((c, i) => {
          const [x, y] = pos[i]
          const dropped = c.status === 'dropped'
          const training = c.status === 'training'
          return (
            <g key={`n${c.zone}`} opacity={dropped ? 0.4 : 1}>
              {training && <circle cx={x} cy={y} r={14} fill="none" stroke={ZONE_COLORS[c.zone]} className="pulse" />}
              <circle cx={x} cy={y} r={30} fill="#0e1530" stroke={ZONE_COLORS[c.zone]} strokeWidth={2.5} />
              <text x={x} y={y - 2} textAnchor="middle" fill="#fff" fontSize={12} fontWeight={700}>{zoneName(c.zone).replace('Zone ', 'Z-')}</text>
              <text x={x} y={y + 12} textAnchor="middle" fill="#7a87b8" fontSize={9.5}>{dropped ? 'offline' : c.status === 'idle' ? `${c.sensors.length} sensors` : c.loss !== null ? `loss ${c.loss.toFixed(3)}` : '…'}</text>
              {/* private data cylinder */}
              <g transform={`translate(${x + (x > cx ? 34 : -58)},${y - 12})`}>
                <rect width={24} height={22} rx={4} fill="#131c3d" stroke="#ffffff30" />
                <text x={12} y={15} textAnchor="middle" fontSize={11}>🔒</text>
              </g>
            </g>
          )
        })}
      </svg>
      <p className="mt-1 min-h-[20px] text-center text-[13px] text-ink-300">{caption[phase]}</p>
    </div>
  )
}
