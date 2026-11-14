import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ReferenceLine } from 'recharts'
import { Check, X, ShieldCheck, Lock, EyeOff, KeyRound, Server } from 'lucide-react'
import type { Assets } from '../engine/data'
import { Badge, Card } from '../ui'

const FLOWS = [
  { what: 'Raw sensor readings (flow, occupancy, speed)', leaves: false, how: 'Stay on the zone server. Used only for local training.' },
  { what: 'Per-vehicle or per-location records', leaves: false, how: 'Never collected by the model; loop detectors already aggregate counts.' },
  { what: 'Normalisation statistics', leaves: true, how: 'Only three aggregate numbers per zone: sum, sum of squares and count.' },
  { what: 'Model updates (Δ weights)', leaves: true, how: '≈14 KB per round, clipped, optionally noised (DP), masked by secure aggregation, sent over TLS.' },
  { what: 'Global model', leaves: true, how: 'Broadcast back to every zone; contains no data, only learned patterns.' },
]

const THREATS = [
  { threat: 'Curious aggregator reads one zone\'s update', defence: 'Secure aggregation: pairwise masks hide every individual update; only the sum is revealed.', icon: Lock },
  { threat: 'Inferring a zone\'s traffic from the model (membership / inversion)', defence: 'Update clipping + Gaussian noise give a formal (ε, δ) guarantee, tracked by an RDP accountant.', icon: EyeOff },
  { threat: 'Malicious or faulty zone sends a huge update', defence: 'L2 clipping bounds any one zone\'s influence; FedProx keeps local models close to the global one.', icon: ShieldCheck },
  { threat: 'Zone server goes offline mid-round', defence: 'Rounds continue with the zones that respond; tested with up to 60 % dropout.', icon: Server },
  { threat: 'Eavesdropping on the network', defence: 'All exchanges over TLS; zones authenticate with per-client keys.', icon: KeyRound },
]

export default function Privacy({ a }: { a: Assets }) {
  const r = a.results
  const noDP = r.runs['FedAvg · GRU']?.metrics.avg.mae
  const pts8 = r.privacy.filter((p) => p.epsilon).map((p) => ({ eps: p.epsilon!, mae: p.mae, sigma: p.sigma }))
  const ptsScale = (r.privacy_scale ?? []).filter((p) => p.epsilon).map((p) => ({ eps: p.epsilon!, mae: p.mae, sigma: p.sigma }))
  const clients = r.privacy_scale?.[0]?.clients
  // secure-aggregation worked example
  const upd = [0.42, -0.17, 0.08]
  const m12 = 5.31, m13 = -2.77, m23 = 8.04
  const masked = [upd[0] + m12 + m13, upd[1] - m12 + m23, upd[2] - m13 - m23]

  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-[28px] font-semibold tracking-tight text-white">Privacy & security by design</h1>
        <p className="mt-1 max-w-3xl text-[14px] text-ink-300">
          Traffic data belongs to different agencies: city corporations, highway authorities, toll operators, police. FedFlow lets them
          build one shared forecasting model without handing their data to each other or to a central server.
        </p>
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.2fr_1fr]">
        <Card title="What leaves a zone, and what never does">
          <table className="w-full text-[13px]">
            <tbody>
              {FLOWS.map((f) => (
                <tr key={f.what} className="border-t border-white/5 first:border-0">
                  <td className="py-2.5 pr-3 align-top">
                    {f.leaves ? <Badge tone="warn"><Check size={11} />shared</Badge> : <Badge tone="good"><X size={11} />never</Badge>}
                  </td>
                  <td className="py-2.5 pr-3 align-top font-medium text-white">{f.what}</td>
                  <td className="py-2.5 align-top text-ink-300">{f.how}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="Secure aggregation in three numbers" subtitle="Three zones, one weight each. Every pair agrees on a random mask; one adds it, the other subtracts it.">
          <div className="space-y-2 font-mono text-[12.5px]">
            {['A', 'B', 'C'].map((z, i) => (
              <div key={z} className="grid grid-cols-[70px_1fr_1fr] items-center gap-2">
                <span className="text-ink-300">Zone {z}</span>
                <span className="rounded-md bg-emerald-400/10 px-2 py-1 text-emerald-200">real {upd[i] >= 0 ? '+' : ''}{upd[i].toFixed(2)}</span>
                <span className="rounded-md bg-amber-400/10 px-2 py-1 text-amber-200">sent {masked[i] >= 0 ? '+' : ''}{masked[i].toFixed(2)}</span>
              </div>
            ))}
            <div className="mt-2 grid grid-cols-[70px_1fr_1fr] items-center gap-2 border-t border-white/10 pt-2">
              <span className="text-ink-300">Sum</span>
              <span className="rounded-md bg-emerald-400/10 px-2 py-1 text-emerald-200">{upd.reduce((x, y) => x + y, 0).toFixed(2)}</span>
              <span className="rounded-md bg-sky-400/10 px-2 py-1 text-sky-200">{masked.reduce((x, y) => x + y, 0).toFixed(2)} ✓ identical</span>
            </div>
          </div>
          <p className="mt-3 text-[12.5px] text-ink-400">Masks: A↔B {m12}, A↔C {m13}, B↔C {m23}. The server sees only the amber numbers, which look random, yet their sum is exact. In production the masks come from Diffie–Hellman key agreement (Bonawitz et al., 2017).</p>
        </Card>
      </div>

      <Card title="Privacy–accuracy trade-off (measured)" subtitle={`Differential privacy on the GRU, FedAvg. Lower ε = stronger privacy. Dashed line = no DP (MAE ${noDP?.toFixed(2)}).`}>
        <div className="h-72">
          <ResponsiveContainer>
            <ScatterChart margin={{ left: -6, right: 20, top: 10 }}>
              <CartesianGrid />
              <XAxis dataKey="eps" type="number" scale="log" domain={['auto', 'auto']} name="ε" tickLine={false} axisLine={false} label={{ value: 'privacy budget ε (log scale)', position: 'insideBottom', offset: -2, fill: '#7a87b8', fontSize: 11 }} />
              <YAxis dataKey="mae" type="number" name="MAE" tickLine={false} axisLine={false} domain={[0, 'auto']} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              {noDP && <ReferenceLine y={noDP} stroke="#38bdf8" strokeDasharray="5 5" />}
              <Scatter name="8 zone clients" data={pts8} fill="#f472b6" line />
              {ptsScale.length > 0 && <Scatter name={`${clients} junction-level clients`} data={ptsScale} fill="#a78bfa" line />}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-[12.5px] text-ink-300">
          Honest finding: with only 8 participants, strong formal DP costs a lot of accuracy, because the noise is spread over few updates.
          The noise per participant shrinks as more clients join, so DP becomes practical at junction scale. Our default for a city pilot is
          secure aggregation + clipping, with DP switched on as the network grows.
        </p>
      </Card>

      <Card title="Threat model">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {THREATS.map((t) => (
            <div key={t.threat} className="rounded-xl bg-white/[0.03] p-4 ring-1 ring-white/5">
              <t.icon size={18} className="text-brand-400" />
              <div className="mt-2 text-[13px] font-semibold text-white">{t.threat}</div>
              <div className="mt-1 text-[12.5px] text-ink-300">{t.defence}</div>
            </div>
          ))}
          <div className="rounded-xl bg-white/[0.03] p-4 ring-1 ring-white/5">
            <ShieldCheck size={18} className="text-emerald-400" />
            <div className="mt-2 text-[13px] font-semibold text-white">Aligned with data-protection principles</div>
            <div className="mt-1 text-[12.5px] text-ink-300">Data minimisation and purpose limitation (as in India's DPDP Act 2023): only model parameters are shared, for one stated purpose.</div>
          </div>
        </div>
      </Card>
    </div>
  )
}
