import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Assets } from '../engine/data'
import { Badge, Card, Stat, fmtBytes, zoneName } from '../ui'

const COLORS: Record<string, string> = {
  'FedAvg · GRU': '#38bdf8', 'FedProx · GRU': '#0ea5e9', 'Centralised · GRU': '#64748b', 'Local-only · GRU': '#f472b6',
  'FedAvg · TemporalMLP': '#a78bfa', 'FedProx · TemporalMLP': '#8b5cf6', 'Centralised · TemporalMLP': '#475569', 'Local-only · TemporalMLP': '#fb7185',
  'Historical average': '#f87171', 'Last value (persistence)': '#fbbf24',
}

export default function Benchmarks({ a }: { a: Assets }) {
  const r = a.results
  const runs = Object.entries(r.runs).sort((x, y) => x[1].metrics.avg.mae - y[1].metrics.avg.mae)
  const fed = r.runs['FedAvg · GRU'].metrics.avg.mae
  const cen = r.runs['Centralised · GRU'].metrics.avg.mae
  const ha = r.runs['Historical average'].metrics.avg.mae
  const curves = r.curves['FedAvg · GRU'].map((c, i) => ({
    round: c.round, 'FedAvg · GRU': c.val_mae, 'FedProx · GRU': r.curves['FedProx · GRU']?.[i]?.val_mae, 'FedAvg · TemporalMLP': r.curves['FedAvg · TemporalMLP']?.[i]?.val_mae,
  }))
  const zones = r.per_zone['GRU']
  const com = r.communication
  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-[28px] font-semibold tracking-tight text-white">Benchmarks</h1>
        <p className="mt-1 max-w-3xl text-[14px] text-ink-300">
          Measured offline with the Python pipeline on {r.dataset.name}: {r.dataset.sensors} sensors, {r.dataset.days} days at {r.dataset.interval_min}-minute
          resolution. Train {r.dataset.train_days} days → validate {r.dataset.val_days} → test on the last {r.dataset.test_days} days, never seen in training.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Federated GRU MAE" value={fed.toFixed(2)} unit="veh/5min" tone="brand" hint="average over 5/15/30/60-min horizons" />
        <Stat label="vs centralised (pooled data)" value={fed <= cen ? `${(((cen - fed) / cen) * 100).toFixed(1)}% better` : `+${(((fed - cen) / cen) * 100).toFixed(1)}%`} tone="good" hint={fed <= cen ? 'no accuracy lost by keeping data private' : 'small cost of keeping data private'} />
        <Stat label="Better than historical avg" value={`${(((ha - fed) / ha) * 100).toFixed(0)}%`} tone="good" hint="today's common practice" />
        <Stat label="Less data moved" value={`${(com.raw_data_bytes / com.total_upload_bytes).toFixed(1)}×`} tone="good" hint={`${fmtBytes(com.total_upload_bytes)} uploaded vs ${fmtBytes(com.raw_data_bytes)} raw`} />
      </div>

      <Card title="All methods on the test period" subtitle="MAE in vehicles per 5 minutes, RMSE, and MAPE (%), per forecast horizon. Sorted by average MAE.">
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="text-left text-ink-400">
                <th className="pb-2 font-medium">Method</th><th className="font-medium">Keeps data private</th>
                {r.horizons_min.map((h) => <th key={h} className="text-right font-medium">{h} min MAE / RMSE / MAPE</th>)}
                <th className="text-right font-medium">Average MAE</th>
              </tr>
            </thead>
            <tbody>
              {runs.map(([name, v]) => (
                <tr key={name} className="border-t border-white/5">
                  <td className="py-2 font-medium text-white"><span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: COLORS[name] ?? '#94a3b8' }} />{name}</td>
                  <td>{v.private ? <Badge tone="good">yes</Badge> : <Badge tone="bad">no – pools raw data</Badge>}</td>
                  {r.horizons_min.map((h) => {
                    const m = v.metrics[`${h}min`]
                    return <td key={h} className="num text-right text-ink-300">{m.mae.toFixed(1)} / {m.rmse.toFixed(1)} / {m.mape.toFixed(1)}%</td>
                  })}
                  <td className="num text-right font-semibold text-white">{v.metrics.avg.mae.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Average MAE by method" subtitle="Lower is better.">
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={runs.map(([n, v]) => ({ n: n.replace(' (persistence)', ''), mae: +v.metrics.avg.mae.toFixed(2) }))} layout="vertical" margin={{ left: 70, right: 20 }}>
                <CartesianGrid horizontal={false} />
                <XAxis type="number" tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="n" tickLine={false} axisLine={false} width={150} />
                <Tooltip />
                <Bar dataKey="mae" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {runs.map(([n]) => <Cell key={n} fill={COLORS[n] ?? '#94a3b8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Convergence" subtitle="Validation MAE after each federated round.">
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={curves} margin={{ left: -10, right: 10 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="round" tickLine={false} axisLine={false} />
                <YAxis domain={[16, 30]} tickLine={false} axisLine={false} allowDataOverflow />
                <Tooltip />
                <Legend />
                {['FedAvg · GRU', 'FedProx · GRU', 'FedAvg · TemporalMLP'].map((k) => <Line key={k} dataKey={k} stroke={COLORS[k]} dot={false} strokeWidth={2} isAnimationActive={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {r.scarce && (
        <Card title="Where federation pays off most: new deployments with little history" subtitle="Every zone has only the last few days of data (e.g. newly installed sensors). One shared federated model vs each zone training alone.">
          <div className="grid gap-5 md:grid-cols-[1fr_1.4fr]">
            <div className="grid grid-cols-2 gap-3">
              {r.scarce.map((s) => (
                <Stat key={s.days} label={`Error cut · ${s.days} days of history`} value={`${s.gain_pct.toFixed(1)}%`} tone={s.gain_pct > 0 ? 'good' : 'warn'} hint={`error: local ${s.local_mae.toFixed(1)} → federated ${s.fed_mae.toFixed(1)}`} />
              ))}
              {r.unseen_zone && (
                <div className="col-span-2 rounded-xl bg-white/[0.03] p-4 ring-1 ring-white/5 text-[12.5px] text-ink-300">
                  <div className="mb-1 font-semibold text-white">Cold start: a brand-new zone with zero history</div>
                  <div className="mb-1.5">The federated model, trained without the zone, is as accurate on day one as a model the zone could only build after a week of collecting data.</div>
                  {r.unseen_zone.map((u) => (
                    <div key={u.zone}>{zoneName(u.zone)} ({u.sensors} sensors): day-one federated MAE <span className="num text-sky-300">{u.fed_unseen_mae.toFixed(1)}</span> · own model after 7 days <span className="num text-pink-300">{u.local_7day_mae.toFixed(1)}</span></div>
                  ))}
                </div>
              )}
            </div>
            <div className="h-64">
              <ResponsiveContainer>
                <BarChart data={r.scarce[0].zones.map((z, i) => ({ zone: zoneName(z.zone), local: z.local_mae, fed: z.fed_mae, local7: r.scarce![1]?.zones[i]?.local_mae, fed7: r.scarce![1]?.zones[i]?.fed_mae }))} margin={{ left: -10, right: 8 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="zone" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="local" name={`Local-only (${r.scarce[0].days} days)`} fill="#f472b6" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  <Bar dataKey="fed" name={`Federated (${r.scarce[0].days} days)`} fill="#38bdf8" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>
      )}

      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Per zone with full history: local-only vs federated" subtitle="With 43 days of data each zone can already train a decent model alone; federation matches it with one shared model.">
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={zones.map((z) => ({ zone: `${zoneName(z.zone)} (${z.sensors})`, local: z.local_mae, fed: z.fed_mae }))} margin={{ left: -10, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="zone" tickLine={false} axisLine={false} fontSize={10} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="local" name="Local-only GRU" fill="#f472b6" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="fed" name="Federated GRU" fill="#38bdf8" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Robustness to zones dropping out" subtitle="Probability that a zone misses a round, and the resulting test MAE.">
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={r.dropout.map((d) => ({ d: `${Math.round(d.dropout * 100)}%`, mae: d.mae, clients: d.avg_clients }))} margin={{ left: -10, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="d" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} domain={[0, 'auto']} />
                <Tooltip />
                <Bar dataKey="mae" name="Test MAE" fill="#34d399" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                  <LabelList dataKey="mae" position="top" fill="#cbd5e1" fontSize={12} formatter={(v: unknown) => Number(v).toFixed(2)} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <p className="text-[12px] text-ink-400">Reproduce: <code className="rounded bg-white/5 px-1.5 py-0.5">cd ml && python run_experiments.py && python run_scenarios.py</code> · total runtime {Math.round(r.runtime_sec / 60)} min on a laptop CPU · pure NumPy, no GPU.</p>
    </div>
  )
}
