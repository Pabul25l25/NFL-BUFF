"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BottomNav } from "@/components/bottom-nav";
import { Target } from "@/lib/types";

const TYPE_TABS: { key: Target; label: string }[] = [
  { key: "winner", label: "Ganador" },
  { key: "spread", label: "Spread" },
  { key: "total", label: "Total" },
];

interface PerfRow {
  season: number;
  week: number;
  n_predictions: number;
  log_loss: number | null;
  brier_score: number | null;
  accuracy: number | null;
  mae: number | null;
  version: number | null;
  family: string | null;
}

interface DriftRow {
  feature_name: string;
  psi: number;
  reference_mean: number;
  current_mean: number;
  is_drifted: boolean;
}

async function fetchPerformance(target: Target) {
  const res = await fetch(`/api/production?target=${target}`);
  if (!res.ok) throw new Error("No se pudo cargar rendimiento");
  const data = await res.json();
  return data.performance as PerfRow[];
}

async function fetchDrift() {
  const res = await fetch("/api/drift");
  if (!res.ok) throw new Error("No se pudo cargar drift");
  const data = await res.json();
  return data.features as DriftRow[];
}

function driftLabel(psi: number): { text: string; color: string } {
  if (psi > 0.25) return { text: "severo", color: "var(--side-b)" };
  if (psi > 0.1) return { text: "moderado", color: "var(--accent)" };
  return { text: "estable", color: "var(--text-3)" };
}

export default function ProduccionPage() {
  const [type, setType] = useState<Target>("winner");

  const { data: perf, isLoading } = useQuery({
    queryKey: ["production", type],
    queryFn: () => fetchPerformance(type),
  });
  const { data: drift } = useQuery({ queryKey: ["drift"], queryFn: fetchDrift });

  const metricKey = type === "winner" ? "log_loss" : "mae";
  const chartData = (perf ?? []).map((p) => ({
    label: `S${p.season}·W${p.week}`,
    value: p[metricKey] ?? null,
  }));

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <div className="mb-[18px] text-[16px] font-bold">Rendimiento en producción</div>
        <div className="flex gap-[22px] border-b" style={{ borderColor: "var(--hair)" }}>
          {TYPE_TABS.map((t) => {
            const active = t.key === type;
            return (
              <button
                key={t.key}
                onClick={() => setType(t.key)}
                className="relative pb-3 text-[13px]"
                style={{ color: active ? "var(--text)" : "var(--text-3)", fontWeight: active ? 600 : 500 }}
              >
                {t.label}
                {active && (
                  <span className="absolute -bottom-px left-0 right-0 h-[2px]" style={{ background: "var(--accent)" }} />
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 px-[22px] pt-4">
        {isLoading && <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Cargando…</p>}
        {!isLoading && (perf ?? []).length === 0 && (
          <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Sin histórico de producción todavía.</p>
        )}

        {chartData.length > 0 && (
          <div style={{ height: 160 }} className="mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 8, right: 4, bottom: 0, left: -24 }}>
                <XAxis dataKey="label" tick={{ fontSize: 9, fill: "var(--text-3)" }} axisLine={{ stroke: "var(--hair)" }} tickLine={false} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 9, fill: "var(--text-3)" }} axisLine={false} tickLine={false} width={32} />
                <Tooltip
                  contentStyle={{ background: "var(--surface)", border: "1px solid var(--hair)", fontSize: 11 }}
                  labelStyle={{ color: "var(--text-2)" }}
                />
                <Line type="monotone" dataKey="value" stroke="var(--accent)" strokeWidth={1.5} dot={{ r: 2, fill: "var(--accent)" }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {(perf ?? []).slice().reverse().slice(0, 10).map((p) => (
          <div key={`${p.season}-${p.week}`} className="flex items-center justify-between py-3" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
            <div>
              <div className="text-[13px] font-semibold">Semana {p.week} · {p.season}</div>
              <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
                {p.family ?? "—"} v{p.version ?? "—"} · {p.n_predictions} predicciones
              </div>
            </div>
            <span className="text-[14px] font-semibold tabular-nums">
              {(p[metricKey] ?? 0).toFixed(metricKey === "log_loss" ? 4 : 2)}
            </span>
          </div>
        ))}

        <div className="mb-1 mt-8 text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
          Drift de features
        </div>
        <p className="mb-2 text-[11.5px] leading-relaxed" style={{ color: "var(--text-2)" }}>
          PSI &lt;0.10 estable · 0.10–0.25 moderado · &gt;0.25 severo.
        </p>
        {(drift ?? []).slice(0, 15).map((d) => {
          const s = driftLabel(d.psi);
          return (
            <div key={d.feature_name} className="flex items-center justify-between py-3" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
              <span className="text-[12.5px]">{d.feature_name}</span>
              <span className="text-[12px] font-semibold tabular-nums" style={{ color: s.color }}>
                {d.psi.toFixed(3)} · {s.text}
              </span>
            </div>
          );
        })}
      </div>
      <BottomNav active="produccion" />
    </div>
  );
}
