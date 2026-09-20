"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BottomNav } from "@/components/bottom-nav";
import { GamePrediction, Target } from "@/lib/types";

const TYPE_TABS: { key: Target; label: string }[] = [
  { key: "winner", label: "Ganador" },
  { key: "spread", label: "Spread" },
  { key: "total", label: "Total" },
];

async function fetchPredictions(target: Target, week?: number) {
  const params = new URLSearchParams({ target });
  if (week) params.set("week", String(week));
  const res = await fetch(`/api/predictions?${params}`);
  if (!res.ok) throw new Error("No se pudo cargar resultados");
  const data = await res.json();
  return data.predictions as GamePrediction[];
}

export default function ResultadosPage() {
  const [type, setType] = useState<Target>("winner");
  const [week, setWeek] = useState<number | undefined>(undefined);

  const { data, isLoading } = useQuery({
    queryKey: ["resultados", type, week],
    queryFn: () => fetchPredictions(type, week),
  });

  const resolved = useMemo(() => {
    if (!data) return [];
    return data
      .filter((g) => g.home_score != null && g.away_score != null)
      .sort((a, b) => new Date(b.kickoff_utc).getTime() - new Date(a.kickoff_utc).getTime());
  }, [data]);

  const summary = useMemo(() => {
    if (resolved.length === 0) return null;
    if (type === "winner") {
      const hits = resolved.filter(
        (g) => (g.probability ?? 0) >= 0.5 === (g.home_score! > g.away_score!)
      );
      const accuracy = hits.length / resolved.length;
      const brier =
        resolved.reduce((acc, g) => {
          const outcome = g.home_score! > g.away_score! ? 1 : 0;
          const p = g.probability ?? 0;
          return acc + (p - outcome) ** 2;
        }, 0) / resolved.length;
      return [
        { label: "Aciertos", value: `${(accuracy * 100).toFixed(1)}%` },
        { label: "Brier", value: brier.toFixed(4) },
        { label: "Partidos", value: String(resolved.length) },
      ];
    }
    const actualFor = (g: GamePrediction) =>
      type === "spread" ? g.actual_margin : g.actual_total;
    const errors = resolved
      .filter((g) => actualFor(g) != null)
      .map((g) => (g.point_estimate ?? 0) - (actualFor(g) as number));
    const mae = errors.reduce((a, b) => a + Math.abs(b), 0) / errors.length;
    const bias = errors.reduce((a, b) => a + b, 0) / errors.length;
    return [
      { label: "MAE", value: mae.toFixed(2) },
      { label: "Sesgo", value: `${bias >= 0 ? "+" : ""}${bias.toFixed(2)}` },
      { label: "Partidos", value: String(resolved.length) },
    ];
  }, [resolved, type]);

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <div className="mb-[18px] flex items-baseline justify-between">
          <span className="text-[16px] font-bold">Resultados y aciertos</span>
          <select
            className="bg-transparent text-right text-[12px]"
            style={{ color: "var(--text-2)", colorScheme: "dark" }}
            value={week ?? ""}
            onChange={(e) => setWeek(e.target.value ? Number(e.target.value) : undefined)}
          >
            <option value="">Todas las semanas</option>
            {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
              <option key={w} value={w}>Semana {w}</option>
            ))}
          </select>
        </div>
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

      <div className="flex-1 px-[22px]">
        {isLoading && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--text-2)" }}>Cargando…</p>
        )}
        {!isLoading && resolved.length === 0 && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--text-2)" }}>
            Todavía no hay partidos jugados con predicción registrada.
          </p>
        )}

        {summary && (
          <div className="flex gap-[22px] py-5" style={{ borderBottom: "1px solid var(--hair)" }}>
            {summary.map((s) => (
              <div key={s.label} className="flex flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>{s.label}</span>
                <span className="text-[18px] font-bold tabular-nums">{s.value}</span>
              </div>
            ))}
          </div>
        )}

        {resolved.map((g) => {
          const homeWon = g.home_score! > g.away_score!;
          const hit =
            type === "winner"
              ? (g.probability ?? 0) >= 0.5 === homeWon
              : null;
          const actual = type === "spread" ? g.actual_margin : type === "total" ? g.actual_total : null;
          return (
            <div key={g.game_id} className="flex items-center justify-between gap-[10px] py-4" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
              <div>
                <div className="text-[14px] font-semibold">{g.away_team} @ {g.home_team}</div>
                <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
                  Sem {g.week} · {g.away_score}–{g.home_score}
                </div>
              </div>
              <div className="shrink-0 text-right">
                {type === "winner" ? (
                  <span className="text-[13px] font-semibold" style={{ color: hit ? "var(--text)" : "var(--side-b)" }}>
                    {hit ? "Acertó" : "Falló"}
                  </span>
                ) : (
                  <>
                    <div className="text-[13px] font-semibold tabular-nums">
                      pred {(g.point_estimate ?? 0).toFixed(1)}
                    </div>
                    <div className="text-[11px] tabular-nums" style={{ color: "var(--text-3)" }}>
                      real {actual ?? "—"}
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
      <BottomNav active="resultados" />
    </div>
  );
}
