"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BottomNav } from "@/components/bottom-nav";
import { STAT_DEFS, PropPrediction } from "@/lib/types";

async function fetchProps(target: string) {
  const params = new URLSearchParams({ target });
  const res = await fetch(`/api/props?${params}`);
  if (!res.ok) throw new Error("No se pudo cargar props");
  const data = await res.json();
  return data.props as PropPrediction[];
}

export default function PropsPage() {
  const [stat, setStat] = useState("receiving_yards");
  const [selectedGsis, setSelectedGsis] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["props", stat],
    queryFn: () => fetchProps(stat),
  });

  const upcoming = useMemo(() => {
    if (!data) return [];
    const now = Date.now();
    const future = data.filter((p) => new Date(p.kickoff_utc).getTime() >= now);
    const pool = future.length > 0 ? future : data;
    return [...pool].sort((a, b) => (b.point_estimate ?? 0) - (a.point_estimate ?? 0)).slice(0, 12);
  }, [data]);

  const statDef = STAT_DEFS.find((s) => s.key === stat)!;
  const selected = upcoming.find((p) => p.gsis_id === selectedGsis) ?? upcoming[0] ?? null;

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <div className="mb-[18px] flex items-baseline justify-between">
          <span className="text-[16px] font-bold">Props</span>
          <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
            {upcoming[0]?.season ?? new Date().getFullYear()} · próximos partidos
          </span>
        </div>
        <div className="flex gap-[18px] overflow-x-auto border-b" style={{ borderColor: "var(--hair)" }}>
          {STAT_DEFS.map((s) => {
            const active = s.key === stat;
            return (
              <button
                key={s.key}
                onClick={() => { setStat(s.key); setSelectedGsis(null); }}
                className="relative shrink-0 whitespace-nowrap pb-3 text-[12.5px]"
                style={{ color: active ? "var(--text)" : "var(--text-3)", fontWeight: active ? 600 : 500 }}
              >
                {s.label}
                {active && (
                  <span className="absolute -bottom-px left-0 right-0 h-[2px]" style={{ background: "var(--accent)" }} />
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 px-[22px]">
        {statDef.sparse && (
          <div
            className="my-4 py-2 pl-3 text-[11.5px] leading-relaxed"
            style={{ color: "var(--accent)", borderLeft: "2px solid var(--accent)" }}
          >
            Conteo poco frecuente — la mayoría de partidos son 0. La proyección es una aproximación más basta que en yardas continuas.
          </div>
        )}

        {isLoading && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--text-2)" }}>Cargando…</p>
        )}

        <div className="mb-1 mt-[22px] text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
          Jugadores destacados
        </div>
        {upcoming.map((p) => (
          <button
            key={`${p.game_id}-${p.gsis_id}`}
            onClick={() => setSelectedGsis(p.gsis_id)}
            className="flex w-full items-baseline justify-between gap-[10px] py-[14px] text-left"
            style={{ borderBottom: "1px solid var(--hair-soft)" }}
          >
            <div>
              <div className="text-[14.5px] font-semibold">{p.player_name}</div>
              <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
                {p.team} · {p.position}
              </div>
            </div>
            <div className="shrink-0 text-right">
              <div className="text-[18px] font-bold tabular-nums">{(p.point_estimate ?? 0).toFixed(1)}</div>
              <div className="text-[10.5px] tabular-nums" style={{ color: "var(--text-3)" }}>
                rango {(p.lower_80 ?? 0).toFixed(0)}–{(p.upper_80 ?? 0).toFixed(0)}
              </div>
            </div>
          </button>
        ))}

        {selected && (
          <>
            <div className="mb-1 mt-[22px] text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
              Detalle — {selected.player_name}
            </div>
            <div className="mt-2 p-4" style={{ border: "1px solid var(--hair)" }}>
              <div className="flex items-baseline justify-between">
                <div>
                  <div className="text-[15px] font-semibold">{selected.player_name}</div>
                  <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
                    {selected.team} · {selected.position} · vs {selected.team === selected.home_team ? selected.away_team : selected.home_team}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[22px] font-bold tabular-nums">{(selected.point_estimate ?? 0).toFixed(1)}</div>
                  <div className="text-[10.5px] tabular-nums" style={{ color: "var(--text-3)" }}>
                    rango {(selected.lower_80 ?? 0).toFixed(0)}–{(selected.upper_80 ?? 0).toFixed(0)}
                  </div>
                </div>
              </div>
              {selected.explanation_es && (
                <p className="mt-3 text-[12.5px] leading-relaxed" style={{ color: "var(--text-2)" }}>
                  {selected.explanation_es}
                </p>
              )}
              <div className="mt-3 pt-[10px] text-[10.5px] leading-loose" style={{ color: "var(--text-3)", borderTop: "1px solid var(--hair)" }}>
                as-of {selected.as_of_ts ? new Date(selected.as_of_ts).toLocaleString("es-MX") : "—"}
              </div>
            </div>
          </>
        )}
      </div>
      <BottomNav active="props" />
    </div>
  );
}
