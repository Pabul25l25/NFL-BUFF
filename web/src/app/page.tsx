"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { BottomNav } from "@/components/bottom-nav";
import { TEAM_COLORS, TEAMS } from "@/lib/teams";
import { GamePrediction, Target, displayProb, pickLabel } from "@/lib/types";

const TYPE_TABS: { key: Target; label: string }[] = [
  { key: "winner", label: "Ganador" },
  { key: "spread", label: "Spread" },
  { key: "total", label: "Total" },
];

async function fetchPredictions(target: Target, week?: number, team?: string) {
  const params = new URLSearchParams({ target });
  if (week) params.set("week", String(week));
  if (team) params.set("team", team);
  const res = await fetch(`/api/predictions?${params}`);
  if (!res.ok) throw new Error("No se pudo cargar predicciones");
  const data = await res.json();
  return data.predictions as GamePrediction[];
}

function confClass(conf: string | null): { color: string; weight: number } {
  if (conf === "alta") return { color: "var(--text)", weight: 600 };
  if (conf === "media") return { color: "var(--text-2)", weight: 500 };
  return { color: "var(--text-3)", weight: 500 };
}

export default function PrediccionesPage() {
  const [type, setType] = useState<Target>("winner");
  const [week, setWeek] = useState<number | undefined>(undefined);
  const [team, setTeam] = useState<string | undefined>(undefined);
  const [sheetOpen, setSheetOpen] = useState(false);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["predictions", type, week, team],
    queryFn: () => fetchPredictions(type, week, team),
  });

  const upcoming = useMemo(() => {
    if (!data) return [];
    const now = Date.now();
    const future = data.filter((g) => new Date(g.kickoff_utc).getTime() >= now);
    const pool = future.length > 0 ? future : data;
    return [...pool].sort(
      (a, b) => new Date(a.kickoff_utc).getTime() - new Date(b.kickoff_utc).getTime()
    );
  }, [data]);

  const weekLabel = week ?? upcoming[0]?.week ?? "—";
  const seasonLabel = upcoming[0]?.season ?? new Date().getFullYear();

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[env(safe-area-inset-top,0px)]">
        <div className="flex items-baseline justify-between pt-[22px] pb-[18px]">
          <span className="text-[16px] font-bold tracking-tight">NFL-Predictor</span>
          <button
            onClick={() => setSheetOpen(true)}
            className="border-b border-dotted text-[12px]"
            style={{ color: "var(--text-2)", borderColor: "var(--text-3)" }}
          >
            Semana {weekLabel} · {seasonLabel}
          </button>
        </div>
        <div className="flex gap-[22px] border-b" style={{ borderColor: "var(--hair)" }}>
          {TYPE_TABS.map((t) => {
            const active = t.key === type;
            return (
              <button
                key={t.key}
                onClick={() => setType(t.key)}
                className="relative pb-3 text-[13px]"
                style={{
                  color: active ? "var(--text)" : "var(--text-3)",
                  fontWeight: active ? 600 : 500,
                }}
              >
                {t.label}
                {active && (
                  <span
                    className="absolute -bottom-px left-0 right-0 h-[2px]"
                    style={{ background: "var(--accent)" }}
                  />
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 px-[22px]">
        {isLoading && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--text-2)" }}>
            Cargando predicciones…
          </p>
        )}
        {isError && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--side-b)" }}>
            No se pudieron cargar las predicciones. Intenta de nuevo.
          </p>
        )}
        {!isLoading && !isError && upcoming.length === 0 && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--text-2)" }}>
            No hay predicciones para este filtro.
          </p>
        )}

        {upcoming.map((g) => {
          const conf = confClass(g.confidence_level);
          const meta =
            g.roof === "dome" || g.roof === "closed"
              ? `${g.stadium} · techo cerrado`
              : `${g.stadium}${g.temperature_c != null ? ` · ${Math.round(g.temperature_c)}°C · ${Math.round(g.wind_kph ?? 0)} km/h` : ""}`;
          return (
            <div
              key={g.game_id}
              className="flex gap-[14px] py-5"
              style={{ borderBottom: "1px solid var(--hair-soft)" }}
            >
              <div
                className="w-[3px] shrink-0 rounded-sm"
                style={{ background: TEAM_COLORS[g.home_team] ?? "var(--text-3)" }}
              />
              <div className="flex min-w-0 flex-1 flex-col gap-2">
                <div className="text-[16px] font-bold tracking-tight">
                  {g.away_team}
                  <span className="mx-[5px] font-medium" style={{ color: "var(--text-3)" }}>
                    @
                  </span>
                  {g.home_team}
                </div>
                <div
                  className="text-[11px] uppercase tracking-wide"
                  style={{ color: "var(--text-3)" }}
                >
                  Semana {g.week} · {meta}
                </div>
                {g.explanation_es && (
                  <p className="text-[12.5px] leading-snug" style={{ color: "var(--text-2)" }}>
                    {g.explanation_es}
                  </p>
                )}
                <Link
                  href={`/partido/${g.game_id}?target=${g.target}`}
                  className="pt-1 text-left text-[11.5px] font-semibold"
                  style={{ color: "var(--accent)" }}
                >
                  Ver detalle →
                </Link>
              </div>
              <div className="flex w-[96px] shrink-0 flex-col items-end gap-[3px]">
                <span className="text-[11.5px]" style={{ color: "var(--text-2)" }}>
                  {pickLabel(g)}
                </span>
                <span className="text-[25px] font-bold leading-none tabular-nums">
                  {displayProb(g)}%
                </span>
                <span
                  className="text-[10.5px] uppercase tracking-wide"
                  style={{ color: conf.color, fontWeight: conf.weight }}
                >
                  {g.confidence_level}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {sheetOpen && (
        <div
          className="fixed inset-0 z-20 flex items-end"
          style={{ background: "rgba(4,5,7,0.65)" }}
          onClick={(e) => e.target === e.currentTarget && setSheetOpen(false)}
        >
          <div
            className="w-full p-[22px]"
            style={{ background: "var(--surface)", borderTop: "1px solid var(--hair)" }}
          >
            <div className="mb-[14px] flex items-center justify-between text-[16px] font-semibold">
              Filtros
              <button
                onClick={() => setSheetOpen(false)}
                className="text-[11px]"
                style={{ color: "var(--text-2)" }}
              >
                CERRAR
              </button>
            </div>
            <label className="flex items-center justify-between py-[13px] text-[13px]" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
              <span className="text-[12px]" style={{ color: "var(--text-2)" }}>Semana</span>
              <select
                className="bg-transparent text-right font-semibold"
                value={week ?? ""}
                onChange={(e) => setWeek(e.target.value ? Number(e.target.value) : undefined)}
              >
                <option value="">Todas</option>
                {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
                  <option key={w} value={w}>{w}</option>
                ))}
              </select>
            </label>
            <label className="flex items-center justify-between py-[13px] text-[13px]" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
              <span className="text-[12px]" style={{ color: "var(--text-2)" }}>Equipo</span>
              <select
                className="bg-transparent text-right font-semibold"
                value={team ?? ""}
                onChange={(e) => setTeam(e.target.value || undefined)}
              >
                <option value="">Todos</option>
                {TEAMS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>
            <button
              onClick={() => setSheetOpen(false)}
              className="mt-[18px] py-[14px] text-[13px] font-semibold uppercase"
              style={{ border: "1px solid var(--accent)", background: "var(--accent-soft)", color: "var(--accent)" }}
            >
              Aplicar filtros
            </button>
          </div>
        </div>
      )}

      <BottomNav active="predicciones" />
    </div>
  );
}
