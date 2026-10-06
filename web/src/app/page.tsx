"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BottomNav } from "@/components/bottom-nav";
import { GameCard } from "@/components/game-card";
import { TEAMS } from "@/lib/teams";
import { GamePrediction, Target } from "@/lib/types";
import { groupByWeek, selectWeek, weekNumbers } from "@/lib/weeks";

const SELECT_STYLE = { background: "var(--surface)", color: "var(--text)" };

const TYPE_TABS: { key: Target; label: string }[] = [
  { key: "winner", label: "Ganador" },
  { key: "spread", label: "Spread" },
  { key: "total", label: "Total" },
];

async function fetchPredictions(target: Target, team?: string) {
  const params = new URLSearchParams({ target });
  if (team) params.set("team", team);
  const res = await fetch(`/api/predictions?${params}`);
  if (!res.ok) throw new Error("No se pudo cargar predicciones");
  const data = await res.json();
  return data.predictions as GamePrediction[];
}

export default function PrediccionesPage() {
  const [type, setType] = useState<Target>("winner");
  const [week, setWeek] = useState<"next" | "all" | number>("next");
  const [team, setTeam] = useState<string | undefined>(undefined);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [now] = useState(() => Date.now());

  const { data, isLoading, isError } = useQuery({
    queryKey: ["predictions", type, team],
    queryFn: () => fetchPredictions(type, team),
  });

  const sel = useMemo(
    () => selectWeek(data, now, { week: typeof week === "number" ? week : undefined }),
    [data, now, week]
  );
  const groups = useMemo(() => (week === "all" ? groupByWeek(data) : []), [data, week]);
  const weekOpts = useMemo(() => weekNumbers(data), [data]);
  const upcoming = week === "all" ? [] : sel.rows;
  const total = week === "all" ? (data?.length ?? 0) : upcoming.length;

  const weekLabel = sel.week ?? (typeof week === "number" ? week : "—");
  const seasonLabel = sel.season ?? "";

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[env(safe-area-inset-top,0px)]">
        <div className="flex items-center justify-between gap-3 pt-[22px] pb-[18px]">
          <span className="flex items-center gap-2 text-[16px] font-bold tracking-tight">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/icon.svg" alt="" width={20} height={20} />
            NFL-Predictor
          </span>
          <button
            onClick={() => setSheetOpen(true)}
            className="min-w-0 border-b border-dotted text-right text-[12px]"
            style={{ color: "var(--text-2)", borderColor: "var(--text-3)" }}
          >
            {week === "all"
              ? "Todas las semanas"
              : `${sel.latest ? "Última semana" : "Semana"} ${weekLabel} · ${seasonLabel}`}
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
        {!isLoading && !isError && total === 0 && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--text-2)" }}>
            No hay predicciones para este filtro.
          </p>
        )}

        {week === "all" ? (
          groups.map((grp) => (
            <section key={`${grp.season}-${grp.week}`}>
              <h2
                className="mt-6 border-b pb-2 text-[11px] font-semibold uppercase tracking-wide tabular-nums"
                style={{ color: "var(--text-2)", borderColor: "var(--hair)" }}
              >
                Semana {grp.week} · {grp.season}
              </h2>
              <div className="grid grid-cols-1 gap-3 pt-4 sm:grid-cols-2 lg:grid-cols-3">
                {grp.rows.map((g) => (
                  <GameCard key={g.game_id} g={g} />
                ))}
              </div>
            </section>
          ))
        ) : (
          <div className="grid grid-cols-1 gap-3 pt-5 sm:grid-cols-2 lg:grid-cols-3">
            {upcoming.map((g) => (
              <GameCard key={g.game_id} g={g} />
            ))}
          </div>
        )}
      </div>

      {sheetOpen && (
        <div
          className="fixed inset-0 z-20 flex items-end"
          style={{ background: "var(--scrim)" }}
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
                className="text-right font-semibold"
                style={SELECT_STYLE}
                value={String(week)}
                onChange={(e) => {
                  const v = e.target.value;
                  setWeek(v === "next" || v === "all" ? v : Number(v));
                }}
              >
                <option value="next" style={SELECT_STYLE}>Próxima</option>
                <option value="all" style={SELECT_STYLE}>Todas las semanas</option>
                {weekOpts.map((w) => (
                  <option key={w} value={w} style={SELECT_STYLE}>{w}</option>
                ))}
              </select>
            </label>
            <label className="flex items-center justify-between py-[13px] text-[13px]" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
              <span className="text-[12px]" style={{ color: "var(--text-2)" }}>Equipo</span>
              <select
                className="text-right font-semibold"
                style={SELECT_STYLE}
                value={team ?? ""}
                onChange={(e) => setTeam(e.target.value || undefined)}
              >
                <option value="" style={SELECT_STYLE}>Todos</option>
                {TEAMS.map((t) => (
                  <option key={t} value={t} style={SELECT_STYLE}>{t}</option>
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
