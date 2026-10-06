"use client";

import { use } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { BottomNav } from "@/components/bottom-nav";
import { ConfidenceTag, gameTint } from "@/components/game-card";
import { Cell, GamePlayersTable, PlayerLine, emptyCell } from "@/components/prop-table";
import { teamColor, teamInk } from "@/lib/teams";
import { GamePrediction, PropPrediction, Target, displayProb, pickLabel } from "@/lib/types";

async function fetchGame(gameId: string, target: Target) {
  const params = new URLSearchParams({ game_id: gameId, target });
  const res = await fetch(`/api/predictions?${params}`);
  if (!res.ok) throw new Error("No se pudo cargar la predicción");
  const data = await res.json();
  return (data.predictions as GamePrediction[])[0] ?? null;
}

const PLAYER_STATS = [
  "receiving_yards",
  "rushing_yards",
  "receptions",
  "passing_yards",
  "receiving_tds",
  "rushing_tds",
  "passing_tds",
] as const;

async function fetchGamePlayers(gameId: string): Promise<PlayerLine[]> {
  const results = await Promise.all(
    PLAYER_STATS.map(async (target) => {
      const params = new URLSearchParams({ game_id: gameId, target });
      const res = await fetch(`/api/props?${params}`);
      if (!res.ok) throw new Error("No se pudo cargar los jugadores");
      const data = await res.json();
      return [target, data.props as PropPrediction[]] as const;
    })
  );
  const byId = new Map<string, PlayerLine>();
  for (const [target, rows] of results) {
    for (const p of rows) {
      let line = byId.get(p.gsis_id);
      if (!line) {
        line = { gsis_id: p.gsis_id, name: p.player_name, team: p.team, position: p.position, rec: emptyCell(), rush: emptyCell(), recs: emptyCell(), pass: emptyCell(), td: emptyCell() };
        byId.set(p.gsis_id, line);
      }
      if (p.point_estimate == null) continue;
      const isTd = target.endsWith("_tds");
      const cell: Cell =
        target === "receiving_yards" ? line.rec
        : target === "rushing_yards" ? line.rush
        : target === "receptions" ? line.recs
        : target === "passing_yards" ? line.pass
        : line.td;
      // Los TD se suman entre tipos; los demas stats asignan directo.
      cell.p = (isTd ? (cell.p ?? 0) : 0) + p.point_estimate;
      cell.lo = isTd ? null : p.lower_80;
      cell.hi = isTd ? null : p.upper_80;
      // Real solo para evaluar partidos jugados.
      const actual = p[target as keyof PropPrediction] as number | null;
      if (actual != null) cell.a = (isTd ? (cell.a ?? 0) : 0) + actual;
    }
  }
  const score = (l: PlayerLine) => (l.rec.p ?? 0) + (l.rush.p ?? 0) + (l.pass.p ?? 0);
  return [...byId.values()].sort((a, b) => score(b) - score(a));
}

function friendlyFactorLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\br(\d+)\b/g, "últimas $1 semanas")
    .replace(/\bewm\b/g, "promedio ponderado")
    .replace(/\belo\b/gi, "Elo")
    .replace(/\bepa\b/gi, "EPA")
    .replace(/^./, (c) => c.toUpperCase());
}

export default function DetallePage({ params }: { params: Promise<{ gameId: string }> }) {
  const { gameId } = use(params);
  const searchParams = useSearchParams();
  const target = (searchParams.get("target") as Target) ?? "winner";

  const { data: g, isLoading } = useQuery({
    queryKey: ["game", gameId, target],
    queryFn: () => fetchGame(gameId, target),
  });

  const { data: players, isLoading: playersLoading, isError: playersError } = useQuery({
    queryKey: ["game-players", gameId],
    queryFn: () => fetchGamePlayers(gameId),
  });

  let factors: { label: string; value: number }[] = [];
  if (g?.shap_values) {
    try {
      const parsed = JSON.parse(g.shap_values) as Record<string, number>;
      factors = Object.entries(parsed)
        .map(([k, v]) => ({ label: friendlyFactorLabel(k), value: v }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 6);
    } catch {
      factors = [];
    }
  }
  const played = g?.home_score != null && g?.away_score != null;
  const maxAbs = Math.max(1e-6, ...factors.map((f) => Math.abs(f.value)));

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <Link href="/" className="text-[12px]" style={{ color: "var(--text-2)" }}>
          ‹ Volver
        </Link>

        {isLoading && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--text-2)" }}>
            Cargando…
          </p>
        )}

        {g && (
          <>
            <div
              className="tint-hero mt-3 p-5"
              style={{ ["--t" as string]: gameTint(g) }}
            >
              <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
                Semana {g.week} · {new Date(g.kickoff_utc).toLocaleString("es-MX", { dateStyle: "short", timeStyle: "short" })} · {g.stadium}
              </span>
              <div className="mt-2 flex flex-wrap items-end justify-between gap-x-7 gap-y-3">
                <span className="font-display text-[clamp(40px,11vw,72px)] font-bold leading-[0.95]">
                  {g.away_team} <span className="text-[0.5em]" style={{ color: "var(--text-3)" }}>@</span> {g.home_team}
                </span>
                <div className="text-right">
                  <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
                    Predicción · {target === "winner" ? "Ganador" : target === "spread" ? "Spread" : "Total"}
                  </div>
                  <div className="font-display t-ink text-[clamp(36px,10vw,60px)] font-bold leading-none tabular-nums">
                    {pickLabel(g)}
                  </div>
                </div>
              </div>
              {target === "winner" && displayProb(g) != null && (
                <div className="pbar mt-[14px]" style={{ height: 10 }} aria-hidden>
                  <i style={{ width: `${displayProb(g)}%` }} />
                </div>
              )}
              <div className="mt-3">
                <span style={{ color: "var(--text-2)" }} className="mr-2 text-[10.5px] uppercase tracking-wide">Confianza</span>
                <ConfidenceTag level={g.confidence_level} />
              </div>
            </div>

            <div className="pt-[18px]">
              {g.explanation_es && (
                <p className="mb-[26px] mt-4 text-[13.5px] leading-relaxed" style={{ color: "var(--text-2)" }}>
                  {g.explanation_es}
                </p>
              )}

              {factors.length > 0 && (
                <>
                  <div className="mb-4 flex gap-[18px] text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
                    <span style={{ color: teamInk(g.home_team) }}>— {g.home_team}</span>
                    <span style={{ color: teamInk(g.away_team) }}>— {g.away_team}</span>
                  </div>
                  {factors.map((f) => {
                    const side = f.value >= 0 ? "home" : "away";
                    const color = teamColor(side === "home" ? g.home_team : g.away_team);
                    const ink = teamInk(side === "home" ? g.home_team : g.away_team);
                    const widthPct = (Math.abs(f.value) / maxAbs) * 50;
                    return (
                      <div key={f.label} className="flex flex-col gap-[5px] py-[10px]" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
                        <div className="flex items-baseline justify-between">
                          <span className="text-[12.5px]">{f.label}</span>
                          <span className="text-[11.5px] font-semibold tabular-nums" style={{ color: ink }}>
                            {f.value >= 0 ? "+" : ""}{f.value.toFixed(3)}
                          </span>
                        </div>
                        <div className="relative h-[3px]" style={{ background: "var(--hair)" }}>
                          <div
                            className="absolute top-0 bottom-0"
                            style={{
                              [side === "home" ? "left" : "right"]: "50%",
                              width: `${widthPct}%`,
                              background: color,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </>
              )}

              <div className="mb-2 mt-[26px] text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
                Proyección de jugadores{played ? " · Proyectado / Real" : ""}
              </div>
              {played && (
                <p className="mb-2 text-[11.5px]" style={{ color: "var(--text-3)" }}>
                  Partido ya jugado: cada celda muestra la proyección del modelo y el resultado real, para evaluar el modelo.
                </p>
              )}
              {playersLoading && (
                <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Cargando jugadores…</p>
              )}
              {playersError && (
                <p className="text-[13px]" style={{ color: "var(--side-b)" }}>No se pudieron cargar los jugadores.</p>
              )}
              {players && players.length === 0 && (
                <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Sin proyecciones de jugadores para este partido.</p>
              )}
              {players && players.length > 0 && <GamePlayersTable players={players} played={played} />}

              <div className="mt-[22px] pt-4 text-[10.5px] leading-loose" style={{ color: "var(--text-3)", borderTop: "1px solid var(--hair)" }}>
                Modelo {g.model_family ?? "—"} v{g.model_version ?? "—"} · predicho{" "}
                {g.predicted_at ? new Date(g.predicted_at).toLocaleString("es-MX") : "—"}
                <br />
                Corte as-of {g.as_of_ts ? new Date(g.as_of_ts).toLocaleString("es-MX") : "—"}
              </div>
            </div>
          </>
        )}
      </div>
      <BottomNav active="predicciones" />
    </div>
  );
}
