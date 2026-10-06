"use client";

import { Fragment, useState } from "react";
import { teamColor } from "@/lib/teams";
import { PropPrediction } from "@/lib/types";

export function TeamDot({ team }: { team: string }) {
  return (
    <span
      className="inline-block h-[10px] w-[10px] shrink-0 rounded-[3px]"
      style={{ background: teamColor(team) }}
    />
  );
}

export function PosTag({ pos }: { pos: string | null }) {
  if (!pos) return null;
  return (
    <span
      className="rounded-[5px] px-[6px] py-px text-[11px] font-semibold"
      style={{ color: "var(--text-2)", border: "1px solid var(--hair)" }}
    >
      {pos}
    </span>
  );
}

const TH = "px-3 py-[10px] text-left text-[11px] font-semibold uppercase tracking-wider";
const TD = "px-3 py-[9px]";

function fmt(v: number | null | undefined, digits = 0): string {
  return v == null ? "—" : v.toFixed(digits);
}

/** Barra del rango 80% con marca en la proyeccion. max fija la escala. */
function RangeBar({ p, max }: { p: PropPrediction; max: number }) {
  if (p.point_estimate == null || p.lower_80 == null || p.upper_80 == null) {
    return <span style={{ color: "var(--text-3)" }}>—</span>;
  }
  const sc = (n: number) => Math.max(0, Math.min(100, (n / max) * 100));
  const lo = sc(p.lower_80);
  const hi = sc(p.upper_80);
  return (
    <div className="flex items-center gap-2">
      <div
        className="relative h-2 min-w-[90px] flex-1 rounded"
        style={{ background: "color-mix(in srgb, var(--text) 8%, transparent)" }}
      >
        <i
          className="absolute bottom-0 top-0 rounded opacity-80"
          style={{ left: `${lo}%`, width: `${Math.max(2, hi - lo)}%`, background: teamColor(p.team) }}
        />
        <b
          className="absolute top-[-3px] h-[14px] w-[3px] rounded-sm"
          style={{ left: `calc(${sc(p.point_estimate)}% - 1px)`, background: "var(--text)" }}
        />
      </div>
      <span className="text-[11px] tabular-nums" style={{ color: "var(--text-3)" }}>
        {p.lower_80.toFixed(1)}–{p.upper_80.toFixed(1)}
      </span>
    </div>
  );
}

const WRAP = { border: "1px solid var(--hair)", background: "var(--surface)" };

export interface GameGroup {
  game_id: string;
  home_team: string;
  away_team: string;
  kickoff_utc: string;
  /** Equipo favorito (ganador) o null si no hay dato. */
  favorite: string | null;
  rows: PropPrediction[];
}

/** Proyecciones de UN stat agrupadas por partido; cada fila expande su explicacion. */
export function PropGroups({ groups, unit }: { groups: GameGroup[]; unit: string }) {
  const [open, setOpen] = useState<string | null>(null);
  const max =
    Math.max(1, ...groups.flatMap((g) => g.rows.map((p) => p.upper_80 ?? p.point_estimate ?? 0))) * 1.1;
  return (
    <div>
      {groups.map((g) => {
        const tint = g.favorite ? teamColor(g.favorite) : "var(--hair)";
        return (
          <section key={g.game_id} className="mb-6">
            <div
              className="flex flex-wrap items-baseline justify-between gap-x-3 py-2 pl-3"
              style={{ borderLeft: `3px solid ${tint}`, borderBottom: "1px solid var(--hair)" }}
            >
              <span className="text-[14px] font-bold">
                {g.away_team} <span style={{ color: "var(--text-3)" }}>@</span> {g.home_team}
              </span>
              <span className="text-[11.5px] tabular-nums" style={{ color: "var(--text-2)" }}>
                {new Date(g.kickoff_utc).toLocaleString("es-MX", {
                  weekday: "short",
                  day: "numeric",
                  month: "short",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
                {g.favorite && <span style={{ color: tint }}> · Favorito {g.favorite}</span>}
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[520px] border-collapse text-[13.5px] tabular-nums">
                <thead>
                  <tr style={{ color: "var(--text-3)", borderBottom: "1px solid var(--hair)" }}>
                    <th className={TH}>Jugador</th>
                    <th className={TH}>Rival</th>
                    <th className={`${TH} text-right`}>Proy.</th>
                    <th className={TH}>Rango 80 %</th>
                  </tr>
                </thead>
                <tbody>
                  {g.rows.map((p) => {
                    const key = `${p.game_id}-${p.gsis_id}`;
                    const isOpen = open === key;
                    const opp = p.team === p.home_team ? `vs ${p.away_team}` : `@ ${p.home_team}`;
                    const can = !!p.explanation_es;
                    return (
                      <Fragment key={key}>
                        <tr
                          onClick={can ? () => setOpen(isOpen ? null : key) : undefined}
                          style={{ borderBottom: "1px solid var(--hair-soft)", cursor: can ? "pointer" : "default" }}
                          aria-expanded={can ? isOpen : undefined}
                        >
                          <td className={TD}>
                            <span className="flex items-center gap-[10px]">
                              <TeamDot team={p.team} />
                              <span className="font-semibold">{p.player_name}</span>
                              <PosTag pos={p.position} />
                              {can && (
                                <span className="text-[11px]" style={{ color: "var(--text-3)" }}>
                                  {isOpen ? "−" : "+"}
                                </span>
                              )}
                            </span>
                          </td>
                          <td className={TD} style={{ color: "var(--text-2)" }}>{opp}</td>
                          <td className={`${TD} text-right`}>
                            <span className="text-[21px] font-bold">{fmt(p.point_estimate, 1)}</span>{" "}
                            <span className="text-[11px]" style={{ color: "var(--text-3)" }}>{unit}</span>
                          </td>
                          <td className={TD}><RangeBar p={p} max={max} /></td>
                        </tr>
                        {isOpen && p.explanation_es && (
                          <tr style={{ borderBottom: "1px solid var(--hair-soft)" }}>
                            <td
                              colSpan={4}
                              className="px-3 pb-3 pt-1 text-[12.5px] leading-relaxed"
                              style={{ color: "var(--text-2)" }}
                            >
                              {p.explanation_es}
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}
    </div>
  );
}

export interface Cell {
  /** Proyeccion y rango 80 %. */
  p: number | null;
  lo: number | null;
  hi: number | null;
  /** Valor real (solo se muestra en partidos ya jugados). */
  a: number | null;
}

export const emptyCell = (): Cell => ({ p: null, lo: null, hi: null, a: null });

export interface PlayerLine {
  gsis_id: string;
  name: string;
  team: string;
  position: string | null;
  rec: Cell;
  rush: Cell;
  recs: Cell;
  pass: Cell;
  td: Cell;
}

type ColKey = "rec" | "rush" | "recs" | "pass" | "td";

const COLS: { k: ColKey; label: string; d: number; range: boolean }[] = [
  { k: "rec", label: "Rec yds", d: 0, range: true },
  { k: "rush", label: "Rush yds", d: 0, range: true },
  { k: "recs", label: "Rec", d: 1, range: true },
  { k: "pass", label: "Pase yds", d: 0, range: true },
  { k: "td", label: "TD", d: 2, range: false },
];

/**
 * Tabla del partido, una fila por jugador. Futuro: proyeccion con rango 80 %.
 * Jugado (played): "Proyectado / Real", evaluacion del modelo.
 */
export function GamePlayersTable({ players, played }: { players: PlayerLine[]; played: boolean }) {
  const proj = (c: Cell, d: number) => (c.p == null ? "—" : c.p.toFixed(d));
  return (
    <div className="overflow-x-auto rounded-[14px]" style={WRAP}>
      <table className="w-full min-w-[560px] border-collapse text-[13.5px] tabular-nums">
        <thead>
          <tr style={{ color: "var(--text-3)", borderBottom: "1px solid var(--hair)" }}>
            <th className={TH}>Jugador</th>
            <th className={TH}>Pos</th>
            {COLS.map((c) => (
              <th key={c.k} className={`${TH} text-right`}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {players.map((x) => (
            <tr key={x.gsis_id} style={{ borderBottom: "1px solid var(--hair-soft)" }}>
              <td className={TD}>
                <span className="flex items-center gap-[10px]">
                  <TeamDot team={x.team} />
                  <span className="font-semibold">{x.name}</span>
                  <span className="text-[12px]" style={{ color: "var(--text-3)" }}>{x.team}</span>
                </span>
              </td>
              <td className={TD}><PosTag pos={x.position} /></td>
              {COLS.map((c) => {
                const cell = x[c.k];
                return (
                  <td key={c.k} className={`${TD} text-right`}>
                    {played ? (
                      <>
                        {proj(cell, c.d)} <span style={{ color: "var(--text-3)" }}>/</span>{" "}
                        {cell.a == null ? "—" : cell.a.toFixed(c.d)}
                      </>
                    ) : (
                      <>
                        <div>{proj(cell, c.d)}</div>
                        {c.range && cell.p != null && cell.lo != null && cell.hi != null && (
                          <div className="text-[10.5px]" style={{ color: "var(--text-3)" }}>
                            {cell.lo.toFixed(c.d)}–{cell.hi.toFixed(c.d)}
                          </div>
                        )}
                      </>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
