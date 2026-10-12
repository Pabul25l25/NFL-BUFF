"use client";

import Link from "next/link";
import { Fragment, useState } from "react";
import { teamColor, teamInk } from "@/lib/teams";
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

/** Media esperada (con respaldo en la mediana si no hay media). */
export const meanOf = (p: PropPrediction): number | null => p.mean_estimate ?? p.point_estimate;

/** Barra del rango 80% con marca en la proyeccion. max fija la escala. */
function RangeBar({ p, max, color }: { p: PropPrediction; max: number; color: string }) {
  const mean = meanOf(p);
  if (mean == null || p.lower_80 == null || p.upper_80 == null) {
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
          style={{ left: `${lo}%`, width: `${Math.max(2, hi - lo)}%`, background: color }}
        />
        <b
          className="absolute top-[-3px] h-[14px] w-[3px] rounded-sm"
          style={{ left: `calc(${sc(mean)}% - 1px)`, background: "var(--text)" }}
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

/** Equipo del jugador: p.team si existe; si no, por is_home. */
export function playerTeam(p: PropPrediction): string | null {
  if (p.team) return p.team;
  if (p.is_home === true) return p.home_team;
  if (p.is_home === false) return p.away_team;
  return null;
}

/** Apellido(s) para el resumen del encabezado. */
function shortName(n: string): string {
  const parts = n.trim().split(/\s+/);
  return parts.length > 1 ? parts.slice(1).join(" ") : n;
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="shrink-0"
      style={{ transform: open ? "rotate(180deg)" : undefined, color: "var(--text-3)" }}
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

function TeamBlock({
  label,
  team,
  rows,
  unit,
  max,
}: {
  label: string;
  team: string | null;
  rows: PropPrediction[];
  unit: string;
  max: number;
}) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="min-w-0">
      <div
        className="flex items-baseline justify-between py-2 pl-3"
        style={{ borderLeft: `3px solid ${teamColor(team)}`, borderBottom: "1px solid var(--hair)" }}
      >
        <span className="text-[14px] font-bold" style={{ color: team ? teamInk(team) : "var(--text-2)" }}>
          {label}
        </span>
        <span className="text-[11.5px] tabular-nums" style={{ color: "var(--text-3)" }}>
          {rows.length} {rows.length === 1 ? "jugador" : "jugadores"}
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[320px] border-collapse text-[13.5px] tabular-nums">
          <thead>
            <tr style={{ color: "var(--text-3)", borderBottom: "1px solid var(--hair)" }}>
              <th className={TH}>Jugador</th>
              <th className={`${TH} text-right`}>Media</th>
              <th className={TH}>Rango 80 %</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => {
              const key = `${p.game_id}-${p.gsis_id}`;
              const isOpen = open === key;
              const can = p.point_estimate != null || !!p.explanation_es;
              return (
                <Fragment key={key}>
                  <tr
                    onClick={can ? () => setOpen(isOpen ? null : key) : undefined}
                    style={{ borderBottom: "1px solid var(--hair-soft)", cursor: can ? "pointer" : "default" }}
                    aria-expanded={can ? isOpen : undefined}
                  >
                    <td className={TD}>
                      <span className="flex items-center gap-[8px]">
                        <span className="font-semibold">{p.player_name}</span>
                        <PosTag pos={p.position} />
                        {can && (
                          <span className="text-[11px]" style={{ color: "var(--text-3)" }}>
                            {isOpen ? "−" : "+"}
                          </span>
                        )}
                        {p.point_estimate != null && (
                          <Link
                            href={`/picks?game_id=${encodeURIComponent(p.game_id)}&add=${encodeURIComponent(
                              `${p.gsis_id}:${p.target}:over:${Math.floor(p.point_estimate) + 0.5}`
                            )}`}
                            onClick={(e) => e.stopPropagation()}
                            aria-label={`Agregar pick de ${p.player_name}`}
                            className="ml-auto whitespace-nowrap text-[11px] font-semibold"
                            style={{ color: "var(--accent)" }}
                          >
                            + Pick
                          </Link>
                        )}
                      </span>
                    </td>
                    <td className={`${TD} text-right`}>
                      <span className="text-[19px] font-bold">{fmt(meanOf(p), 1)}</span>{" "}
                      <span className="text-[11px]" style={{ color: "var(--text-3)" }}>{unit}</span>
                      {p.point_estimate != null && (
                        <div className="text-[11px]" style={{ color: "var(--text-3)" }}>
                          Línea 50/50 {fmt(p.point_estimate, 1)}
                        </div>
                      )}
                    </td>
                    <td className={TD}><RangeBar p={p} max={max} color={teamColor(team)} /></td>
                  </tr>
                  {isOpen && (
                    <tr style={{ borderBottom: "1px solid var(--hair-soft)" }}>
                      <td
                        colSpan={3}
                        className="px-3 pb-3 pt-1 text-[12.5px] leading-relaxed"
                        style={{ color: "var(--text-2)" }}
                      >
                        <p style={{ color: "var(--text-3)" }}>
                          Media: valor esperado; Línea 50/50: el jugador supera esta cifra la mitad de las veces.
                        </p>
                        {p.explanation_es && <p className="mt-1">{p.explanation_es}</p>}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const byEstimate = (a: PropPrediction, b: PropPrediction) => (meanOf(b) ?? -1) - (meanOf(a) ?? -1);

/** Partidos colapsables; al abrir muestra ambos equipos lado a lado (md) o apilados. */
export function PropGroups({
  groups,
  unit,
  open,
  onToggle,
}: {
  groups: GameGroup[];
  unit: string;
  open: Set<string>;
  onToggle: (gameId: string) => void;
}) {
  const max =
    Math.max(1, ...groups.flatMap((g) => g.rows.map((p) => p.upper_80 ?? meanOf(p) ?? 0))) * 1.1;
  return (
    <div>
      {groups.map((g) => {
        const tint = g.favorite ? teamColor(g.favorite) : "var(--hair)";
        const isOpen = open.has(g.game_id);
        const away: PropPrediction[] = [];
        const home: PropPrediction[] = [];
        const other: PropPrediction[] = [];
        for (const p of g.rows) {
          const t = playerTeam(p);
          if (t === g.away_team) away.push(p);
          else if (t === g.home_team) home.push(p);
          else other.push(p);
        }
        away.sort(byEstimate);
        home.sort(byEstimate);
        other.sort(byEstimate);
        const top = [away[0], home[0]].filter((p): p is PropPrediction => !!p && meanOf(p) != null);
        const panel = `panel-${g.game_id}`;
        return (
          <section key={g.game_id} className="mb-3">
            <button
              type="button"
              aria-expanded={isOpen}
              aria-controls={panel}
              onClick={() => onToggle(g.game_id)}
              className="flex w-full items-center gap-3 py-3 pl-3 text-left"
              style={{ borderLeft: `3px solid ${tint}`, borderBottom: "1px solid var(--hair)", color: "var(--text)" }}
            >
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-baseline justify-between gap-x-3">
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
                  </span>
                </span>
                <span
                  className="mt-[2px] flex flex-wrap gap-x-3 text-[11.5px] tabular-nums"
                  style={{ color: "var(--text-2)" }}
                >
                  {g.favorite && <span style={{ color: teamInk(g.favorite) }}>Favorito {g.favorite}</span>}
                  <span>{g.rows.length} {g.rows.length === 1 ? "jugador" : "jugadores"}</span>
                  {top.length > 0 && (
                    <span>
                      {top.map((p, i) => (
                        <Fragment key={p.gsis_id}>
                          {i > 0 && " · "}
                          {shortName(p.player_name)} {fmt(meanOf(p), 1)}
                        </Fragment>
                      ))}
                    </span>
                  )}
                </span>
              </span>
              <Chevron open={isOpen} />
            </button>
            {isOpen && (
              <div id={panel} className="grid grid-cols-1 gap-x-6 gap-y-4 pt-3 md:grid-cols-2">
                <TeamBlock label={g.away_team} team={g.away_team} rows={away} unit={unit} max={max} />
                <TeamBlock label={g.home_team} team={g.home_team} rows={home} unit={unit} max={max} />
                {other.length > 0 && (
                  <TeamBlock label="Sin equipo" team={null} rows={other} unit={unit} max={max} />
                )}
              </div>
            )}
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
