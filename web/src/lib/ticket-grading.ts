/**
 * Calificacion de selecciones y tickets contra resultados reales. Funciones puras.
 */
import { Leg, TicketLeg } from "./parlay";
import type { TicketResultGame, TicketResultStat, TicketResultsResponse } from "./ticket-results-types";

export type LegGrade =
  | "hit" // acerto
  | "miss" // fallo
  | "push" // empate en linea entera: la seleccion se anula
  | "void" // el jugador no jugo: la seleccion se anula
  | "pending" // el partido no termino
  | "unverifiable"; // sin datos para calificar

export type TicketStatus = "ganado" | "perdido" | "pendiente" | "anulado" | "sin verificar";

export interface GameOutcome {
  game: TicketResultGame | undefined;
  /** estadisticas de los jugadores de ESTE partido */
  stats: TicketResultStat[];
  firstTd: { game_id: string; gsis_id: string }[] | null;
}

const STAT_FIELDS = [
  "receiving_yards",
  "rushing_yards",
  "receptions",
  "receiving_tds",
  "rushing_tds",
  "passing_yards",
  "passing_tds",
  "sacks_suffered",
] as const;
type StatField = (typeof STAT_FIELDS)[number];

const isStatField = (t: string): t is StatField => (STAT_FIELDS as readonly string[]).includes(t);

/** Compara un valor con la linea. Lineas enteras con valor igual son empate. */
function compare(value: number, line: number, side: "over" | "under"): LegGrade {
  if (value === line) return "push";
  return (side === "over" ? value > line : value < line) ? "hit" : "miss";
}

type StatLookup = { kind: "value"; value: number } | { kind: "void" } | { kind: "unverifiable" };

/**
 * Los jugadores con prediccion que no jugaron llegan con todos los campos en null (anulan la seleccion);
 * los que no tienen fila no se pueden verificar.
 */
function lookup(stats: TicketResultStat[], gsis: string, fields: StatField[]): StatLookup {
  const row = stats.find((r) => r.gsis_id === gsis);
  if (!row) return { kind: "unverifiable" };
  const vals = fields.map((f) => row[f]);
  if (STAT_FIELDS.every((f) => row[f] == null)) return { kind: "void" };
  if (vals.some((v) => v == null || !Number.isFinite(v))) return { kind: "unverifiable" };
  return { kind: "value", value: (vals as number[]).reduce((a, b) => a + b, 0) };
}

export function gradeLeg(leg: Leg, ctx: GameOutcome): LegGrade {
  const g = ctx.game;
  if (!g || g.status !== "final") return "pending";
  const h = g.home_score;
  const a = g.away_score;
  switch (leg.kind) {
    case "winner": {
      if (h == null || a == null) return "unverifiable";
      if (h === a) return "push";
      return (leg.team === "home" ? h > a : a > h) ? "hit" : "miss";
    }
    case "spread": {
      if (h == null || a == null) return "unverifiable";
      const adj = (leg.team === "home" ? h - a : a - h) + leg.line;
      return adj === 0 ? "push" : adj > 0 ? "hit" : "miss";
    }
    case "total": {
      if (h == null || a == null) return "unverifiable";
      return compare(h + a, leg.line, leg.side);
    }
    case "first_td": {
      if (ctx.firstTd == null) return "unverifiable";
      const row = ctx.firstTd.find((r) => r.game_id === g.game_id);
      if (!row) return "unverifiable";
      return row.gsis_id === leg.gsis_id ? "hit" : "miss";
    }
    case "anytime_td":
    case "two_td": {
      const r = lookup(ctx.stats, leg.gsis_id, ["receiving_tds", "rushing_tds"]);
      if (r.kind === "void") return "void";
      if (r.kind !== "value") return "unverifiable";
      return r.value >= (leg.kind === "anytime_td" ? 1 : 2) ? "hit" : "miss";
    }
    case "player": {
      if (!isStatField(leg.target)) return "unverifiable";
      const r = lookup(ctx.stats, leg.gsis_id, [leg.target]);
      if (r.kind === "void") return "void";
      if (r.kind !== "value") return "unverifiable";
      return compare(r.value, leg.line, leg.side);
    }
  }
}

export function ticketStatus(grades: LegGrade[]): TicketStatus {
  if (grades.length === 0) return "pendiente";
  if (grades.includes("miss")) return "perdido";
  if (grades.includes("pending")) return "pendiente";
  if (grades.includes("unverifiable")) return "sin verificar";
  if (grades.every((g) => g === "push" || g === "void")) return "anulado";
  return "ganado";
}

export function outcomeFor(gameId: string, res: TicketResultsResponse | undefined): GameOutcome {
  return {
    game: res?.games.find((g) => g.game_id === gameId),
    stats: (res?.stats ?? []).filter((s) => s.game_id === gameId),
    firstTd: res?.first_td ?? null,
  };
}

export function gradeTicket(
  legs: TicketLeg[],
  res: TicketResultsResponse | undefined
): { grades: LegGrade[]; status: TicketStatus } {
  const grades = legs.map((t) => gradeLeg(t.leg, outcomeFor(t.gameId, res)));
  return { grades, status: ticketStatus(grades) };
}

// ---------------------------------------------------------------- historial

export interface HistoryItem {
  probability: number;
  status: TicketStatus;
}

export interface CalibrationBand {
  label: string;
  /** probabilidad media predicha de la banda */
  avgPredicted: number;
  n: number;
  wins: number;
}

export interface HistorySummary {
  total: number;
  graded: number;
  wins: number;
  hitRate: number | null;
  avgPredicted: number | null;
  bands: CalibrationBand[];
}

const BANDS: [number, number][] = [
  [0, 0.05],
  [0.05, 0.15],
  [0.15, 0.3],
  [0.3, 0.5],
  [0.5, 1.0001],
];

export function summarizeHistory(items: HistoryItem[]): HistorySummary {
  const graded = items.filter((i) => i.status === "ganado" || i.status === "perdido");
  const wins = graded.filter((i) => i.status === "ganado").length;
  const bands: CalibrationBand[] = [];
  for (const [lo, hi] of BANDS) {
    const inBand = graded.filter((i) => i.probability >= lo && i.probability < hi);
    if (inBand.length === 0) continue;
    bands.push({
      label: `${Math.round(lo * 100)}-${Math.min(100, Math.round(hi * 100))} %`,
      avgPredicted: inBand.reduce((a, i) => a + i.probability, 0) / inBand.length,
      n: inBand.length,
      wins: inBand.filter((i) => i.status === "ganado").length,
    });
  }
  return {
    total: items.length,
    graded: graded.length,
    wins,
    hitRate: graded.length > 0 ? wins / graded.length : null,
    avgPredicted: graded.length > 0 ? graded.reduce((a, i) => a + i.probability, 0) / graded.length : null,
    bands,
  };
}
