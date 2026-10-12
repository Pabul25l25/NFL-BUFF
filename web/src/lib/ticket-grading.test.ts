import { describe, expect, it } from "vitest";
import { Leg } from "./parlay";
import { GameOutcome, gradeLeg, gradeTicket, summarizeHistory, ticketStatus } from "./ticket-grading";
import type { TicketResultStat } from "./ticket-results-types";

const stat = (gsis: string, o: Partial<TicketResultStat> = {}): TicketResultStat => ({
  game_id: "g", gsis_id: gsis, receiving_yards: 0, rushing_yards: 0, receptions: 0, receiving_tds: 0, rushing_tds: 0,
  passing_yards: 0, passing_tds: 0, sacks_suffered: 0, ...o,
});
const allNull = (gsis: string): TicketResultStat => ({
  game_id: "g", gsis_id: gsis, receiving_yards: null, rushing_yards: null, receptions: null, receiving_tds: null,
  rushing_tds: null, passing_yards: null, passing_tds: null, sacks_suffered: null,
});
const ctx = (h: number | null, a: number | null, stats: TicketResultStat[] = [], firstTd: GameOutcome["firstTd"] = null): GameOutcome => ({
  game: { game_id: "g", status: h != null && a != null ? "final" : "pending", home_team: "KC", away_team: "BUF", home_score: h, away_score: a },
  stats,
  firstTd,
});
const ov = (line: number, target = "receiving_yards"): Leg => ({ kind: "player", gsis_id: "A", target, side: "over", line });
const un = (line: number): Leg => ({ kind: "player", gsis_id: "A", target: "receiving_yards", side: "under", line });

describe("gradeLeg", () => {
  const st = [stat("A", { receiving_yards: 70 }), stat("B", { receiving_yards: 30 })];
  it("over / under", () => {
    expect(gradeLeg(ov(69.5), ctx(24, 20, st))).toBe("hit");
    expect(gradeLeg(ov(70.5), ctx(24, 20, st))).toBe("miss");
    expect(gradeLeg(un(70.5), ctx(24, 20, st))).toBe("hit");
    expect(gradeLeg(un(69.5), ctx(24, 20, st))).toBe("miss");
  });
  it("empate exacto en linea entera", () => {
    expect(gradeLeg(ov(70), ctx(24, 20, st))).toBe("push");
    expect(gradeLeg(un(70), ctx(24, 20, st))).toBe("push");
  });
  it("pendiente si el partido no termino", () => {
    expect(gradeLeg(ov(10.5), ctx(null, null))).toBe("pending");
    expect(gradeLeg({ kind: "winner", team: "home" }, { game: undefined, stats: [], firstTd: null })).toBe("pending");
  });
  it("ganador, con empate de marcador", () => {
    expect(gradeLeg({ kind: "winner", team: "home" }, ctx(24, 20))).toBe("hit");
    expect(gradeLeg({ kind: "winner", team: "away" }, ctx(24, 20))).toBe("miss");
    expect(gradeLeg({ kind: "winner", team: "home" }, ctx(20, 20))).toBe("push");
  });
  it("spread con empate", () => {
    expect(gradeLeg({ kind: "spread", team: "home", line: -3.5 }, ctx(24, 20))).toBe("hit");
    expect(gradeLeg({ kind: "spread", team: "home", line: -4.5 }, ctx(24, 20))).toBe("miss");
    expect(gradeLeg({ kind: "spread", team: "home", line: -4 }, ctx(24, 20))).toBe("push");
    expect(gradeLeg({ kind: "spread", team: "away", line: 4 }, ctx(24, 20))).toBe("push");
    expect(gradeLeg({ kind: "spread", team: "away", line: 5.5 }, ctx(24, 20))).toBe("hit");
  });
  it("total", () => {
    expect(gradeLeg({ kind: "total", side: "over", line: 43.5 }, ctx(24, 20))).toBe("hit");
    expect(gradeLeg({ kind: "total", side: "under", line: 43.5 }, ctx(24, 20))).toBe("miss");
    expect(gradeLeg({ kind: "total", side: "over", line: 44 }, ctx(24, 20))).toBe("push");
  });
  it("TD en cualquier momento y 2+ TD suman recepcion y carrera", () => {
    const rows = [stat("A", { receiving_tds: 1, rushing_tds: 1 }), stat("B"), stat("C", { rushing_tds: 1 })];
    expect(gradeLeg({ kind: "anytime_td", gsis_id: "A" }, ctx(1, 0, rows))).toBe("hit");
    expect(gradeLeg({ kind: "two_td", gsis_id: "A" }, ctx(1, 0, rows))).toBe("hit");
    expect(gradeLeg({ kind: "two_td", gsis_id: "C" }, ctx(1, 0, rows))).toBe("miss");
    expect(gradeLeg({ kind: "anytime_td", gsis_id: "B" }, ctx(1, 0, rows))).toBe("miss");
  });
  it("primer TD: sin play-by-play no es verificable", () => {
    expect(gradeLeg({ kind: "first_td", gsis_id: "A" }, ctx(7, 0, [], null))).toBe("unverifiable");
    expect(gradeLeg({ kind: "first_td", gsis_id: "A" }, ctx(7, 0, [], [{ game_id: "g", gsis_id: "A" }]))).toBe("hit");
    expect(gradeLeg({ kind: "first_td", gsis_id: "A" }, ctx(7, 0, [], [{ game_id: "g", gsis_id: "Z" }]))).toBe("miss");
    expect(gradeLeg({ kind: "first_td", gsis_id: "A" }, ctx(7, 0, [], []))).toBe("unverifiable");
  });
  it("jugador sin fila no es verificable; con todo en null no jugo (anulada)", () => {
    expect(gradeLeg(ov(10.5), ctx(24, 20, [stat("B")]))).toBe("unverifiable");
    expect(gradeLeg(ov(10.5), ctx(24, 20, [allNull("A"), stat("B")]))).toBe("void");
    expect(gradeLeg({ kind: "anytime_td", gsis_id: "A" }, ctx(24, 20, [allNull("A")]))).toBe("void");
  });
  it("campo faltante en un jugador que si jugo no es verificable", () => {
    expect(gradeLeg(ov(1.5, "passing_tds"), ctx(24, 20, [stat("A", { passing_tds: null })]))).toBe("unverifiable");
  });
  it("mercado desconocido no es verificable", () => {
    expect(gradeLeg(ov(1.5, "fumbles"), ctx(24, 20, [stat("A")]))).toBe("unverifiable");
  });
});

describe("estado del ticket", () => {
  it("combina calificaciones", () => {
    expect(ticketStatus(["hit", "hit"])).toBe("ganado");
    expect(ticketStatus(["hit", "miss", "pending"])).toBe("perdido");
    expect(ticketStatus(["hit", "pending"])).toBe("pendiente");
    expect(ticketStatus(["hit", "unverifiable"])).toBe("sin verificar");
    expect(ticketStatus(["hit", "push"])).toBe("ganado");
    expect(ticketStatus(["push", "void"])).toBe("anulado");
  });
  it("gradeTicket usa los resultados por partido", () => {
    const res = {
      games: [{ game_id: "g", status: "final" as const, home_team: "KC", away_team: "BUF", home_score: 24, away_score: 20 }],
      stats: [stat("A", { receiving_yards: 70 })],
      first_td: null,
    };
    const r = gradeTicket(
      [
        { gameId: "g", leg: { kind: "winner", team: "home" } },
        { gameId: "g", leg: ov(60.5) },
        { gameId: "h", leg: { kind: "winner", team: "home" } },
      ],
      res
    );
    expect(r.grades).toEqual(["hit", "hit", "pending"]);
    expect(r.status).toBe("pendiente");
  });
});

describe("historial", () => {
  it("tasa de acierto y calibracion por banda", () => {
    const items = [
      ...Array.from({ length: 12 }, (_, i) => ({ probability: 0.25, status: i < 3 ? ("ganado" as const) : ("perdido" as const) })),
      { probability: 0.6, status: "pendiente" as const },
      { probability: 0.6, status: "sin verificar" as const },
    ];
    const h = summarizeHistory(items);
    expect(h.total).toBe(14);
    expect(h.graded).toBe(12);
    expect(h.hitRate).toBeCloseTo(0.25, 12);
    expect(h.avgPredicted).toBeCloseTo(0.25, 12);
    expect(h.bands).toHaveLength(1);
    expect(h.bands[0]).toMatchObject({ n: 12, wins: 3 });
    expect(summarizeHistory([]).hitRate).toBeNull();
  });
});
