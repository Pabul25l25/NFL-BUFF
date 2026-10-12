import { describe, expect, it } from "vitest";
import {
  Leg,
  ParlayInputs,
  TicketLeg,
  buildTicketSim,
  evaluateTicket,
  simulateGame,
  ticketLegHits,
} from "./parlay";
import {
  altLines,
  binom,
  legCountDistribution,
  optimizeToOdds,
  roundRobin,
  scenarioSummary,
  suggestCorrelated,
  typicalScenario,
  weakestLink,
} from "./parlay-analysis";

const Z = [
  -1.6449, -1.2816, -1.0364, -0.8416, -0.6745, -0.5244, -0.3853, -0.2533, -0.1257, 0, 0.1257, 0.2533, 0.3853,
  0.5244, 0.6745, 0.8416, 1.0364, 1.2816, 1.6449,
];
const q = (mu: number, sd: number) => Z.map((z) => mu + sd * z);

function inputs(id = "g1", rho = 0.6): ParlayInputs {
  return {
    game: {
      game_id: id, season: 2026, week: 5, home_team: "KC", away_team: "BUF", kickoff_utc: "2026-10-11T17:00:00Z",
      p_home_win: 0.6, margin_mu: 3, margin_sigma: 13, total_mu: 47, total_sigma: 10,
    },
    players: [
      { gsis_id: "A", player_name: "A", position: "WR", team: "KC", is_home: true },
      { gsis_id: "B", player_name: "B", position: "WR", team: "KC", is_home: true },
    ],
    markets: [
      { gsis_id: "A", target: "receiving_yards", point_estimate: 100, quantiles: q(100, 20), lambda: null },
      { gsis_id: "B", target: "receiving_yards", point_estimate: 80, quantiles: q(80, 20), lambda: null },
      { gsis_id: "A", target: "receiving_tds", point_estimate: 0.5, quantiles: null, lambda: 0.5 },
    ],
    corr: {
      legs: ["p:A:receiving_yards", "p:B:receiving_yards", "g:margin", "g:total"],
      matrix: [
        [1, rho, 0.2, 0.3],
        [rho, 1, 0.1, 0.4],
        [0.2, 0.1, 1, 0],
        [0.3, 0.4, 0, 1],
      ],
    },
    calibration: null,
  };
}

const over = (id: string, line: number): Leg => ({ kind: "player", gsis_id: id, target: "receiving_yards", side: "over", line });
const T = (gameId: string, leg: Leg): TicketLeg => ({ gameId, leg });
const by = () => ({ g1: inputs("g1"), g2: inputs("g2", 0.2) });
const ticket1: TicketLeg[] = [
  T("g1", over("A", 100)),
  T("g1", over("B", 80)),
  T("g1", { kind: "winner", team: "home" }),
  T("g1", { kind: "total", side: "over", line: 44.5 }),
];

describe("matriz de aciertos", () => {
  it("conjunta == promedio de las filas donde todas aciertan", () => {
    const gs = simulateGame(inputs(), ticket1.map((t) => t.leg), {});
    let all = 0;
    for (let s = 0; s < gs.sims; s++) {
      if (gs.hits.every((h) => (h as Uint8Array)[s] === 1)) all++;
    }
    expect(gs.joint).toBeCloseTo(all / gs.sims, 12);
    gs.results.forEach((r, i) => {
      const h = gs.hits[i] as Uint8Array;
      expect(r.p).toBeCloseTo(h.reduce((a, b) => a + b, 0) / gs.sims, 12);
    });
  });

  it("ticket de un partido: escenarios coinciden con evaluateTicket", () => {
    const ts = buildTicketSim(by(), ticket1);
    const r = evaluateTicket(by(), ticket1);
    const sc = scenarioSummary(ts);
    expect(sc.sims).toBe(20000);
    expect(sc.hits / sc.sims).toBeCloseTo(r.joint, 12);
  });

  it("ticket multi-partido: emparejar por indice es consistente con el producto de conjuntas", () => {
    const t = [T("g1", over("A", 100)), T("g2", over("A", 100)), T("g2", { kind: "winner", team: "away" })];
    const ts = buildTicketSim(by(), t);
    const r = evaluateTicket(by(), t);
    const sc = scenarioSummary(ts);
    expect(Math.abs(sc.hits / sc.sims - r.joint)).toBeLessThan(0.01);
  });
});

describe("distribucion de selecciones cumplidas", () => {
  it("P(k de M) suma 1 y P(M) = conjunta", () => {
    const ts = buildTicketSim(by(), ticket1);
    const d = legCountDistribution(ts);
    expect(d.m).toBe(4);
    expect(d.probs).toHaveLength(5);
    expect(d.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(d.probs[4]).toBeCloseTo(scenarioSummary(ts).hits / ts.sims, 12);
  });
});

describe("eslabon debil y escenario tipico", () => {
  it("P(ticket | sin la seleccion) >= conjunta y hay una peor seleccion", () => {
    const ts = buildTicketSim(by(), ticket1);
    const w = weakestLink(ts);
    expect(w.rows).toHaveLength(4);
    for (const r of w.rows) expect(r.pWithout).toBeGreaterThanOrEqual(w.joint - 1e-12);
    expect(w.worst).not.toBeNull();
    const worst = w.rows.find((r) => r.index === w.worst);
    expect(worst?.missShare).toBe(Math.max(...w.rows.map((r) => r.missShare)));
  });

  it("la mediana de cada estadistica en escenarios ganadores cumple su propia seleccion", () => {
    const ts = buildTicketSim(by(), ticket1);
    const sc = typicalScenario(ts);
    expect(sc.n).toBeGreaterThan(100);
    const a = sc.rows.find((r) => r.index === 0);
    expect(a?.median).toBeGreaterThan(100);
    const tot = sc.rows.find((r) => r.index === 3);
    expect(tot?.median).toBeGreaterThan(44.5);
  });
});

describe("round robin", () => {
  it("las combinaciones coinciden con la evaluacion directa y la esperanza es la suma", () => {
    const ts = buildTicketSim(by(), ticket1);
    const rr = roundRobin(ts, 2);
    expect(rr).not.toBeNull();
    if (!rr) return;
    expect(rr.total).toBe(binom(4, 2));
    expect(rr.combos).toHaveLength(6);
    for (const c of rr.combos) {
      // exacto contra las mismas filas de la matriz
      const vs = c.indices.map((i) => ticketLegHits(ts, i) as Uint8Array);
      let n = 0;
      for (let s = 0; s < ts.sims; s++) if (vs.every((v) => v[s])) n++;
      expect(c.joint).toBeCloseTo(n / ts.sims, 12);
      // y contra una evaluacion directa del sub-ticket (otra simulacion: tolerancia de Monte Carlo)
      const direct = evaluateTicket(by(), c.indices.map((i) => ticket1[i]));
      expect(Math.abs(direct.joint - c.joint)).toBeLessThan(0.02);
    }
    expect(rr.expectedWins).toBeCloseTo(rr.combos.reduce((a, c) => a + c.joint, 0), 12);
    // E[#combos ganadoras] = E[C(h, k)] sobre las mismas simulaciones
    const d = legCountDistribution(ts);
    const viaDist = d.probs.reduce((a, p, h) => a + p * binom(h, 2), 0);
    expect(rr.expectedWins).toBeCloseTo(viaDist, 10);
    expect(rr.pAny).toBeCloseTo(d.probs[2] + d.probs[3] + d.probs[4], 12);
  });

  it("limita k a 2..min(4, M-1) y a 50 combinaciones", () => {
    const ts = buildTicketSim(by(), ticket1);
    expect(roundRobin(ts, 4)).toBeNull();
    expect(roundRobin(ts, 1)).toBeNull();
    const many: TicketLeg[] = [];
    for (let l = 40; l < 60; l++) many.push(T("g1", { kind: "total", side: "over", line: l + 0.5 }));
    const big = buildTicketSim(by(), many);
    const rr = roundRobin(big, 4);
    expect(rr?.total).toBe(4845);
    expect(rr?.combos).toHaveLength(50);
  });
});

describe("optimizacion", () => {
  it("lineas alternativas: la linea actual reproduce la conjunta y subir la linea de un over la baja", () => {
    const ts = buildTicketSim(by(), ticket1);
    const rows = altLines(ts, 0);
    const cur = rows.find((r) => r.current);
    expect(cur?.ticketP).toBeCloseTo(evaluateTicket(by(), ticket1).joint, 12);
    for (let i = 1; i < rows.length; i++) expect(rows[i].ticketP).toBeLessThanOrEqual(rows[i - 1].ticketP + 1e-12);
  });

  it("cuota objetivo: queda bajo el tope 1/cuota y mejora sobre el punto de partida mas duro", () => {
    const ts = buildTicketSim(by(), ticket1);
    const start = evaluateTicket(by(), ticket1).joint;
    // objetivo mas largo que el ticket actual: debe endurecer lineas
    const r = optimizeToOdds(ts, (1 / start) * 2);
    expect(r.status).toBe("ok");
    expect(r.joint).toBeLessThanOrEqual(r.cap + 1e-12);
    expect(r.joint).toBeGreaterThan(r.cap * 0.5);
    expect(r.changes.length).toBeGreaterThan(0);
    // objetivo mas corto: relaja y la conjunta sube
    const r2 = optimizeToOdds(ts, (1 / start) / 1.5);
    expect(r2.joint).toBeGreaterThan(start);
    expect(r2.joint).toBeLessThanOrEqual(r2.cap + 1e-12);
  });

  it("sugerencias correlacionadas: positivas, sin repetir y con conjunta calculada", () => {
    const t = [T("g1", over("A", 100))];
    const ts = buildTicketSim(by(), t);
    const s = suggestCorrelated(by(), ts);
    expect(s.length).toBeGreaterThan(0);
    expect(s.length).toBeLessThanOrEqual(3);
    for (const x of s) {
      expect(x.corr).toBeGreaterThan(0);
      expect(x.jointAfter).toBeLessThanOrEqual(x.jointBefore + 1e-12);
      expect(x.jointAfter).toBeGreaterThan(0);
    }
    expect(s[0].leg.kind === "player" && s[0].leg.gsis_id === "B").toBe(true);
  });
});
