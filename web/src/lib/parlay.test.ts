import { describe, expect, it } from "vitest";
import {
  ParlayInputs,
  choleskyRaw,
  choleskyRepair,
  decodeAdd,
  decodeLeg,
  evaluateParlay,
  hasMarket,
  legKey,
  normCdf,
  parseOdds,
  poissonCdfAt,
  quantileInverse,
  toAmerican,
} from "./parlay";

// Cuantiles de Normal(100, 20) en p05..p95
const Z = [
  -1.6449, -1.2816, -1.0364, -0.8416, -0.6745, -0.5244, -0.3853, -0.2533, -0.1257, 0, 0.1257, 0.2533, 0.3853,
  0.5244, 0.6745, 0.8416, 1.0364, 1.2816, 1.6449,
];
const q = (mu: number, sd: number) => Z.map((z) => mu + sd * z);

function inputs(rho = 0): ParlayInputs {
  return {
    game: {
      game_id: "g1", season: 2026, week: 5, home_team: "KC", away_team: "BUF", kickoff_utc: "2026-10-11T17:00:00Z",
      p_home_win: 0.6, margin_mu: 3, margin_sigma: 13, total_mu: 47, total_sigma: 10,
    },
    players: [
      { gsis_id: "A", player_name: "A", position: "WR", team: "KC", is_home: true },
      { gsis_id: "B", player_name: "B", position: "WR", team: "KC", is_home: true },
    ],
    markets: [
      { gsis_id: "A", target: "receiving_yards", point_estimate: 100, quantiles: q(100, 20), lambda: null },
      { gsis_id: "B", target: "receiving_yards", point_estimate: 100, quantiles: q(100, 20), lambda: null },
      { gsis_id: "A", target: "receiving_tds", point_estimate: 0.5, quantiles: null, lambda: 0.5 },
      { gsis_id: "A", target: "rushing_tds", point_estimate: 0.2, quantiles: null, lambda: 0.2 },
      { gsis_id: "B", target: "receiving_tds", point_estimate: 0.3, quantiles: null, lambda: 0.3 },
    ],
    corr: {
      legs: ["p:A:receiving_yards", "p:B:receiving_yards", "g:margin"],
      matrix: [
        [1, rho, 0],
        [rho, 1, 0],
        [0, 0, 1],
      ],
    },
    calibration: null,
  };
}

const over = (id: string, line: number) => ({ kind: "player" as const, gsis_id: id, target: "receiving_yards", side: "over" as const, line });

describe("validacion de mercados", () => {
  it("cuantiles no finitos dejan el mercado no disponible", () => {
    const i = inputs();
    expect(hasMarket(i, "A", "receiving_yards")).toBe(true);
    i.markets[0].quantiles = q(100, 20).map((v, k) => (k === 3 ? NaN : v));
    expect(hasMarket(i, "A", "receiving_yards")).toBe(false);
    i.markets[1].quantiles = q(100, 20).map((v, k) => (k === 5 ? Infinity : v));
    expect(hasMarket(i, "B", "receiving_yards")).toBe(false);
  });
});

describe("numerica", () => {
  it("normCdf", () => {
    expect(normCdf(1.96)).toBeCloseTo(0.975, 4);
    expect(normCdf(0)).toBeCloseTo(0.5, 10);
  });
  it("quantileInverse interpola y no baja de 0", () => {
    const g = q(100, 20);
    expect(quantileInverse(g, 0.5)).toBeCloseTo(100, 6);
    expect(quantileInverse(g, 0.0001)).toBeGreaterThanOrEqual(0);
    expect(quantileInverse([...Array(19).fill(0)], 0.3)).toBe(0);
  });
  it("Cholesky maneja entrada no PSD", () => {
    const bad = [
      [1, 0.9, -0.9],
      [0.9, 1, 0.9],
      [-0.9, 0.9, 1],
    ];
    expect(choleskyRaw(bad)).toBeNull();
    const { L, shrink } = choleskyRepair(bad);
    expect(shrink).toBeGreaterThan(0);
    expect(L.every((r) => r.every(Number.isFinite))).toBe(true);
  });
});

describe("evaluateParlay", () => {
  it("una leg continua coincide con el marginal analitico", () => {
    const r = evaluateParlay(inputs(), [over("A", 110)]);
    expect(r.legs[0].p).toBeCloseTo(1 - normCdf(0.5), 1); // 0.3085
    expect(Math.abs(r.legs[0].p - (1 - normCdf(0.5)))).toBeLessThan(0.012);
  });
  it("margen: ganador local = Phi(mu/sigma)", () => {
    const r = evaluateParlay(inputs(), [{ kind: "winner", team: "home" }]);
    expect(Math.abs(r.legs[0].p - normCdf(3 / 13))).toBeLessThan(0.012);
  });
  it("TD anytime coincide con Poisson(lambda_rec + lambda_rush)", () => {
    const r = evaluateParlay(inputs(), [{ kind: "anytime_td", gsis_id: "A" }]);
    expect(Math.abs(r.legs[0].p - (1 - Math.exp(-0.7)))).toBeLessThan(0.012);
    const r2 = evaluateParlay(inputs(), [{ kind: "two_td", gsis_id: "A" }]);
    expect(Math.abs(r2.legs[0].p - (1 - poissonCdfAt(0.7, 1)))).toBeLessThan(0.01);
  });
  it("correlacion cero: conjunta ~ producto de marginales", () => {
    const r = evaluateParlay(inputs(0), [over("A", 100), over("B", 100)]);
    expect(Math.abs(r.joint - r.independent)).toBeLessThan(0.015);
    expect(Math.abs(r.joint - 0.25)).toBeLessThan(0.015);
  });
  it("correlacion positiva aumenta P(ambos over)", () => {
    const r = evaluateParlay(inputs(0.7), [over("A", 100), over("B", 100)]);
    expect(r.joint).toBeGreaterThan(r.independent + 0.08);
  });
  it("es determinista y calcula cuotas y EV", () => {
    const a = evaluateParlay(inputs(), [over("A", 100)], { bookDecimal: 2.2 });
    const b = evaluateParlay(inputs(), [over("A", 100)], { bookDecimal: 2.2 });
    expect(a.joint).toBe(b.joint);
    expect(a.fairDecimal).toBeCloseTo(1 / a.joint, 10);
    expect(a.ev).toBeCloseTo(a.joint * 2.2 - 1, 10);
  });
  it("primer TD es aproximado y se marca; leg sin datos no cuenta", () => {
    const r = evaluateParlay(inputs(), [{ kind: "first_td", gsis_id: "A" }, over("Z", 10)]);
    expect(r.approx).toBe(true);
    expect(r.legs[0].p).toBeGreaterThan(0);
    expect(r.legs[1].available).toBe(false);
  });
});

describe("utilidades", () => {
  it("cuotas", () => {
    expect(toAmerican(0.5)).toBe(-100);
    expect(toAmerican(0.25)).toBe(300);
    expect(parseOdds("+150")).toBeCloseTo(2.5);
    expect(parseOdds("-110")).toBeCloseTo(1.9091, 3);
    expect(parseOdds("2.4")).toBe(2.4);
    expect(parseOdds("abc")).toBeNull();
    expect(parseOdds("0.5")).toBeNull();
  });
  it("codifica y decodifica legs", () => {
    const l = over("00-0033873", 57.5);
    expect(decodeLeg(legKey(l))).toEqual(l);
    expect(decodeAdd("00-1:receiving_yards:over:57.5")).toEqual({ ...l, gsis_id: "00-1" });
    expect(decodeAdd("x")).toBeNull();
    expect(decodeLeg("p:a:b:sideways:1")).toBeNull();
  });
});
