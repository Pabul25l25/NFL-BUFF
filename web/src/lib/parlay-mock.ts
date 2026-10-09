/**
 * Fixture LOCAL para desarrollar /picks sin la API. Solo se usa en desarrollo cuando
 * /api/parlay-inputs responde 404; la pagina lo marca como "mock" y en produccion nunca se importa
 * como dato real. Todos los numeros son sinteticos.
 */
import { ParlayInputs, ParlayMarket, ParlayPlayer, normCdf } from "./parlay";

export const MOCK_ENABLED = process.env.NODE_ENV !== "production";

const Z = [
  -1.6449, -1.2816, -1.0364, -0.8416, -0.6745, -0.5244, -0.3853, -0.2533, -0.1257, 0, 0.1257, 0.2533, 0.3853,
  0.5244, 0.6745, 0.8416, 1.0364, 1.2816, 1.6449,
];

/** Cuantiles de una lognormal con media `mean` y dispersion `s`. */
function lognormalQuantiles(mean: number, s: number): number[] {
  return Z.map((z) => Math.round(mean * Math.exp(s * z - (s * s) / 2) * 10) / 10);
}

const ROSTER: { n: string; pos: string; rec?: number; rush?: number; pass?: number; recs?: number; td: [number, number] }[] = [
  { n: "QB1", pos: "QB", pass: 255, rush: 14, td: [0, 0.12] },
  { n: "RB1", pos: "RB", rush: 70, rec: 22, recs: 2.6, td: [0.12, 0.42] },
  { n: "RB2", pos: "RB", rush: 28, rec: 9, recs: 1.2, td: [0.05, 0.18] },
  { n: "WR1", pos: "WR", rec: 82, recs: 6.1, td: [0.42, 0] },
  { n: "WR2", pos: "WR", rec: 58, recs: 4.4, td: [0.32, 0] },
  { n: "WR3", pos: "WR", rec: 38, recs: 3, td: [0.2, 0] },
  { n: "TE1", pos: "TE", rec: 44, recs: 4, td: [0.3, 0] },
];

export function mockParlayInputs(g: {
  game_id: string;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  kickoff_utc: string;
  p_home_win: number | null;
}): ParlayInputs {
  const p = Math.min(0.95, Math.max(0.05, g.p_home_win ?? 0.55));
  // inversa de Phi por biseccion (solo para el fixture)
  let lo = -4;
  let hi = 4;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (normCdf(mid) < p) lo = mid;
    else hi = mid;
  }
  const sigma = 13.5;
  const players: ParlayPlayer[] = [];
  const markets: ParlayMarket[] = [];
  const legs: string[] = ["g:margin", "g:total"];

  for (const home of [true, false]) {
    const team = home ? g.home_team : g.away_team;
    const tilt = home ? 1 : 0.94;
    ROSTER.forEach((r, i) => {
      const id = `MOCK-${team}-${i + 1}`;
      players.push({ gsis_id: id, player_name: `${team} ${r.n}`, position: r.pos, team, is_home: home });
      const cont = (target: string, mean: number | undefined, s: number) => {
        if (mean == null) return;
        markets.push({
          gsis_id: id,
          target,
          point_estimate: Math.round(mean * tilt * 10) / 10,
          quantiles: lognormalQuantiles(mean * tilt, s),
          lambda: null,
        });
        legs.push(`p:${id}:${target}`);
      };
      cont("receiving_yards", r.rec, 0.62);
      cont("rushing_yards", r.rush, 0.6);
      cont("receptions", r.recs, 0.5);
      cont("passing_yards", r.pass, 0.28);
      const count = (target: string, lam: number) => {
        if (lam <= 0) return;
        markets.push({ gsis_id: id, target, point_estimate: lam, quantiles: null, lambda: lam });
      };
      count("receiving_tds", r.td[0] * tilt);
      count("rushing_tds", r.td[1] * tilt);
      if (r.pos === "QB") count("passing_tds", 1.5 * tilt);
      if (r.pos === "QB") count("sacks_suffered", 2.3);
    });
  }

  const n = legs.length;
  const matrix: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  const set = (a: string, b: string, v: number) => {
    const i = legs.indexOf(a);
    const j = legs.indexOf(b);
    if (i >= 0 && j >= 0 && i !== j) matrix[i][j] = matrix[j][i] = v;
  };
  for (const home of [true, false]) {
    const team = home ? g.home_team : g.away_team;
    const qb = `p:MOCK-${team}-1:passing_yards`;
    // el equipo que anota mas gana: margen positivo ayuda al local, negativo al visitante
    set("g:margin", qb, home ? 0.25 : -0.25);
    set("g:total", qb, 0.3);
    for (let i = 2; i <= 7; i++) {
      const rec = `p:MOCK-${team}-${i}:receiving_yards`;
      set(qb, rec, 0.45);
      set("g:total", rec, 0.2);
      set("g:margin", rec, home ? 0.1 : -0.1);
      for (let j = i + 1; j <= 7; j++) set(rec, `p:MOCK-${team}-${j}:receiving_yards`, -0.05);
    }
  }

  return {
    game: {
      game_id: g.game_id,
      season: g.season,
      week: g.week,
      home_team: g.home_team,
      away_team: g.away_team,
      kickoff_utc: g.kickoff_utc,
      p_home_win: p,
      margin_mu: lo * sigma,
      margin_sigma: sigma,
      total_mu: 45.5,
      total_sigma: 10.5,
    },
    players,
    markets,
    corr: { legs, matrix },
    calibration: null,
  };
}
