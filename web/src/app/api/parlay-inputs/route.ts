import { NextRequest, NextResponse } from "next/server";
import { getSql } from "@/lib/db";
import { GAME_ID_RE } from "@/lib/parlay";


/** Las columnas JSON se publican como VARCHAR; el driver puede devolverlas ya parseadas. */
function parseJson<T>(value: unknown): T | null {
  if (value == null) return null;
  if (typeof value !== "string") return value as T;
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

function num(value: unknown): number | null {
  if (value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export async function GET(req: NextRequest) {
  const gameId = new URL(req.url).searchParams.get("game_id") ?? "";
  if (!GAME_ID_RE.test(gameId)) {
    return NextResponse.json({ error: "game_id invalido" }, { status: 400 });
  }

  const sql = getSql();

  // Datos del partido: un solo SELECT; las piernas de parlay salen de su tabla.
  const games = await sql.query(
    `SELECT g.game_id, g.season, g.week, g.home_team, g.away_team, g.kickoff_utc,
            l.p_home_win, l.margin_mu, l.margin_sigma, l.total_mu, l.total_sigma
     FROM (
       SELECT DISTINCT ON (game_id) game_id, season, week, home_team, away_team, kickoff_utc
       FROM pub_predictions_games WHERE game_id = $1
     ) g
     LEFT JOIN pub_parlay_game_legs l ON l.game_id = g.game_id`,
    [gameId]
  );
  if (games.length === 0) {
    return NextResponse.json({ error: "partido desconocido" }, { status: 404 });
  }
  const g = games[0];
  if (g.p_home_win == null) {
    return NextResponse.json(
      { error: "sin entradas de parlay publicadas para este partido" },
      { status: 404 }
    );
  }

  const [propRows, corrRows, calRows] = await Promise.all([
    sql.query(
      `SELECT gsis_id, player_name, position, team, is_home, target,
              point_estimate, quantiles, poisson_lambda
       FROM pub_predictions_props
       WHERE game_id = $1
       ORDER BY player_name, target`,
      [gameId]
    ),
    sql.query(`SELECT legs, matrix FROM pub_parlay_game_corr WHERE game_id = $1`, [gameId]),
    sql.query(
      `SELECT ece, n FROM pub_parlay_calibration ORDER BY created_at DESC LIMIT 1`
    ),
  ]);

  const players: Record<string, unknown>[] = [];
  const seen = new Set<string>();
  for (const r of propRows) {
    if (seen.has(r.gsis_id) || r.team == null || r.is_home == null) continue;
    seen.add(r.gsis_id);
    players.push({
      gsis_id: r.gsis_id,
      player_name: r.player_name ?? r.gsis_id,
      position: r.position ?? null,
      team: r.team,
      is_home: Boolean(r.is_home),
    });
  }

  const markets = propRows
    .filter((r) => seen.has(r.gsis_id))
    .map((r) => {
      const q = parseJson<number[]>(r.quantiles);
      return {
        gsis_id: r.gsis_id,
        target: r.target,
        point_estimate: num(r.point_estimate),
        quantiles: Array.isArray(q) && q.length === 19 ? q.map(Number) : null,
        lambda: num(r.poisson_lambda),
      };
    });

  const corr = corrRows.length
    ? {
        legs: parseJson<string[]>(corrRows[0].legs) ?? [],
        matrix: parseJson<number[][]>(corrRows[0].matrix) ?? [],
      }
    : { legs: [], matrix: [] };

  const calibration = calRows.length
    ? { ece: Number(calRows[0].ece), n: Number(calRows[0].n) }
    : null;

  return NextResponse.json({
    game: {
      game_id: g.game_id,
      season: Number(g.season),
      week: Number(g.week),
      home_team: g.home_team,
      away_team: g.away_team,
      kickoff_utc: g.kickoff_utc,
      p_home_win: Number(g.p_home_win),
      margin_mu: Number(g.margin_mu),
      margin_sigma: Number(g.margin_sigma),
      total_mu: Number(g.total_mu),
      total_sigma: Number(g.total_sigma),
    },
    players,
    markets,
    corr,
    calibration,
  });
}
