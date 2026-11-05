import { NextRequest, NextResponse } from "next/server";
import { getSql } from "@/lib/db";
import { GAME_ID_RE } from "@/lib/parlay";
import { CACHE_5MIN, GAME_TARGETS, NUM_RE, TEAM_RE, bad, serverError } from "@/lib/api";

const MAX_ROWS = 2000;

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const target = searchParams.get("target") ?? "winner";
  const season = searchParams.get("season");
  const week = searchParams.get("week");
  const team = searchParams.get("team");
  const gameId = searchParams.get("game_id");

  if (!GAME_TARGETS.includes(target)) return bad("target invalido");
  if (season && !NUM_RE.test(season)) return bad("season invalido");
  if (week && !NUM_RE.test(week)) return bad("week invalido");
  if (team && !TEAM_RE.test(team)) return bad("team invalido");
  if (gameId && !GAME_ID_RE.test(gameId)) return bad("game_id invalido");

  const conditions: string[] = ["target = $1"];
  const params: (string | number)[] = [target];

  if (gameId) {
    params.push(gameId);
    conditions.push(`game_id = $${params.length}`);
  }
  if (season) {
    params.push(Number(season));
    conditions.push(`season = $${params.length}`);
  }
  if (week) {
    params.push(Number(week));
    conditions.push(`week = $${params.length}`);
  }
  if (team) {
    params.push(team);
    conditions.push(`(home_team = $${params.length} OR away_team = $${params.length})`);
  }

  // Si hay mas filas que el tope se conservan las mas recientes (no las mas antiguas),
  // y el resultado se devuelve en orden cronologico como antes.
  const query = `
    SELECT * FROM (
      SELECT game_id, target, point_estimate, probability, confidence_level,
             explanation_es, shap_values, predicted_at, as_of_ts, season, week,
             season_type, gameday, kickoff_utc, home_team, away_team, home_score,
             away_score, actual_margin, actual_total, roof, surface, stadium,
             model_version, model_family, temperature_c, wind_kph, precipitation_mm
      FROM pub_predictions_games
      WHERE ${conditions.join(" AND ")}
      ORDER BY kickoff_utc DESC
      LIMIT ${MAX_ROWS}
    ) t
    ORDER BY kickoff_utc ASC
  `;

  try {
    const rows = await getSql().query(query, params);
    return NextResponse.json({ predictions: rows }, { headers: { "Cache-Control": CACHE_5MIN } });
  } catch {
    return serverError();
  }
}
