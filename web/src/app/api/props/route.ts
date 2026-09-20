import { NextRequest, NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const target = searchParams.get("target") ?? "receiving_yards";
  const season = searchParams.get("season");
  const week = searchParams.get("week");
  const team = searchParams.get("team");
  const gameId = searchParams.get("game_id");
  const gsisId = searchParams.get("gsis_id");

  const conditions: string[] = ["target = $1"];
  const params: (string | number)[] = [target];

  if (gameId) {
    params.push(gameId);
    conditions.push(`game_id = $${params.length}`);
  }
  if (gsisId) {
    params.push(gsisId);
    conditions.push(`gsis_id = $${params.length}`);
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
    conditions.push(`team = $${params.length}`);
  }

  const query = `
    SELECT game_id, gsis_id, target, point_estimate, lower_80, upper_80,
           explanation_es, shap_values, predicted_at, as_of_ts, season, week,
           home_team, away_team, kickoff_utc, player_name, position, team,
           receiving_yards, rushing_yards, receptions, receiving_tds,
           rushing_tds, passing_yards, passing_tds, sacks_suffered
    FROM pub_predictions_props
    WHERE ${conditions.join(" AND ")}
    ORDER BY point_estimate DESC
  `;

  const rows = await getSql().query(query, params);
  return NextResponse.json({ props: rows });
}
