import { NextRequest, NextResponse } from "next/server";
import { getSql } from "@/lib/db";
import { GAME_ID_RE } from "@/lib/parlay";
import { serverError } from "@/lib/api";
import type { TicketResultsResponse } from "@/lib/ticket-results-types";

const MAX_IDS = 10;

function num(value: unknown): number | null {
  if (value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export async function GET(req: NextRequest) {
  try {
    return await handle(req);
  } catch {
    return serverError();
  }
}

async function handle(req: NextRequest) {
  const raw = new URL(req.url).searchParams.get("game_ids") ?? "";
  const ids = Array.from(new Set(raw.split(",").map((s) => s.trim())));
  if (ids.length < 1 || ids.length > MAX_IDS || !ids.every((id) => GAME_ID_RE.test(id))) {
    return NextResponse.json(
      { error: `game_ids invalido: de 1 a ${MAX_IDS} ids con formato AAAA_SS_AWAY_HOME` },
      { status: 400 }
    );
  }

  const sql = getSql();

  // Una fila por partido (la tabla tiene una por target).
  const gameRows = await sql.query(
    `SELECT DISTINCT ON (game_id) game_id, home_team, away_team, home_score, away_score
     FROM pub_predictions_games
     WHERE game_id = ANY($1)
     ORDER BY game_id`,
    [ids]
  );

  const games = gameRows.map((r) => {
    const home = num(r.home_score);
    const away = num(r.away_score);
    return {
      game_id: r.game_id as string,
      status: home != null && away != null ? ("final" as const) : ("pending" as const),
      home_team: (r.home_team as string) ?? null,
      away_team: (r.away_team as string) ?? null,
      home_score: home,
      away_score: away,
    };
  });

  const finalIds = games.filter((g) => g.status === "final").map((g) => g.game_id);

  let stats: TicketResultsResponse["stats"] = [];
  if (finalIds.length > 0) {
    // Las estadisticas reales se repiten por target; basta una fila por jugador.
    const statRows = await sql.query(
      `SELECT DISTINCT ON (game_id, gsis_id)
              game_id, gsis_id, receiving_yards, rushing_yards, receptions,
              receiving_tds, rushing_tds, passing_yards, passing_tds, sacks_suffered
       FROM pub_predictions_props
       WHERE game_id = ANY($1)
       ORDER BY game_id, gsis_id`,
      [finalIds]
    );
    stats = statRows.map((r) => ({
      game_id: r.game_id,
      gsis_id: r.gsis_id,
      receiving_yards: num(r.receiving_yards),
      rushing_yards: num(r.rushing_yards),
      receptions: num(r.receptions),
      receiving_tds: num(r.receiving_tds),
      rushing_tds: num(r.rushing_tds),
      passing_yards: num(r.passing_yards),
      passing_tds: num(r.passing_tds),
      sacks_suffered: num(r.sacks_suffered),
    }));
  }

  const body: TicketResultsResponse = { games, stats, first_td: null };
  const allFinal = games.length === ids.length && finalIds.length === ids.length;
  return NextResponse.json(body, {
    headers: {
      "Cache-Control": allFinal
        ? "public, s-maxage=600, stale-while-revalidate=3600"
        : "public, s-maxage=30, stale-while-revalidate=60",
    },
  });
}
