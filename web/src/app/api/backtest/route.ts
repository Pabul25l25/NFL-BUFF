import { NextRequest } from "next/server";
import { getSql } from "@/lib/db";
import { GAME_TARGETS, PROP_TARGETS, bad, ok, serverError } from "@/lib/api";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const target = searchParams.get("target") ?? "winner";

  if (![...GAME_TARGETS, ...PROP_TARGETS].includes(target)) return bad("target invalido");

  try {
  const sql = getSql();
  const runRows = await sql.query(
    "SELECT run_id FROM backtest_results WHERE target = $1 ORDER BY created_at DESC LIMIT 1",
    [target]
  );
  const runId = (runRows as { run_id: string }[])[0]?.run_id;
  if (!runId) {
    return ok({ overall: [], seasons: [] });
  }

  const overall = await sql.query(
    `SELECT model_label, target, n_games, log_loss, brier_score, ece, accuracy,
            mae, rmse, crps, ats_win_pct, beats_market
     FROM backtest_results
     WHERE run_id = $1 AND target = $2 AND segment_type = 'overall'
     ORDER BY model_label
     LIMIT 200`,
    [runId, target]
  );
  const seasons = await sql.query(
    `SELECT segment_value, n_games, log_loss, brier_score, accuracy, mae, rmse
     FROM backtest_results
     WHERE run_id = $1 AND target = $2 AND segment_type = 'season'
     ORDER BY segment_value
     LIMIT 2000`,
    [runId, target]
  );

  return ok({ overall, seasons });
  } catch {
    return serverError();
  }
}
