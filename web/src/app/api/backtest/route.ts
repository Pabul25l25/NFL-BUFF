import { NextRequest, NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const target = searchParams.get("target") ?? "winner";

  const sql = getSql();
  const runRows = await sql.query(
    "SELECT run_id FROM backtest_results WHERE target = $1 ORDER BY created_at DESC LIMIT 1",
    [target]
  );
  const runId = (runRows as { run_id: string }[])[0]?.run_id;
  if (!runId) {
    return NextResponse.json({ overall: [], seasons: [] });
  }

  const overall = await sql.query(
    `SELECT model_label, target, n_games, log_loss, brier_score, ece, accuracy,
            mae, rmse, crps, ats_win_pct, beats_market
     FROM backtest_results
     WHERE run_id = $1 AND target = $2 AND segment_type = 'overall'
     ORDER BY model_label`,
    [runId, target]
  );
  const seasons = await sql.query(
    `SELECT segment_value, n_games, log_loss, brier_score, accuracy, mae, rmse
     FROM backtest_results
     WHERE run_id = $1 AND target = $2 AND segment_type = 'season'
     ORDER BY segment_value`,
    [runId, target]
  );

  return NextResponse.json({ overall, seasons });
}
