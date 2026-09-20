import { NextRequest, NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const target = searchParams.get("target") ?? "winner";

  const rows = await getSql().query(
    `SELECT pp.season, pp.week, pp.n_predictions, pp.log_loss, pp.brier_score,
            pp.accuracy, pp.mae, r.version, r.family
     FROM production_performance pp
     LEFT JOIN model_registry r ON r.model_id = pp.model_id
     WHERE pp.target = $1
     ORDER BY pp.season, pp.week`,
    [target]
  );

  return NextResponse.json({ performance: rows });
}
