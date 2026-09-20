import { NextRequest, NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const target = searchParams.get("target") ?? "winner";

  const sql = getSql();
  const runRows = await sql.query(
    "SELECT run_id FROM calibration_bins WHERE target = $1 LIMIT 1",
    [target]
  );
  const runId = (runRows as { run_id: string }[])[0]?.run_id;
  if (!runId) {
    return NextResponse.json({ bins: [] });
  }

  const bins = await sql.query(
    `SELECT model_label, bin_index, bin_lower, bin_upper, mean_predicted,
            observed_frequency, n_samples
     FROM calibration_bins
     WHERE run_id = $1 AND target = $2
     ORDER BY model_label, bin_index`,
    [runId, target]
  );

  return NextResponse.json({ bins });
}
