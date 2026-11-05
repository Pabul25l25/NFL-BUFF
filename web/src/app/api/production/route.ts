import { NextRequest } from "next/server";
import { getSql } from "@/lib/db";
import { GAME_TARGETS, PROP_TARGETS, bad, ok, serverError } from "@/lib/api";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const target = searchParams.get("target") ?? "winner";

  if (![...GAME_TARGETS, ...PROP_TARGETS].includes(target)) return bad("target invalido");

  try {
  const rows = await getSql().query(
    `SELECT pp.season, pp.week, pp.n_predictions, pp.log_loss, pp.brier_score,
            pp.accuracy, pp.mae, r.version, r.family
     FROM production_performance pp
     LEFT JOIN model_registry r ON r.model_id = pp.model_id
     WHERE pp.target = $1
     ORDER BY pp.season, pp.week
     LIMIT 1000`,
    [target]
  );

  return ok({ performance: rows });
  } catch {
    return serverError();
  }
}
