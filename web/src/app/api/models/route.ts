import { getSql } from "@/lib/db";
import { ok, serverError } from "@/lib/api";

export async function GET() {
  try {
  const rows = await getSql().query(
    `SELECT model_id, target, family, version, status, metric_name,
            train_metric, val_metric, train_val_gap, n_train_rows,
            train_seasons, created_at, promotion_blocked_reason
     FROM model_registry
     ORDER BY target, version DESC
     LIMIT 500`
  );

  return ok({ models: rows });
  } catch {
    return serverError();
  }
}
