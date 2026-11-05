import { getSql } from "@/lib/db";
import { ok, serverError } from "@/lib/api";

export async function GET() {
  try {
  const rows = await getSql().query(
    `SELECT occurred_at, target, action, model_id, previous_model_id, reason
     FROM retraining_log
     ORDER BY occurred_at DESC
     LIMIT 50`
  );

  return ok({ log: rows });
  } catch {
    return serverError();
  }
}
