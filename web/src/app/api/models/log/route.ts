import { NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export async function GET() {
  const rows = await getSql().query(
    `SELECT occurred_at, target, action, model_id, previous_model_id, reason
     FROM retraining_log
     ORDER BY occurred_at DESC
     LIMIT 50`
  );

  return NextResponse.json({ log: rows });
}
