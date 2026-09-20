import { NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export async function GET() {
  const sql = getSql();
  // Subconsulta en una sola sentencia en vez de round-tripear el timestamp
  // maximo por JS: convertirlo a string y pasarlo de vuelta como parametro
  // puede perder precision y no volver a matchear con el valor original.
  const features = await sql.query(
    `SELECT feature_name, psi, kl_divergence, reference_mean, current_mean, is_drifted
     FROM drift_metrics
     WHERE checked_at = (SELECT max(checked_at) FROM drift_metrics)
     ORDER BY psi DESC`
  );

  const checkedAtRows = await sql.query(
    "SELECT max(checked_at) AS checked_at FROM drift_metrics"
  );
  const checkedAt = (checkedAtRows as { checked_at: string | null }[])[0]?.checked_at ?? null;

  return NextResponse.json({ checked_at: checkedAt, features });
}
