import { NextResponse } from "next/server";
import { sql } from "@/lib/db";

export async function GET() {
  const rows = await sql.query(
    "SELECT table_name, synced_at, row_count FROM sync_meta ORDER BY table_name"
  );
  return NextResponse.json({ sync: rows });
}
