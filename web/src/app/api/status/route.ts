import { NextResponse } from "next/server";
import { getSql } from "@/lib/db";

export async function GET() {
  const rows = await getSql().query(
    "SELECT table_name, synced_at, row_count FROM sync_meta ORDER BY table_name"
  );
  return NextResponse.json({ sync: rows });
}
