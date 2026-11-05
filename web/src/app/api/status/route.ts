import { getSql } from "@/lib/db";
import { CACHE_1MIN, ok, serverError } from "@/lib/api";

export async function GET() {
  try {
  const rows = await getSql().query(
    "SELECT table_name, synced_at, row_count FROM sync_meta ORDER BY table_name LIMIT 200"
  );
  return ok({ sync: rows }, CACHE_1MIN);
  } catch {
    return serverError();
  }
}
