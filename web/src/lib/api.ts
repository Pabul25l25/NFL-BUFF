import { NextResponse } from "next/server";

export const CACHE_5MIN = "public, s-maxage=300, stale-while-revalidate=3600";
export const CACHE_1MIN = "public, s-maxage=60, stale-while-revalidate=3600";

export const GAME_TARGETS = ["winner", "spread", "total"];
export const PROP_TARGETS = [
  "receiving_yards", "rushing_yards", "receptions", "receiving_tds",
  "rushing_tds", "passing_yards", "passing_tds", "sacks_suffered",
];

export const NUM_RE = /^\d{1,4}$/;
export const TEAM_RE = /^[A-Z]{2,3}$/;

export function bad(message: string) {
  return NextResponse.json({ error: message }, { status: 400 });
}

export function ok(body: unknown, cache: string = CACHE_5MIN) {
  return NextResponse.json(body, { headers: { "Cache-Control": cache } });
}

/** Error generico: el detalle nunca sale al cliente. */
export function serverError() {
  return NextResponse.json({ error: "error interno" }, { status: 500 });
}
