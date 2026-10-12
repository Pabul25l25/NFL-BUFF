import { Leg, ParlayInputs, Side, halfLine, hasMarket, tdRate } from "@/lib/parlay";

export const MARKETS: { key: string; label: string }[] = [
  { key: "receiving_yards", label: "Yds recepción" },
  { key: "rushing_yards", label: "Yds carrera" },
  { key: "receptions", label: "Recepciones" },
  { key: "passing_yards", label: "Yds pase" },
  { key: "passing_tds", label: "TD pase" },
  { key: "sacks_suffered", label: "Capturas" },
  { key: "anytime_td", label: "Anota TD" },
  { key: "two_td", label: "2+ TD" },
  { key: "first_td", label: "Primer TD" },
];
export const marketLabel = (k: string) => MARKETS.find((m) => m.key === k)?.label ?? k;

export function playerName(inputs: ParlayInputs, id: string) {
  return inputs.players.find((p) => p.gsis_id === id)?.player_name ?? id;
}

export const sideText = (s: Side) => (s === "over" ? "Más de" : "Menos de");
export const fmtLine = (n: number) => (n > 0 ? `+${n}` : `${n}`);

export interface Teams {
  home: string;
  away: string;
}

export function describeLeg(leg: Leg, inputs: ParlayInputs | undefined, g: Teams): string {
  const name = (id: string) => (inputs ? playerName(inputs, id) : id);
  switch (leg.kind) {
    case "player":
      return `${name(leg.gsis_id)} · ${marketLabel(leg.target)} · ${sideText(leg.side)} ${leg.line}`;
    case "anytime_td":
      return `${name(leg.gsis_id)} · Anota TD`;
    case "two_td":
      return `${name(leg.gsis_id)} · 2+ TD`;
    case "first_td":
      return `${name(leg.gsis_id)} · Primer TD`;
    case "winner":
      return `Gana ${leg.team === "home" ? g.home : g.away}`;
    case "spread":
      return `Spread · ${leg.team === "home" ? g.home : g.away} ${fmtLine(leg.line)}`;
    case "total":
      return `Total · ${sideText(leg.side)} ${leg.line}`;
  }
}

/** Mercados disponibles para un jugador (segun los datos del modelo). */
export function marketsFor(inputs: ParlayInputs, gsis: string): string[] {
  return MARKETS.filter((m) => {
    if (m.key === "anytime_td" || m.key === "two_td" || m.key === "first_td") return tdRate(inputs, gsis) != null;
    return hasMarket(inputs, gsis, m.key);
  }).map((m) => m.key);
}

export function defaultPlayerLine(inputs: ParlayInputs, gsis: string, target: string): number {
  const m = inputs.markets.find((x) => x.gsis_id === gsis && x.target === target);
  const v = m?.point_estimate ?? m?.lambda ?? 0;
  return halfLine(v);
}

export const pct = (p: number, d = 1) => `${(p * 100).toFixed(d)} %`;
export const gameLabel = (g: { away_team: string; home_team: string }) => `${g.away_team} @ ${g.home_team}`;
