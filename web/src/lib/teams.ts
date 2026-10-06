export const TEAMS = [
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
  "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
  "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
  "TEN", "WAS",
];

export const TEAM_COLORS: Record<string, string> = {
  ARI: "#97233F", ATL: "#A71930", BAL: "#241773", BUF: "#00338D", CAR: "#0085CA",
  CHI: "#0B162A", CIN: "#FB4F14", CLE: "#311D00", DAL: "#041E42", DEN: "#FB4F14",
  DET: "#0076B6", GB: "#203731", HOU: "#03202F", IND: "#002C5F", JAX: "#006778",
  KC: "#E31837", LA: "#003594", LAC: "#0080C6", LV: "#000000", MIA: "#008E97",
  MIN: "#4F2683", NE: "#002244", NO: "#D3BC8D", NYG: "#0B2265", NYJ: "#125740",
  PHI: "#004C54", PIT: "#FFB612", SEA: "#69BE28", SF: "#AA0000", TB: "#D50A0A",
  TEN: "#4B92DB", WAS: "#5A1414",
};

// Variantes legibles sobre fondo claro y oscuro: aclara los casi negros y
// oscurece los muy claros.
const READABLE_COLORS: Record<string, string> = {
  LV: "#6b7480", CHI: "#3d5a8c", DAL: "#4a6fb5", HOU: "#3a6a85", NE: "#3a5f9a",
  BAL: "#5b42b8", WAS: "#8a3a3a", CLE: "#8a5a2b", GB: "#3d7a5c",
  NO: "#9a7f45", PIT: "#b5830a", SEA: "#3f8a16", TEN: "#2f76c0",
};

export function teamColor(abbr: string | null | undefined): string {
  if (!abbr) return "#6b7280";
  return READABLE_COLORS[abbr] ?? TEAM_COLORS[abbr] ?? "#6b7280";
}
