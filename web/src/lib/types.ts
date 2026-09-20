export type Target = "winner" | "spread" | "total";

export interface GamePrediction {
  game_id: string;
  target: Target;
  point_estimate: number | null;
  probability: number | null;
  confidence_level: "alta" | "media" | "baja" | null;
  explanation_es: string | null;
  shap_values: string | null;
  predicted_at: string | null;
  as_of_ts: string | null;
  season: number;
  week: number;
  season_type: string;
  gameday: string;
  kickoff_utc: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  actual_margin: number | null;
  actual_total: number | null;
  roof: string | null;
  surface: string | null;
  stadium: string | null;
  model_version: number | null;
  model_family: string | null;
  temperature_c: number | null;
  wind_kph: number | null;
  precipitation_mm: number | null;
}

export interface PropPrediction {
  game_id: string;
  gsis_id: string;
  target: string;
  point_estimate: number | null;
  lower_80: number | null;
  upper_80: number | null;
  explanation_es: string | null;
  shap_values: string | null;
  predicted_at: string | null;
  as_of_ts: string | null;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  kickoff_utc: string;
  player_name: string;
  position: string | null;
  team: string;
  receiving_yards: number | null;
  rushing_yards: number | null;
  receptions: number | null;
  receiving_tds: number | null;
  rushing_tds: number | null;
  passing_yards: number | null;
  passing_tds: number | null;
  sacks_suffered: number | null;
}

export const STAT_DEFS: { key: string; label: string; sparse?: boolean }[] = [
  { key: "receiving_yards", label: "Recepción" },
  { key: "rushing_yards", label: "Carrera" },
  { key: "receptions", label: "Recepciones" },
  { key: "receiving_tds", label: "TD recep.", sparse: true },
  { key: "rushing_tds", label: "TD carrera", sparse: true },
  { key: "passing_yards", label: "Pase" },
  { key: "passing_tds", label: "TD pase", sparse: true },
  { key: "sacks_suffered", label: "Capturas", sparse: true },
];

export function pickLabel(pred: GamePrediction): string {
  if (pred.target === "winner") {
    const winner = (pred.probability ?? 0) >= 0.5 ? pred.home_team : pred.away_team;
    const prob = pred.probability ?? 0;
    const pct = Math.round((prob >= 0.5 ? prob : 1 - prob) * 100);
    return `${winner} · ${pct}%`;
  }
  if (pred.target === "spread") {
    const margin = pred.point_estimate ?? 0;
    const favored = margin >= 0 ? pred.home_team : pred.away_team;
    return `${favored} ${margin >= 0 ? "-" : "+"}${Math.abs(margin).toFixed(1)}`;
  }
  return `${(pred.point_estimate ?? 0).toFixed(1)} pts`;
}

export function displayProb(pred: GamePrediction): number {
  if (pred.target === "winner") {
    const prob = pred.probability ?? 0;
    return Math.round((prob >= 0.5 ? prob : 1 - prob) * 100);
  }
  return Math.round((pred.probability ?? 0) * 100);
}
