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
  opponent_team: string | null;
  is_home: boolean | null;
  quantiles?: string | null;
  poisson_lambda?: number | null;
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

/** Equipo favorito segun el target; null si no aplica (total) o falta el dato. */
export function favoriteTeam(pred: GamePrediction): string | null {
  if (pred.target === "winner") {
    if (pred.probability == null) return null;
    return pred.probability >= 0.5 ? pred.home_team : pred.away_team;
  }
  if (pred.target === "spread") {
    if (pred.point_estimate == null) return null;
    return pred.point_estimate >= 0 ? pred.home_team : pred.away_team;
  }
  return null;
}

export function pickLabel(pred: GamePrediction): string {
  if (pred.target === "total") {
    return pred.point_estimate == null ? "—" : `${pred.point_estimate.toFixed(1)} pts`;
  }
  const fav = favoriteTeam(pred);
  if (!fav) return "—";
  if (pred.target === "winner") {
    return `${fav} · ${displayProb(pred)}%`;
  }
  const margin = pred.point_estimate ?? 0;
  return `${fav} ${margin >= 0 ? "-" : "+"}${Math.abs(margin).toFixed(1)}`;
}

/** Probabilidad en %, o null si falta (spread/total suelen no tenerla). Winner: la del favorito. */
export function displayProb(pred: GamePrediction): number | null {
  if (pred.probability == null) return null;
  if (pred.target === "winner") {
    const prob = pred.probability;
    return Math.round((prob >= 0.5 ? prob : 1 - prob) * 100);
  }
  return Math.round(pred.probability * 100);
}
