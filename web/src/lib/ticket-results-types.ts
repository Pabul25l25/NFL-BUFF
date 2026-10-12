export interface TicketResultGame {
  game_id: string;
  status: "final" | "pending";
  home_team: string | null;
  away_team: string | null;
  home_score: number | null;
  away_score: number | null;
}

export interface TicketResultStat {
  game_id: string;
  gsis_id: string;
  receiving_yards: number | null;
  rushing_yards: number | null;
  receptions: number | null;
  receiving_tds: number | null;
  rushing_tds: number | null;
  passing_yards: number | null;
  passing_tds: number | null;
  sacks_suffered: number | null;
}

export interface TicketResultsResponse {
  games: TicketResultGame[];
  /** Solo para partidos finalizados. */
  stats: TicketResultStat[];
  /** null: no hay play-by-play ingerido. */
  first_td: { game_id: string; gsis_id: string }[] | null;
}
