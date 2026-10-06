import Link from "next/link";
import { teamColor } from "@/lib/teams";
import { GamePrediction, displayProb, favoriteTeam, pickLabel } from "@/lib/types";

/** Color de tinte: el del favorito; sin favorito (total) usa el acento. */
export function gameTint(g: GamePrediction): string {
  const fav = favoriteTeam(g);
  return fav ? teamColor(fav) : "var(--accent)";
}

export function ConfidenceTag({ level }: { level: GamePrediction["confidence_level"] }) {
  if (!level) return null;
  const color =
    level === "alta" ? "var(--text)" : level === "media" ? "var(--text-2)" : "var(--text-3)";
  return (
    <span className="text-[10.5px] font-semibold uppercase tracking-wide" style={{ color }}>
      {level}
    </span>
  );
}

function meta(g: GamePrediction): string {
  if (g.roof === "dome" || g.roof === "closed") return `${g.stadium ?? ""} · techo cerrado`;
  const w =
    g.temperature_c != null
      ? ` · ${Math.round(g.temperature_c)}°C · ${Math.round(g.wind_kph ?? 0)} km/h`
      : "";
  return `${g.stadium ?? ""}${w}`;
}

export function GameCard({ g }: { g: GamePrediction }) {
  const fav = favoriteTeam(g);
  const prob = displayProb(g);
  const when = new Date(g.kickoff_utc).toLocaleString("es-MX", {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
  return (
    <Link
      href={`/partido/${g.game_id}?target=${g.target}`}
      className="tint-card block p-[14px]"
      style={{ ["--t" as string]: gameTint(g) }}
    >
      <div
        className="flex justify-between gap-3 text-[11px] uppercase tracking-wide"
        style={{ color: "var(--text-2)" }}
      >
        <span>{when}</span>
        <span className="truncate">{meta(g)}</span>
      </div>
      <div className="font-display my-[10px] flex items-center justify-between text-[34px] font-bold leading-none">
        <span style={{ opacity: fav && fav !== g.away_team ? 0.7 : 1 }}>{g.away_team}</span>
        <span className="text-[12px] font-medium" style={{ color: "var(--text-3)" }}>
          @
        </span>
        <span style={{ opacity: fav && fav !== g.home_team ? 0.7 : 1 }}>{g.home_team}</span>
      </div>
      {g.target === "winner" && prob != null && (
        <div className="pbar" aria-hidden>
          <i style={{ width: `${prob}%` }} />
        </div>
      )}
      <div className="mt-[9px] flex items-baseline justify-between gap-3">
        <span className="font-display t-ink text-[22px] font-bold tabular-nums">
          {pickLabel(g)}
        </span>
        <ConfidenceTag level={g.confidence_level} />
      </div>
      {g.explanation_es && (
        <p className="mt-2 text-[12.5px] leading-snug" style={{ color: "var(--text-2)" }}>
          {g.explanation_es}
        </p>
      )}
    </Link>
  );
}
