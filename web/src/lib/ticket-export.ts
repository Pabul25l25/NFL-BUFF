/** Exportar un ticket: texto plano y enlace. Funciones puras. */
import { formatAmerican } from "./parlay";

export interface ExportLeg {
  game: string;
  text: string;
  /** probabilidad marginal; null si no esta disponible */
  p: number | null;
}

export interface ExportData {
  name?: string;
  legs: ExportLeg[];
  joint: number;
  independent: number;
  fairDecimal: number | null;
  fairAmerican: number | null;
  calibration: { ece: number; n: number } | null;
  url?: string;
}

const pct = (p: number, d = 1) => `${(p * 100).toFixed(d)} %`;

export function formatTicketText(d: ExportData): string {
  const lines: string[] = [];
  lines.push(d.name ? `Ticket: ${d.name}` : "Ticket");
  const games: string[] = [];
  for (const l of d.legs) if (!games.includes(l.game)) games.push(l.game);
  for (const g of games) {
    lines.push("", g);
    for (const l of d.legs.filter((x) => x.game === g)) {
      lines.push(`  - ${l.text}: ${l.p == null ? "sin datos" : pct(l.p)}`);
    }
  }
  lines.push("", `Probabilidad conjunta (con correlación): ${pct(d.joint)}`);
  lines.push(`Si todo fuera independiente: ${pct(d.independent)}`);
  lines.push(
    `Cuota justa: ${d.fairDecimal ? d.fairDecimal.toFixed(2) : "—"} (${formatAmerican(d.fairAmerican)})`
  );
  if (d.calibration) lines.push(`Calibración del modelo: ECE ${pct(d.calibration.ece)} sobre ${d.calibration.n} predicciones`);
  if (d.url) lines.push("", d.url);
  lines.push("", "Estimación del modelo, no es recomendación de apuesta.");
  return lines.join("\n");
}

export function buildShareUrl(origin: string, pathname: string, gameId: string, legKeys: string[]): string {
  const qs = new URLSearchParams();
  if (gameId) qs.set("game_id", gameId);
  if (legKeys.length > 0) qs.set("legs", legKeys.join(","));
  return `${origin}${pathname}?${qs}`;
}

/** Copia al portapapeles; usa un textarea temporal si la API moderna no esta disponible. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // cae al respaldo
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
