/**
 * "Mis tickets": tickets guardados en localStorage. Todo se valida al leer; los datos malformados se descartan.
 */
import { GAME_ID_RE, MAX_LEGS, decodeTicketLeg, TicketLeg } from "./parlay";

export const SAVED_KEY = "picks:saved:v1";
export const MAX_SAVED = 50;
const MAX_NAME = 80;

export interface SavedTicket {
  id: string;
  name: string;
  /** ISO 8601 */
  created: string;
  /** `<game_id>~<legcode>` */
  legs: string[];
  /** texto de cada seleccion al guardar (misma longitud que legs) */
  legText: string[];
  /** etiqueta de cada partido al guardar */
  games: Record<string, string>;
  /** conjunta del modelo al guardar */
  probability: number;
  fairDecimal: number | null;
  calibration: { ece: number; n: number } | null;
}

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x);
const isProb = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1;

/** Valida un elemento; null si no tiene la forma esperada. */
export function parseSavedTicket(x: unknown): SavedTicket | null {
  if (!isObj(x)) return null;
  const { id, name, created, legs, legText, games, probability, fairDecimal, calibration } = x;
  if (typeof id !== "string" || !id || id.length > 64) return null;
  if (typeof name !== "string") return null;
  if (typeof created !== "string" || Number.isNaN(Date.parse(created))) return null;
  if (!Array.isArray(legs) || legs.length === 0 || legs.length > MAX_LEGS) return null;
  if (!legs.every((l) => typeof l === "string" && decodeTicketLeg(l) !== null)) return null;
  if (!isProb(probability)) return null;
  if (fairDecimal != null && !(typeof fairDecimal === "number" && Number.isFinite(fairDecimal) && fairDecimal > 0)) return null;

  let cal: SavedTicket["calibration"] = null;
  if (calibration != null) {
    if (!isObj(calibration) || typeof calibration.ece !== "number" || typeof calibration.n !== "number") return null;
    if (!Number.isFinite(calibration.ece) || !Number.isFinite(calibration.n)) return null;
    cal = { ece: calibration.ece, n: calibration.n };
  }
  const texts =
    Array.isArray(legText) && legText.length === legs.length && legText.every((t) => typeof t === "string")
      ? (legText as string[])
      : (legs as string[]);
  const gm: Record<string, string> = {};
  if (isObj(games)) for (const [k, v] of Object.entries(games)) if (typeof v === "string") gm[k] = v;
  return {
    id,
    name: name.trim().slice(0, MAX_NAME),
    created,
    legs: legs as string[],
    legText: texts,
    games: gm,
    probability,
    fairDecimal: (fairDecimal as number | null | undefined) ?? null,
    calibration: cal,
  };
}

/** Valida el arreglo completo: descarta elementos invalidos, duplicados de id y el exceso sobre el maximo. */
export function parseSavedTickets(raw: unknown): SavedTicket[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: SavedTicket[] = [];
  for (const x of raw) {
    const t = parseSavedTicket(x);
    if (!t || seen.has(t.id)) continue;
    seen.add(t.id);
    out.push(t);
    if (out.length >= MAX_SAVED) break;
  }
  return out;
}

export function loadSaved(): SavedTicket[] {
  try {
    const raw = window.localStorage.getItem(SAVED_KEY);
    if (!raw) return [];
    return parseSavedTickets(JSON.parse(raw));
  } catch {
    return [];
  }
}

/** true si se pudo escribir. */
export function persistSaved(list: SavedTicket[]): boolean {
  try {
    window.localStorage.setItem(SAVED_KEY, JSON.stringify(list.slice(0, MAX_SAVED)));
    return true;
  } catch {
    return false;
  }
}

export function newId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
  }
}

export function addSaved(list: SavedTicket[], t: SavedTicket): { list: SavedTicket[]; error?: string } {
  if (list.length >= MAX_SAVED) return { list, error: `Máximo ${MAX_SAVED} tickets guardados. Elimina alguno primero.` };
  return { list: [t, ...list] };
}

export function renameSaved(list: SavedTicket[], id: string, name: string): SavedTicket[] {
  const n = name.trim().slice(0, MAX_NAME);
  if (!n) return list;
  return list.map((t) => (t.id === id ? { ...t, name: n } : t));
}

export function deleteSaved(list: SavedTicket[], id: string): SavedTicket[] {
  return list.filter((t) => t.id !== id);
}

/** Selecciones del ticket guardado, ya decodificadas. */
export function savedLegs(t: SavedTicket): TicketLeg[] {
  return t.legs.flatMap((l) => {
    const d = decodeTicketLeg(l);
    return d ? [d] : [];
  });
}

export function savedGameIds(t: SavedTicket): string[] {
  return Array.from(new Set(savedLegs(t).map((l) => l.gameId))).filter((id) => GAME_ID_RE.test(id));
}
