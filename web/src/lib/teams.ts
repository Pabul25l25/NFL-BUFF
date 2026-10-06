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

const BG_LIGHT = "#f3f5f8";
const SURFACE_DARK = "#171c23";
const FALLBACK = "#6b7280";

function lin(c: number) {
  const v = c / 255;
  return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}
function rgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function lum(hex: string) {
  const [r, g, b] = rgb(hex);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}
function contrast(a: string, b: string) {
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}
function mixTo(hex: string, target: number, t: number) {
  const c = rgb(hex).map((v) => Math.round(v + (target - v) * t));
  return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
}
/** Acerca el color a negro (0) o blanco (255) hasta alcanzar `min` de contraste contra `bg`. */
function ensure(hex: string, bg: string, min: number, target: number) {
  let out = hex;
  for (let t = 0; t <= 1 && contrast(out, bg) < min; t += 0.04) out = mixTo(hex, target, t);
  return out;
}

function baseColor(abbr: string | null | undefined): string {
  if (!abbr) return FALLBACK;
  return READABLE_COLORS[abbr] ?? TEAM_COLORS[abbr] ?? FALLBACK;
}

// Mezclas del tinte (globals.css: .tint-hero = --mix + 12%): el peor caso es el hero.
const SURFACE_LIGHT = "#ffffff";
const TINT_LIGHT = 0.23;
const TINT_DARK = 0.32;

function mixHex(a: string, b: string, t: number) {
  const [x, y] = [rgb(a), rgb(b)];
  const c = x.map((v, i) => Math.round(v + (y[i] - v) * t));
  return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
}

function fillDark(c: string) {
  return ensure(c, SURFACE_DARK, 3, 255);
}

/** Un valor por tema sin light-dark(): --tl / --td (globals.css) valen "initial" o
 *  vacio segun el tema, asi que solo uno de los dos var() aporta su respaldo. Funciona
 *  tambien en navegadores sin light-dark() y respeta data-theme. */
function perTheme(light: string, dark: string) {
  return `var(--tl, ${light})var(--td, ${dark})`;
}

/** Relleno (barras, bordes, tinte): una variante por tema. En oscuro se aclara
 *  hasta 3:1 contra la superficie. */
export function teamColor(abbr: string | null | undefined): string {
  const c = baseColor(abbr);
  return perTheme(c, fillDark(c));
}

/** Color de texto con contraste >= 4.5:1 en ambos temas, tanto sobre el fondo
 *  plano como sobre la superficie tintada con el propio color del equipo. */
export function teamInk(abbr: string | null | undefined): string {
  const c = baseColor(abbr);
  const tintL = mixHex(SURFACE_LIGHT, c, TINT_LIGHT);
  const tintD = mixHex(SURFACE_DARK, fillDark(c), TINT_DARK);
  const light = ensure(ensure(c, tintL, 4.5, 0), BG_LIGHT, 4.5, 0);
  const dark = ensure(ensure(c, tintD, 4.5, 255), SURFACE_DARK, 4.5, 255);
  return perTheme(light, dark);
}
