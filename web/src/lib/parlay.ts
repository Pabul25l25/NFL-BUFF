/**
 * Motor de parlays del mismo partido (same-game parlay). TypeScript puro, sin React.
 * Cópula gaussiana + Monte Carlo sobre los marginales publicados por el modelo.
 */

// ---------------------------------------------------------------- contrato

export interface ParlayGame {
  game_id: string;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  kickoff_utc: string;
  p_home_win: number;
  /** margen = puntos local - puntos visitante */
  margin_mu: number;
  margin_sigma: number;
  total_mu: number;
  total_sigma: number;
}

export interface ParlayPlayer {
  gsis_id: string;
  player_name: string;
  position: string | null;
  team: string;
  is_home: boolean;
}

export interface ParlayMarket {
  gsis_id: string;
  target: string;
  point_estimate: number | null;
  /** 19 cuantiles p05..p95 (continuos) */
  quantiles: number[] | null;
  /** tasa de Poisson (conteos) */
  lambda: number | null;
}

export interface ParlayInputs {
  game: ParlayGame;
  players: ParlayPlayer[];
  markets: ParlayMarket[];
  corr: { legs: string[]; matrix: number[][] };
  calibration: { ece: number; n: number } | null;
}

export const CONTINUOUS_TARGETS = ["receiving_yards", "rushing_yards", "receptions", "passing_yards"];
export const COUNT_TARGETS = ["receiving_tds", "rushing_tds", "passing_tds", "sacks_suffered"];

// ---------------------------------------------------------------- numerica

/** PRNG determinista (mulberry32). Devuelve uniformes en [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Muestreador de normales estandar (Box-Muller, usa los dos valores del par). */
export function normalSampler(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0;
    while (u === 0) u = rand();
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

/** CDF normal estandar (Hart/West, error ~1e-15). */
export function normCdf(x: number): number {
  const xa = Math.abs(x);
  let c: number;
  if (xa > 37) {
    c = 0;
  } else {
    const e = Math.exp((-xa * xa) / 2);
    if (xa < 7.07106781186547) {
      let b = 3.52624965998911e-2 * xa + 0.700383064443688;
      b = b * xa + 6.37396220353165;
      b = b * xa + 33.912866078383;
      b = b * xa + 112.079291497871;
      b = b * xa + 221.213596169931;
      b = b * xa + 220.206867912376;
      c = e * b;
      b = 8.83883476483184e-2 * xa + 1.75566716318264;
      b = b * xa + 16.064177579207;
      b = b * xa + 86.7807322029461;
      b = b * xa + 296.564248779674;
      b = b * xa + 637.333633378831;
      b = b * xa + 793.826512519948;
      b = b * xa + 440.413735824752;
      c = c / b;
    } else {
      let b = xa + 0.65;
      b = xa + 4 / b;
      b = xa + 3 / b;
      b = xa + 2 / b;
      b = xa + 1 / b;
      c = e / b / 2.506628274631;
    }
  }
  return x > 0 ? 1 - c : c;
}

/** Cholesky inferior; null si la matriz no es definida positiva. */
export function choleskyRaw(a: number[][]): number[][] | null {
  const n = a.length;
  const l: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j];
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k];
      if (i === j) {
        if (!(s > 1e-10)) return null;
        l[i][i] = Math.sqrt(s);
      } else {
        l[i][j] = s / l[j][j];
      }
    }
  }
  return l;
}

/**
 * Cholesky con reparacion PSD: simetriza, fuerza diagonal 1 y entradas en [-1, 1]; si falla,
 * encoge hacia la identidad ((1-a)C + aI) con a creciente hasta que factoriza.
 * `shrink` > 0 indica que hubo reparacion.
 */
export function choleskyRepair(matrix: number[][]): { L: number[][]; shrink: number } {
  const n = matrix.length;
  const base: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      const v = (matrix[i][j] + matrix[j][i]) / 2;
      return Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0;
    })
  );
  for (const a of [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1]) {
    const m = base.map((row, i) => row.map((v, j) => (i === j ? 1 : (1 - a) * v)));
    const L = choleskyRaw(m);
    if (L) return { L, shrink: a };
  }
  // inalcanzable (la identidad siempre factoriza), por seguridad
  return { L: base.map((_, i) => base.map((_, j) => (i === j ? 1 : 0))), shrink: 1 };
}

/** Inversa de CDF por interpolacion lineal de la malla de 19 cuantiles (p05..p95) en espacio u. */
export function quantileInverse(q: number[], u: number): number {
  const s = [...q].sort((a, b) => a - b);
  const step = 0.05;
  let v: number;
  if (u <= 0.05) {
    v = s[0] - ((s[1] - s[0]) / step) * (0.05 - u);
  } else if (u >= 0.95) {
    v = s[18] + ((s[18] - s[17]) / step) * (u - 0.95);
  } else {
    const pos = (u - 0.05) / step;
    const i = Math.min(17, Math.floor(pos));
    v = s[i] + (s[i + 1] - s[i]) * (pos - i);
  }
  return Math.max(0, v);
}

/** Tabla CDF de Poisson(lambda) hasta cubrir ~1 - 1e-12. */
export function poissonTable(lambda: number): number[] {
  const lam = Math.max(0, lambda);
  const cdf: number[] = [];
  let p = Math.exp(-lam);
  let acc = p;
  cdf.push(acc);
  for (let k = 1; k < 500 && acc < 1 - 1e-12; k++) {
    p *= lam / k;
    acc += p;
    cdf.push(acc);
  }
  return cdf;
}

/** Inversa de CDF de Poisson: menor k con CDF(k) >= u. */
export function poissonInverse(table: number[], u: number): number {
  let lo = 0;
  let hi = table.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (table[mid] >= u) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

export function poissonCdfAt(lambda: number, k: number): number {
  const t = poissonTable(lambda);
  return k < 0 ? 0 : k >= t.length ? 1 : t[k];
}

// ---------------------------------------------------------------- cuotas

export function fairDecimal(p: number): number | null {
  return p > 0 ? 1 / p : null;
}

export function toAmerican(p: number): number | null {
  if (!(p > 0) || p >= 1) return null;
  return p >= 0.5 ? Math.round((-100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p);
}

export function formatAmerican(a: number | null): string {
  if (a == null) return "—";
  return a > 0 ? `+${a}` : `${a}`;
}

/**
 * Cuota del usuario a decimal. Americana si lleva signo o |x| >= 100; decimal (> 1) en otro caso.
 * Devuelve null si no se puede interpretar.
 */
export function parseOdds(text: string): number | null {
  const t = text.trim().replace(",", ".");
  if (!t) return null;
  const x = Number(t);
  if (!Number.isFinite(x) || x === 0) return null;
  if (/^[+-]/.test(t) || Math.abs(x) >= 100) {
    if (Math.abs(x) < 100) return null;
    return x > 0 ? 1 + x / 100 : 1 + 100 / -x;
  }
  return x > 1 ? x : null;
}

// ---------------------------------------------------------------- legs

export type Side = "over" | "under";
export type TeamSide = "home" | "away";

export type Leg =
  | { kind: "player"; gsis_id: string; target: string; side: Side; line: number }
  | { kind: "anytime_td"; gsis_id: string }
  | { kind: "two_td"; gsis_id: string }
  | { kind: "first_td"; gsis_id: string }
  | { kind: "winner"; team: TeamSide }
  /** el equipo cubre `line` (handicap del equipo): cubre si sus puntos + line > rival */
  | { kind: "spread"; team: TeamSide; line: number }
  | { kind: "total"; side: Side; line: number };

export function legKey(l: Leg): string {
  switch (l.kind) {
    case "player":
      return `p:${l.gsis_id}:${l.target}:${l.side}:${l.line}`;
    case "anytime_td":
      return `atd:${l.gsis_id}`;
    case "two_td":
      return `2td:${l.gsis_id}`;
    case "first_td":
      return `ftd:${l.gsis_id}`;
    case "winner":
      return `w:${l.team}`;
    case "spread":
      return `s:${l.team}:${l.line}`;
    case "total":
      return `t:${l.side}:${l.line}`;
  }
}

const isSide = (x: string): x is Side => x === "over" || x === "under";
const isTeam = (x: string): x is TeamSide => x === "home" || x === "away";

export function decodeLeg(s: string): Leg | null {
  const parts = s.split(":");
  const num = (x: string | undefined) => {
    const n = Number(x);
    return x != null && x !== "" && Number.isFinite(n) ? n : null;
  };
  switch (parts[0]) {
    case "p": {
      const line = num(parts[4]);
      if (parts.length !== 5 || !parts[1] || !parts[2] || !isSide(parts[3]) || line == null) return null;
      return { kind: "player", gsis_id: parts[1], target: parts[2], side: parts[3], line };
    }
    case "atd":
    case "2td":
    case "ftd": {
      if (parts.length !== 2 || !parts[1]) return null;
      const kind = parts[0] === "atd" ? "anytime_td" : parts[0] === "2td" ? "two_td" : "first_td";
      return { kind, gsis_id: parts[1] };
    }
    case "w":
      return parts.length === 2 && isTeam(parts[1]) ? { kind: "winner", team: parts[1] } : null;
    case "s": {
      const line = num(parts[2]);
      return parts.length === 3 && isTeam(parts[1]) && line != null
        ? { kind: "spread", team: parts[1], line }
        : null;
    }
    case "t": {
      const line = num(parts[2]);
      return parts.length === 3 && isSide(parts[1]) && line != null ? { kind: "total", side: parts[1], line } : null;
    }
    default:
      return null;
  }
}

/** Parte `<gsis_id>:<target>:over:<line>` (enlace desde Props). target puede ser anytime_td, two_td, first_td. */
export function decodeAdd(s: string): Leg | null {
  const [id, target, side, line] = s.split(":");
  if (!id || !target) return null;
  if (target === "anytime_td") return { kind: "anytime_td", gsis_id: id };
  if (target === "two_td") return { kind: "two_td", gsis_id: id };
  if (target === "first_td") return { kind: "first_td", gsis_id: id };
  const n = Number(line);
  if (!isSide(side ?? "") || line == null || line === "" || !Number.isFinite(n)) return null;
  return { kind: "player", gsis_id: id, target, side: side as Side, line: n };
}

/** Linea por defecto: la proyeccion redondeada al .5 mas cercano. */
export function halfLine(v: number): number {
  return Math.floor(v) + 0.5;
}

// ---------------------------------------------------------------- latentes

type Latent =
  | { kind: "normal"; mu: number; sigma: number }
  | { kind: "quantile"; q: number[] }
  | { kind: "poisson"; table: number[] };

function marketOf(inputs: ParlayInputs, gsis: string, target: string): ParlayMarket | undefined {
  return inputs.markets.find((m) => m.gsis_id === gsis && m.target === target);
}

function usableMarket(m: ParlayMarket | undefined): m is ParlayMarket {
  if (!m) return false;
  if (CONTINUOUS_TARGETS.includes(m.target)) return Array.isArray(m.quantiles) && m.quantiles.length === 19 && m.quantiles.every((x) => typeof x === "number" && Number.isFinite(x));
  return m.lambda != null && Number.isFinite(m.lambda) && m.lambda >= 0;
}

export function hasMarket(inputs: ParlayInputs, gsis: string, target: string): boolean {
  return usableMarket(marketOf(inputs, gsis, target));
}

const latentKey = (gsis: string, target: string) => `p:${gsis}:${target}`;

function latentFor(inputs: ParlayInputs, key: string): Latent | null {
  if (key === "g:margin") return { kind: "normal", mu: inputs.game.margin_mu, sigma: inputs.game.margin_sigma };
  if (key === "g:total") return { kind: "normal", mu: inputs.game.total_mu, sigma: inputs.game.total_sigma };
  const [, gsis, target] = key.split(":");
  const m = marketOf(inputs, gsis, target);
  if (!usableMarket(m)) return null;
  if (CONTINUOUS_TARGETS.includes(target)) return { kind: "quantile", q: m.quantiles as number[] };
  return { kind: "poisson", table: poissonTable(m.lambda as number) };
}

/** Claves latentes que necesita una leg (vacio si es de primer TD o no hay datos). */
function legLatents(inputs: ParlayInputs, leg: Leg): string[] | null {
  switch (leg.kind) {
    case "player": {
      return hasMarket(inputs, leg.gsis_id, leg.target) ? [latentKey(leg.gsis_id, leg.target)] : null;
    }
    case "anytime_td":
    case "two_td": {
      const ks = ["receiving_tds", "rushing_tds"]
        .filter((t) => hasMarket(inputs, leg.gsis_id, t))
        .map((t) => latentKey(leg.gsis_id, t));
      return ks.length > 0 ? ks : null;
    }
    case "first_td":
      return [];
    case "winner":
    case "spread":
      return ["g:margin"];
    case "total":
      return ["g:total"];
  }
}

/** Tasa de TD (recepcion + carrera) del jugador; null si no hay ninguna. */
export function tdRate(inputs: ParlayInputs, gsis: string): number | null {
  let s = 0;
  let any = false;
  for (const t of ["receiving_tds", "rushing_tds"]) {
    const m = marketOf(inputs, gsis, t);
    if (usableMarket(m)) {
      s += m.lambda as number;
      any = true;
    }
  }
  return any ? s : null;
}

/** P(el partido tiene al menos un TD): ~75 % de los puntos vienen de TD a ~7 pts cada uno. */
export function gameTdProb(inputs: ParlayInputs): number {
  const lamAll = inputs.players.reduce((s, p) => s + (tdRate(inputs, p.gsis_id) ?? 0), 0);
  const expected = Number.isFinite(inputs.game.total_mu) && inputs.game.total_mu > 0 ? (inputs.game.total_mu * 0.75) / 7 : lamAll;
  return 1 - Math.exp(-expected);
}

/** Primer TD: lambda_i / suma(lambda) x P(>=1 TD). Analitico. */
export function firstTdProb(inputs: ParlayInputs, gsis: string): number | null {
  const mine = tdRate(inputs, gsis);
  if (mine == null) return null;
  const total = inputs.players.reduce((s, p) => s + (tdRate(inputs, p.gsis_id) ?? 0), 0);
  if (!(total > 0)) return null;
  return (mine / total) * gameTdProb(inputs);
}

/** Probabilidad analitica de anotar TD (sin correlacion entre recepcion y carrera del mismo jugador). */
export function anytimeTdProb(inputs: ParlayInputs, gsis: string): number | null {
  const r = tdRate(inputs, gsis);
  return r == null ? null : 1 - Math.exp(-r);
}

// ---------------------------------------------------------------- evaluacion

export interface LegResult {
  leg: Leg;
  available: boolean;
  /** por que no esta disponible */
  reason?: string;
  /** probabilidad marginal */
  p: number;
  /** aproximada (primer TD) */
  approx: boolean;
}

export interface ParlayResult {
  legs: LegResult[];
  /** probabilidad conjunta con correlacion (solo legs disponibles) */
  joint: number;
  /** conjunta asumiendo independencia (producto de marginales) */
  independent: number;
  fairDecimal: number | null;
  fairAmerican: number | null;
  /** hay alguna leg aproximada (primer TD) */
  approx: boolean;
  /** grado de encogimiento aplicado para reparar la matriz (0 = sin reparacion) */
  repairShrink: number;
  sims: number;
  /** EV por unidad apostada con la cuota del usuario; null si no hay cuota */
  ev: number | null;
  bookDecimal: number | null;
}

export interface EvalOptions {
  sims?: number;
  seed?: number;
  /** cuota decimal del libro para el parlay completo */
  bookDecimal?: number | null;
}

type Pred = (i: number) => boolean;

export function evaluateParlay(inputs: ParlayInputs, legs: Leg[], opts: EvalOptions = {}): ParlayResult {
  const sims = opts.sims ?? 20000;
  const seed = opts.seed ?? 20240907;

  // 1) latentes necesarias (en orden estable)
  const perLeg = legs.map((l) => legLatents(inputs, l));
  const keys: string[] = [];
  for (const ks of perLeg) for (const k of ks ?? []) if (!keys.includes(k)) keys.push(k);

  // 2) submatriz de correlacion (pares faltantes = 0)
  const idx = new Map(inputs.corr.legs.map((k, i) => [k, i]));
  const C = keys.map((a, i) =>
    keys.map((b, j) => {
      if (i === j) return 1;
      const ia = idx.get(a);
      const ib = idx.get(b);
      if (ia == null || ib == null) return 0;
      return inputs.corr.matrix[ia]?.[ib] ?? 0;
    })
  );
  const { L, shrink } = choleskyRepair(C);

  // 3) simulacion
  const latents = keys.map((k) => latentFor(inputs, k) as Latent);
  const k = keys.length;
  const values = keys.map(() => new Float64Array(sims));
  const z = new Array<number>(k).fill(0);
  const rand = mulberry32(seed);
  const gauss = normalSampler(rand);
  for (let s = 0; s < sims; s++) {
    for (let j = 0; j < k; j++) z[j] = gauss();
    for (let j = 0; j < k; j++) {
      let x = 0;
      for (let m = 0; m <= j; m++) x += L[j][m] * z[m];
      const lat = latents[j];
      if (lat.kind === "normal") values[j][s] = lat.mu + lat.sigma * x;
      else if (lat.kind === "quantile") values[j][s] = quantileInverse(lat.q, normCdf(x));
      else values[j][s] = poissonInverse(lat.table, normCdf(x));
    }
  }
  const col = (key: string) => values[keys.indexOf(key)];

  // 4) predicados por leg
  const preds: (Pred | null)[] = legs.map((leg, i) => {
    const ks = perLeg[i];
    if (!ks) return null;
    switch (leg.kind) {
      case "player": {
        const v = col(ks[0]);
        return leg.side === "over" ? (s) => v[s] > leg.line : (s) => v[s] < leg.line;
      }
      case "anytime_td":
      case "two_td": {
        const need = leg.kind === "anytime_td" ? 1 : 2;
        const vs = ks.map(col);
        return (s) => {
          let t = 0;
          for (const v of vs) t += v[s];
          return t >= need;
        };
      }
      case "first_td":
        return null;
      case "winner": {
        const v = col("g:margin");
        return leg.team === "home" ? (s) => v[s] > 0 : (s) => v[s] < 0;
      }
      case "spread": {
        const v = col("g:margin");
        return leg.team === "home" ? (s) => v[s] + leg.line > 0 : (s) => -v[s] + leg.line > 0;
      }
      case "total": {
        const v = col("g:total");
        return leg.side === "over" ? (s) => v[s] > leg.line : (s) => v[s] < leg.line;
      }
    }
  });

  // 5) marginales y conjunta
  const results: LegResult[] = legs.map((leg, i) => {
    if (leg.kind === "first_td") {
      const p = firstTdProb(inputs, leg.gsis_id);
      return p == null
        ? { leg, available: false, reason: "Sin proyección de TD para este jugador", p: 0, approx: true }
        : { leg, available: true, p, approx: true };
    }
    const pr = preds[i];
    if (!pr) return { leg, available: false, reason: "Sin datos del modelo para esta selección", p: 0, approx: false };
    let hit = 0;
    for (let s = 0; s < sims; s++) if (pr(s)) hit++;
    return { leg, available: true, p: hit / sims, approx: false };
  });

  const active = results.filter((r) => r.available);
  const mcPreds = results.flatMap((r, i) => (r.available && r.leg.kind !== "first_td" ? [preds[i] as Pred] : []));
  let hits = 0;
  if (mcPreds.length > 0) {
    for (let s = 0; s < sims; s++) {
      let ok = true;
      for (const pr of mcPreds) {
        if (!pr(s)) {
          ok = false;
          break;
        }
      }
      if (ok) hits++;
    }
  }
  const mcJoint = mcPreds.length > 0 ? hits / sims : 1;
  const ftdProduct = active.filter((r) => r.leg.kind === "first_td").reduce((a, r) => a * r.p, 1);
  const joint = active.length === 0 ? 0 : mcJoint * ftdProduct;
  const independent = active.length === 0 ? 0 : active.reduce((a, r) => a * r.p, 1);

  const bookDecimal = opts.bookDecimal ?? null;
  return {
    legs: results,
    joint,
    independent,
    fairDecimal: active.length === 0 ? null : fairDecimal(joint),
    fairAmerican: active.length === 0 ? null : toAmerican(joint),
    approx: results.some((r) => r.available && r.approx),
    repairShrink: shrink,
    sims,
    ev: bookDecimal != null && active.length > 0 ? joint * bookDecimal - 1 : null,
    bookDecimal,
  };
}
