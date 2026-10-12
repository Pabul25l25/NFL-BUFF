/**
 * Analisis de escenarios y optimizacion sobre la misma simulacion del ticket (ver `buildTicketSim`).
 * TypeScript puro, sin React.
 */
import {
  GameSim,
  Leg,
  ParlayInputs,
  TicketSim,
  fairDecimal,
  halfLine,
  hasMarket,
  legKey,
  legLatentKeys,
  simulateGame,
  gameSeed,
  DEFAULT_SEED,
  ticketLegHits,
  ticketLegStat,
  ticketLegKey,
  hitVector as hitVectorFor,
} from "./parlay";

// ---------------------------------------------------------------- bits

const packedCache = new WeakMap<Uint8Array, Uint32Array>();

/** Empaqueta un vector 0/1 en palabras de 32 bits (los bits sobrantes quedan en 0). */
export function packBits(v: Uint8Array): Uint32Array {
  const cached = packedCache.get(v);
  if (cached) return cached;
  const words = Math.ceil(v.length / 32);
  const out = new Uint32Array(words);
  for (let i = 0; i < v.length; i++) if (v[i]) out[i >>> 5] |= 1 << (i & 31);
  packedCache.set(v, out);
  return out;
}

export function popcount32(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/** Numero de simulaciones en que todos los vectores empaquetados valen 1. */
function andCount(vs: Uint32Array[]): number {
  if (vs.length === 0) return 0;
  const n = vs[0].length;
  let c = 0;
  if (vs.length === 1) {
    for (let w = 0; w < n; w++) c += popcount32(vs[0][w]);
  } else if (vs.length === 2) {
    const a = vs[0];
    const b = vs[1];
    for (let w = 0; w < n; w++) c += popcount32(a[w] & b[w]);
  } else {
    for (let w = 0; w < n; w++) {
      let x = vs[0][w];
      for (let j = 1; j < vs.length && x !== 0; j++) x &= vs[j][w];
      c += popcount32(x);
    }
  }
  return c;
}

// ---------------------------------------------------------------- a) escenarios

export function scenarioSummary(ts: TicketSim): { hits: number; sims: number; active: number } {
  let hits = 0;
  for (let s = 0; s < ts.sims; s++) hits += ts.hit[s];
  return { hits, sims: ts.sims, active: ts.nActive };
}

/** Indices (en el ticket) de las selecciones con vector de aciertos. */
export function activeLegIndices(ts: TicketSim): number[] {
  const out: number[] = [];
  for (let i = 0; i < ts.ticket.length; i++) if (ticketLegHits(ts, i)) out.push(i);
  return out;
}

// ---------------------------------------------------------------- b) distribucion k de M

export interface CountDistribution {
  /** numero de selecciones disponibles */
  m: number;
  counts: number[];
  /** probs[k] = P(se cumplen exactamente k de M) */
  probs: number[];
  sims: number;
}

export function legCountDistribution(ts: TicketSim): CountDistribution {
  const idx = activeLegIndices(ts);
  const vs = idx.map((i) => ticketLegHits(ts, i) as Uint8Array);
  const m = vs.length;
  const counts = new Array<number>(m + 1).fill(0);
  for (let s = 0; s < ts.sims; s++) {
    let h = 0;
    for (const v of vs) h += v[s];
    counts[h]++;
  }
  return { m, counts, probs: counts.map((c) => c / ts.sims), sims: ts.sims };
}

// ---------------------------------------------------------------- c) eslabon debil

export interface WeakLinkRow {
  /** indice en el ticket */
  index: number;
  p: number;
  /** P(ticket | sin esta seleccion) */
  pWithout: number;
  /** fraccion de escenarios fallidos en que esta seleccion falla */
  missShare: number;
  /** fraccion de escenarios fallidos en que es la unica que falla */
  soleShare: number;
}

export interface WeakLink {
  joint: number;
  failing: number;
  rows: WeakLinkRow[];
  /** indice (en el ticket) de la seleccion que mas escenarios fallidos explica; null si no hay fallos */
  worst: number | null;
}

export function weakestLink(ts: TicketSim): WeakLink {
  const idx = activeLegIndices(ts);
  const vs = idx.map((i) => ticketLegHits(ts, i) as Uint8Array);
  const n = ts.sims;
  const misses = new Uint8Array(n);
  for (const v of vs) for (let s = 0; s < n; s++) misses[s] += v[s] ? 0 : 1;
  let okCount = 0;
  for (let s = 0; s < n; s++) if (misses[s] === 0) okCount++;
  const failing = n - okCount;
  const rows: WeakLinkRow[] = idx.map((index, k) => {
    const v = vs[k];
    let p = 0;
    let without = 0;
    let missed = 0;
    let sole = 0;
    for (let s = 0; s < n; s++) {
      const h = v[s];
      p += h;
      const others = misses[s] - (h ? 0 : 1);
      if (others === 0) without++;
      if (!h) {
        missed++;
        if (misses[s] === 1) sole++;
      }
    }
    return {
      index,
      p: p / n,
      pWithout: without / n,
      missShare: failing > 0 ? missed / failing : 0,
      soleShare: failing > 0 ? sole / failing : 0,
    };
  });
  let worst: number | null = null;
  if (failing > 0 && rows.length > 1) {
    let best = rows[0];
    for (const r of rows) if (r.missShare > best.missShare || (r.missShare === best.missShare && r.p < best.p)) best = r;
    worst = best.index;
  }
  return { joint: okCount / n, failing, rows, worst };
}

// ---------------------------------------------------------------- d) escenario tipico

export interface TypicalRow {
  index: number;
  /** mediana de la estadistica en las simulaciones donde el ticket se cumple */
  median: number;
}

export interface TypicalScenario {
  /** simulaciones en que el ticket se cumple */
  n: number;
  rows: TypicalRow[];
}

export function typicalScenario(ts: TicketSim): TypicalScenario {
  const hitIdx: number[] = [];
  for (let s = 0; s < ts.sims; s++) if (ts.hit[s]) hitIdx.push(s);
  const n = hitIdx.length;
  const rows: TypicalRow[] = [];
  if (n === 0) return { n, rows };
  for (let i = 0; i < ts.ticket.length; i++) {
    const st = ticketLegStat(ts, i);
    if (!st || !ticketLegHits(ts, i)) continue;
    const vals = new Float64Array(n);
    for (let k = 0; k < n; k++) vals[k] = st[hitIdx[k]];
    vals.sort();
    const mid = n >> 1;
    rows.push({ index: i, median: n % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2 });
  }
  return { n, rows };
}

// ---------------------------------------------------------------- lineas

/** Linea de una leg (null si no tiene). */
export function lineOf(leg: Leg): number | null {
  return "line" in leg ? leg.line : null;
}

export function withLine(leg: Leg, line: number): Leg {
  return "line" in leg ? ({ ...leg, line } as Leg) : leg;
}

/** Paso entre lineas alternativas; null si la leg no tiene linea ajustable. */
export function lineStep(leg: Leg): number | null {
  switch (leg.kind) {
    case "player":
      return leg.target === "receiving_yards" || leg.target === "rushing_yards"
        ? 5
        : leg.target === "passing_yards"
          ? 10
          : 1;
    case "total":
      return 3;
    case "spread":
      return 1;
    default:
      return null;
  }
}

const fixLine = (x: number) => Math.round(x * 100) / 100;

/** Linea del desplazamiento `offset` (en pasos) o null si no es valida (jugadores no admiten lineas negativas). */
export function offsetLine(leg: Leg, offset: number): number | null {
  const base = lineOf(leg);
  const step = lineStep(leg);
  if (base == null || step == null) return null;
  const line = fixLine(base + offset * step);
  if (leg.kind === "player" && line < 0) return null;
  return line;
}

// ---------------------------------------------------------------- probabilidad por partido con vectores sustituidos

/** Conjunta del partido con vectores de aciertos sustituidos (indice local -> vector). */
function gameJoint(gs: GameSim, overrides?: Map<number, Uint8Array>): number {
  const vs: Uint32Array[] = [];
  gs.results.forEach((r, i) => {
    if (!r.available || r.leg.kind === "first_td") return;
    const v = overrides?.get(i) ?? gs.hits[i];
    if (v) vs.push(packBits(v));
  });
  if (gs.nActive === 0) return 0;
  if (vs.length === 0) return gs.ftdProduct;
  return (andCount(vs) / gs.sims) * gs.ftdProduct;
}

/** Producto de conjuntas por partido; `skip` omite un partido. */
function productOthers(ts: TicketSim, skip: number): number {
  let p = 1;
  ts.games.forEach((g, i) => {
    if (i !== skip && g && g.nActive > 0) p *= g.joint;
  });
  return p;
}

// ---------------------------------------------------------------- 4a) lineas alternativas

export interface AltLineRow {
  offset: number;
  line: number;
  leg: Leg;
  /** probabilidad de la seleccion sola */
  marginal: number;
  /** probabilidad del ticket completo con esta linea */
  ticketP: number;
  current: boolean;
}

export const ALT_RADIUS = 3;

export function altLines(ts: TicketSim, index: number, radius = ALT_RADIUS): AltLineRow[] {
  const leg = ts.ticket[index]?.leg;
  const stat = ticketLegStat(ts, index);
  const { gameIdx, localIdx } = ts.legs[index];
  const gs = ts.games[gameIdx];
  if (!leg || !stat || !gs || !ticketLegHits(ts, index)) return [];
  const others = productOthers(ts, gameIdx);
  const rows: AltLineRow[] = [];
  for (let o = -radius; o <= radius; o++) {
    const line = offsetLine(leg, o);
    if (line == null) continue;
    const variant = withLine(leg, line);
    const v = o === 0 ? (gs.hits[localIdx] as Uint8Array) : hitVectorFor(variant, stat);
    rows.push({
      offset: o,
      line,
      leg: variant,
      marginal: andCount([packBits(v)]) / ts.sims,
      ticketP: others * gameJoint(gs, new Map([[localIdx, v]])),
      current: o === 0,
    });
  }
  return rows;
}

// ---------------------------------------------------------------- 4b) cuota objetivo

export interface OptimizeChange {
  index: number;
  leg: Leg;
  from: number;
  to: number;
}

export interface OptimizeResult {
  status: "ok" | "unreachable" | "no-adjustable";
  /** conjunta maxima permitida: 1 / cuota objetivo */
  cap: number;
  startJoint: number;
  joint: number;
  fairDecimal: number | null;
  changes: OptimizeChange[];
  iterations: number;
}

/**
 * Busca el ticket mas probable cuya cuota justa sea >= la cuota objetivo (conjunta <= 1/objetivo),
 * moviendo las lineas de las selecciones un paso a la vez dentro de +-radio pasos. Voraz y acotado.
 */
export function optimizeToOdds(ts: TicketSim, targetDecimal: number, radius = ALT_RADIUS, maxIter = 40): OptimizeResult {
  const cap = 1 / targetDecimal;
  const adjustable: number[] = [];
  ts.ticket.forEach((t, i) => {
    if (lineStep(t.leg) != null && ticketLegHits(ts, i) && ticketLegStat(ts, i)) adjustable.push(i);
  });
  const startJoint = ts.games.reduce((p, g) => (g && g.nActive > 0 ? p * g.joint : p), 1) * (ts.nActive > 0 ? 1 : 0);
  const base: OptimizeResult = {
    status: "no-adjustable", cap, startJoint, joint: startJoint, fairDecimal: fairDecimal(startJoint), changes: [], iterations: 0,
  };
  if (adjustable.length === 0 || ts.nActive === 0) return base;

  // vectores por (seleccion, desplazamiento), perezosos
  const vecCache = new Map<string, Uint8Array | null>();
  const vecAt = (i: number, o: number): Uint8Array | null => {
    const key = `${i}|${o}`;
    if (vecCache.has(key)) return vecCache.get(key) ?? null;
    const leg = ts.ticket[i].leg;
    const line = offsetLine(leg, o);
    let v: Uint8Array | null = null;
    if (o === 0) v = ticketLegHits(ts, i);
    else if (line != null) v = hitVectorFor(withLine(leg, line), ticketLegStat(ts, i) as Float64Array);
    vecCache.set(key, v);
    return v;
  };

  const offsets = new Map<number, number>(adjustable.map((i) => [i, 0]));
  // conjunta de cada partido con los desplazamientos actuales
  const gameJoints = ts.games.map((g, gi) => (g && g.nActive > 0 ? evalGame(gi, offsets) : 1));
  function evalGame(gi: number, offs: Map<number, number>): number {
    const gs = ts.games[gi] as GameSim;
    const ov = new Map<number, Uint8Array>();
    for (const [i, o] of offs) {
      if (ts.legs[i].gameIdx !== gi || o === 0) continue;
      const v = vecAt(i, o);
      if (v) ov.set(ts.legs[i].localIdx, v);
    }
    return gameJoint(gs, ov);
  }
  const total = () => gameJoints.reduce((p, x) => p * x, 1);

  let iterations = 0;
  for (; iterations < maxIter; iterations++) {
    const cur = total();
    let bestUnder: { i: number; o: number; j: number; gj: number } | null = null; // mayor conjunta <= cap
    let lowest: { i: number; o: number; j: number; gj: number } | null = null; // menor conjunta
    for (const i of adjustable) {
      const gi = ts.legs[i].gameIdx;
      const o0 = offsets.get(i) as number;
      for (const o of [o0 - 1, o0 + 1]) {
        if (Math.abs(o) > radius || !vecAt(i, o)) continue;
        const trial = new Map(offsets);
        trial.set(i, o);
        const gj = evalGame(gi, trial);
        let j = gj;
        gameJoints.forEach((x, gk) => {
          if (gk !== gi) j *= x;
        });
        const c = { i, o, j, gj };
        if (!lowest || j < lowest.j) lowest = c;
        if (j <= cap && (!bestUnder || j > bestUnder.j)) bestUnder = c;
      }
    }
    let pick: typeof bestUnder = null;
    if (cur > cap) {
      pick = bestUnder ?? lowest; // fase 1: bajar hasta cap
      if (pick && pick.j >= cur) pick = null;
    } else if (bestUnder && bestUnder.j > cur) {
      pick = bestUnder; // fase 2: acercarse a cap por debajo
    }
    if (!pick) break;
    offsets.set(pick.i, pick.o);
    gameJoints[ts.legs[pick.i].gameIdx] = pick.gj;
  }

  const joint = total();
  const changes: OptimizeChange[] = [];
  for (const [i, o] of offsets) {
    if (o === 0) continue;
    const leg = ts.ticket[i].leg;
    changes.push({ index: i, leg: withLine(leg, offsetLine(leg, o) as number), from: lineOf(leg) as number, to: offsetLine(leg, o) as number });
  }
  return {
    status: joint <= cap + 1e-12 ? "ok" : "unreachable",
    cap,
    startJoint,
    joint,
    fairDecimal: fairDecimal(joint),
    changes,
    iterations,
  };
}

// ---------------------------------------------------------------- 4c) sugerencias correlacionadas

export interface Suggestion {
  gameId: string;
  leg: Leg;
  /** correlacion media (orientada) con las selecciones actuales del partido */
  corr: number;
  marginal: number;
  jointBefore: number;
  jointAfter: number;
}

function orientation(leg: Leg): number {
  switch (leg.kind) {
    case "player":
    case "total":
      return leg.side === "over" ? 1 : -1;
    case "winner":
    case "spread":
      return leg.team === "home" ? 1 : -1;
    default:
      return 1;
  }
}

export function suggestCorrelated(
  inputsByGame: Record<string, ParlayInputs | undefined>,
  ts: TicketSim,
  max = 3
): Suggestion[] {
  const out: Suggestion[] = [];
  const have = new Set(ts.ticket.map(ticketLegKey));
  ts.gameIds.forEach((gameId, gi) => {
    const inputs = inputsByGame[gameId];
    const gs = ts.games[gi];
    if (!inputs || !gs || gs.nActive === 0) return;
    const idx = new Map(inputs.corr.legs.map((k, i) => [k, i]));
    // pares (latente, orientacion) de las selecciones actuales
    const pairs: { key: string; o: number }[] = [];
    const used = new Set<string>();
    gs.legs.forEach((leg, li) => {
      if (!gs.results[li].available) return;
      for (const k of legLatentKeys(inputs, leg) ?? []) {
        pairs.push({ key: k, o: orientation(leg) });
        used.add(k);
      }
    });
    if (pairs.length === 0) return;
    const scored: { key: string; s: number }[] = [];
    for (const key of inputs.corr.legs) {
      if (used.has(key)) continue;
      const ia = idx.get(key);
      if (ia == null) continue;
      let sum = 0;
      for (const p of pairs) {
        const ib = idx.get(p.key);
        if (ib != null) sum += p.o * (inputs.corr.matrix[ia]?.[ib] ?? 0);
      }
      const s = sum / pairs.length;
      if (Math.abs(s) >= 0.05) scored.push({ key, s });
    }
    scored.sort((a, b) => Math.abs(b.s) - Math.abs(a.s));
    const others = productOthers(ts, gi);
    const gameLegs = ts.ticket.filter((t) => t.gameId === gameId).map((t) => t.leg);
    let taken = 0;
    for (const { key, s } of scored) {
      if (taken >= 3) break;
      const cand = candidateLeg(inputs, key, s >= 0 ? 1 : -1);
      if (!cand || have.has(`${gameId}~${legKey(cand)}`)) continue;
      const after = simulateGame(inputs, [...gameLegs, cand], { seed: gameSeed(DEFAULT_SEED, gameId), sims: ts.sims });
      const r = after.results[after.results.length - 1];
      if (!r.available) continue;
      out.push({ gameId, leg: cand, corr: Math.abs(s), marginal: r.p, jointBefore: others * gs.joint, jointAfter: others * after.joint });
      taken++;
    }
  });
  out.sort((a, b) => b.corr - a.corr);
  return out.slice(0, max);
}

function candidateLeg(inputs: ParlayInputs, key: string, orient: number): Leg | null {
  const side = orient >= 0 ? "over" : "under";
  if (key === "g:margin") return { kind: "winner", team: orient >= 0 ? "home" : "away" };
  if (key === "g:total") return { kind: "total", side, line: halfLine(inputs.game.total_mu) };
  const [, gsis, target] = key.split(":");
  if (!gsis || !target || !hasMarket(inputs, gsis, target)) return null;
  const m = inputs.markets.find((x) => x.gsis_id === gsis && x.target === target);
  const mid = m?.point_estimate ?? (m?.quantiles ? m.quantiles[9] : m?.lambda) ?? null;
  if (mid == null) return null;
  return { kind: "player", gsis_id: gsis, target, side, line: halfLine(mid) };
}

// ---------------------------------------------------------------- 5) round robin

export interface RoundRobinCombo {
  /** indices en el ticket */
  indices: number[];
  joint: number;
  fairDecimal: number | null;
}

export interface RoundRobinResult {
  k: number;
  m: number;
  /** C(M, k) */
  total: number;
  /** combinaciones mostradas (las mas probables, hasta `cap`) */
  combos: RoundRobinCombo[];
  /** suma de probabilidades de todas las combinaciones = combinaciones ganadoras esperadas */
  expectedWins: number;
  /** P(al menos una combinacion gana) = P(se cumplen >= k selecciones) */
  pAny: number;
}

export function binom(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return Math.round(r);
}

export const ROUND_ROBIN_CAP = 50;

export function roundRobin(ts: TicketSim, k: number, cap = ROUND_ROBIN_CAP): RoundRobinResult | null {
  const idx = activeLegIndices(ts);
  const m = idx.length;
  if (k < 2 || k > m - 1 || k > 4) return null;
  const total = binom(m, k);
  const packed = idx.map((i) => packBits(ticketLegHits(ts, i) as Uint8Array));
  const gameOf = idx.map((i) => ts.legs[i].gameIdx);
  const all: RoundRobinCombo[] = [];
  const pick: number[] = [];
  const evalCombo = () => {
    // agrupa por partido (independientes entre si): producto de conjuntas por partido
    const groups = new Map<number, Uint32Array[]>();
    for (const j of pick) {
      const arr = groups.get(gameOf[j]);
      if (arr) arr.push(packed[j]);
      else groups.set(gameOf[j], [packed[j]]);
    }
    let p = 1;
    for (const vs of groups.values()) p *= andCount(vs) / ts.sims;
    all.push({ indices: pick.map((j) => idx[j]), joint: p, fairDecimal: fairDecimal(p) });
  };
  const rec = (start: number) => {
    if (pick.length === k) {
      evalCombo();
      return;
    }
    for (let j = start; j < m; j++) {
      pick.push(j);
      rec(j + 1);
      pick.pop();
    }
  };
  rec(0);
  const expectedWins = all.reduce((a, c) => a + c.joint, 0);
  const dist = legCountDistribution(ts);
  let pAny = 0;
  for (let h = k; h <= dist.m; h++) pAny += dist.probs[h];
  all.sort((a, b) => b.joint - a.joint);
  return { k, m, total, combos: all.slice(0, cap), expectedWins, pAny };
}
