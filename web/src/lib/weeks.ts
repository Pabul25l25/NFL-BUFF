interface Weekly {
  season: number;
  week: number;
  kickoff_utc: string;
}

export interface WeekSelection<T> {
  rows: T[];
  season: number | null;
  week: number | null;
  /** true cuando no hay partidos futuros y se muestra la semana mas reciente. */
  latest: boolean;
}

/**
 * Elige UNA semana: la indicada, o la del proximo partido, o (sin futuros) la mas reciente.
 * Con onlyFuture, en la semana proxima descarta partidos ya jugados (no mezcla pasado y futuro).
 */
export function selectWeek<T extends Weekly>(
  data: T[] | undefined,
  now: number,
  opts: { week?: number; onlyFuture?: boolean; forceLatest?: boolean } = {}
): WeekSelection<T> {
  if (!data || data.length === 0) return { rows: [], season: null, week: null, latest: false };
  const ts = (r: T) => new Date(r.kickoff_utc).getTime();
  let season: number;
  let week: number;
  let latest = false;
  if (opts.week) {
    const pool = data.filter((r) => r.week === opts.week);
    if (pool.length === 0) return { rows: [], season: null, week: opts.week, latest: false };
    season = Math.max(...pool.map((r) => r.season));
    week = opts.week;
  } else {
    const future = opts.forceLatest ? [] : data.filter((r) => ts(r) >= now);
    if (future.length > 0) {
      const next = future.reduce((a, b) => (ts(b) < ts(a) ? b : a));
      season = next.season;
      week = next.week;
    } else {
      const last = data.reduce((a, b) => (ts(b) > ts(a) ? b : a));
      season = last.season;
      week = last.week;
      latest = true;
    }
  }
  let rows = data.filter((r) => r.season === season && r.week === week);
  if (opts.onlyFuture && !latest && !opts.week) rows = rows.filter((r) => ts(r) >= now);
  return { rows: [...rows].sort((a, b) => ts(a) - ts(b)), season, week, latest };
}

export interface WeekGroup<T> {
  season: number;
  week: number;
  rows: T[];
}

/** Agrupa por temporada y semana: mas recientes primero; dentro de cada semana, por kickoff. */
export function groupByWeek<T extends Weekly>(data: T[] | undefined): WeekGroup<T>[] {
  const map = new Map<string, WeekGroup<T>>();
  for (const r of data ?? []) {
    const k = `${r.season}-${r.week}`;
    const g = map.get(k) ?? { season: r.season, week: r.week, rows: [] };
    g.rows.push(r);
    map.set(k, g);
  }
  const t = (r: T) => new Date(r.kickoff_utc).getTime();
  return [...map.values()]
    .map((g) => ({ ...g, rows: [...g.rows].sort((a, b) => t(a) - t(b)) }))
    .sort((a, b) => b.season - a.season || b.week - a.week);
}

/**
 * Solo proyecciones futuras: filas con kickoff >= now, de la semana del proximo partido.
 * Sin partidos futuros devuelve rows vacio (nunca cae a datos pasados).
 */
export function upcomingWeek<T extends Weekly>(
  data: T[] | undefined,
  now: number
): { rows: T[]; season: number | null; week: number | null } {
  const ts = (r: T) => new Date(r.kickoff_utc).getTime();
  const future = (data ?? []).filter((r) => ts(r) >= now);
  if (future.length === 0) return { rows: [], season: null, week: null };
  const next = future.reduce((a, b) => (ts(b) < ts(a) ? b : a));
  const rows = future
    .filter((r) => r.season === next.season && r.week === next.week)
    .sort((a, b) => ts(a) - ts(b));
  return { rows, season: next.season, week: next.week };
}

/** Semanas 1-18 siempre; playoffs (19-22) solo si hay datos. */
export function weekNumbers(data: { week: number }[] | undefined): number[] {
  const max = Math.min(22, Math.max(18, ...(data ?? []).map((r) => r.week)));
  return Array.from({ length: max }, (_, i) => i + 1);
}
