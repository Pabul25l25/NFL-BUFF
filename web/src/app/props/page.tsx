"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BottomNav } from "@/components/bottom-nav";
import { GameGroup, PropGroups, meanOf, playerTeam } from "@/components/prop-table";
import { GamePrediction, STAT_DEFS, PropPrediction, favoriteTeam } from "@/lib/types";
import { upcomingWeek } from "@/lib/weeks";

const UNITS: Record<string, string> = {
  receiving_yards: "yds",
  rushing_yards: "yds",
  passing_yards: "yds",
  receptions: "rec",
  receiving_tds: "TD",
  rushing_tds: "TD",
  passing_tds: "TD",
  sacks_suffered: "caps.",
};

async function fetchProps(target: string) {
  const params = new URLSearchParams({ target });
  const res = await fetch(`/api/props?${params}`);
  if (!res.ok) throw new Error("No se pudo cargar props");
  const data = await res.json();
  return data.props as PropPrediction[];
}

async function fetchWinners() {
  const params = new URLSearchParams({ target: "winner" });
  const res = await fetch(`/api/predictions?${params}`);
  if (!res.ok) throw new Error("No se pudo cargar favoritos");
  const data = await res.json();
  return data.predictions as GamePrediction[];
}

interface SyncRow {
  table_name: string;
  synced_at: string | null;
}

async function fetchStatus() {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error("No se pudo cargar el estado");
  const data = await res.json();
  return data.sync as SyncRow[];
}

/** Fecha de la ultima publicacion: la de la tabla de props si existe, o la mas reciente. */
function lastPublished(sync: SyncRow[] | undefined): Date | null {
  const rows = (sync ?? []).filter((r) => r.synced_at);
  const props = rows.filter((r) => /prop/i.test(r.table_name));
  const pool = props.length > 0 ? props : rows;
  if (pool.length === 0) return null;
  return new Date(Math.max(...pool.map((r) => new Date(r.synced_at as string).getTime())));
}

/** Filtra y ordena las proyecciones; agrupa por partido en orden de kickoff. */
function buildGroups(
  rows: PropPrediction[],
  favorites: Map<string, string | null>,
  f: { pos: string; team: string; game: string }
): { groups: GameGroup[]; total: number } {
  const ts = (r: PropPrediction) => new Date(r.kickoff_utc).getTime();
  const filtered = rows
    .filter(
      (p) =>
        (!f.pos || p.position === f.pos) &&
        (!f.team || playerTeam(p) === f.team) &&
        (!f.game || p.game_id === f.game)
    )
    .sort((a, b) => ts(a) - ts(b) || a.game_id.localeCompare(b.game_id) || (meanOf(b) ?? 0) - (meanOf(a) ?? 0));
  const map = new Map<string, GameGroup>();
  for (const p of filtered) {
    let g = map.get(p.game_id);
    if (!g) {
      g = {
        game_id: p.game_id,
        home_team: p.home_team,
        away_team: p.away_team,
        kickoff_utc: p.kickoff_utc,
        favorite: favorites.get(p.game_id) ?? null,
        rows: [],
      };
      map.set(p.game_id, g);
    }
    g.rows.push(p);
  }
  return { groups: [...map.values()], total: filtered.length };
}

const selectStyle = {
  background: "var(--surface)",
  color: "var(--text)",
  border: "1px solid var(--hair)",
};

export default function PropsPage() {
  const [stat, setStat] = useState("receiving_yards");
  const [pos, setPos] = useState("");
  const [team, setTeam] = useState("");
  const [game, setGame] = useState("");
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const [now] = useState(() => Date.now());

  const { data, isLoading, isError } = useQuery({
    queryKey: ["props", stat],
    queryFn: () => fetchProps(stat),
  });
  const { data: winners } = useQuery({
    queryKey: ["predictions", "winner", undefined],
    queryFn: fetchWinners,
  });
  const { data: sync } = useQuery({ queryKey: ["status"], queryFn: fetchStatus });

  // Solo partidos futuros de la proxima semana; nunca datos pasados.
  const sel = useMemo(() => upcomingWeek(data, now), [data, now]);

  const favorites = useMemo(() => {
    const m = new Map<string, string | null>();
    for (const w of winners ?? []) m.set(w.game_id, favoriteTeam(w));
    return m;
  }, [winners]);

  const positions = useMemo(
    () => [...new Set(sel.rows.map((p) => p.position).filter((x): x is string => !!x))].sort(),
    [sel.rows]
  );
  const teams = useMemo(() => [...new Set(sel.rows.map(playerTeam).filter((x): x is string => !!x))].sort(), [sel.rows]);
  const games = useMemo(() => {
    const m = new Map<string, string>();
    for (const p of sel.rows) m.set(p.game_id, `${p.away_team} @ ${p.home_team}`);
    return [...m.entries()];
  }, [sel.rows]);

  const { groups, total } = useMemo(
    () => buildGroups(sel.rows, favorites, { pos, team, game }),
    [sel.rows, favorites, pos, team, game]
  );
  const allOpen = groups.length > 0 && groups.every((g) => open.has(g.game_id));
  const toggleGame = (id: string) =>
    setOpen((s) => {
      const n = new Set(s);
      if (!n.delete(id)) n.add(id);
      return n;
    });
  const toggleAll = () => setOpen(allOpen ? new Set() : new Set(groups.map((g) => g.game_id)));

  const statDef = STAT_DEFS.find((s) => s.key === stat)!;
  const published = lastPublished(sync);
  const noUpcoming = !isLoading && !isError && sel.rows.length === 0;

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <div className="mb-[14px] flex items-baseline justify-between">
          <span className="text-[16px] font-bold">Proyecciones de jugadores</span>
          <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
            {sel.week != null ? `Próxima semana · Semana ${sel.week} · ${sel.season}` : "—"}
          </span>
        </div>
        <div className="flex flex-wrap gap-[6px] pb-3">
          {STAT_DEFS.map((s) => {
            const active = s.key === stat;
            return (
              <button
                key={s.key}
                onClick={() => { setStat(s.key); setPos(""); setTeam(""); setGame(""); }}
                aria-pressed={active}
                className="rounded-full px-3 py-[6px] text-[12.5px] font-medium"
                style={{
                  background: active ? "var(--text)" : "var(--surface)",
                  color: active ? "var(--bg)" : "var(--text-2)",
                  border: `1px solid ${active ? "var(--text)" : "var(--hair)"}`,
                }}
              >
                {s.label}
              </button>
            );
          })}
        </div>
        {sel.rows.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 pb-3 text-[12px]" style={{ color: "var(--text-2)" }}>
            <label htmlFor="f-game">Partido</label>
            <select
              id="f-game"
              className="rounded-lg px-2 py-[6px]"
              style={selectStyle}
              value={game}
              onChange={(e) => {
                const id = e.target.value;
                setGame(id);
                if (id) setOpen((s) => new Set(s).add(id));
              }}
            >
              <option value="">Todos</option>
              {games.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
            </select>
            <label htmlFor="f-pos">Posición</label>
            <select
              id="f-pos"
              className="rounded-lg px-2 py-[6px]"
              style={selectStyle}
              value={pos}
              onChange={(e) => { setPos(e.target.value); }}
            >
              <option value="">Todas</option>
              {positions.map((x) => <option key={x} value={x}>{x}</option>)}
            </select>
            <label htmlFor="f-team">Equipo</label>
            <select
              id="f-team"
              className="rounded-lg px-2 py-[6px]"
              style={selectStyle}
              value={team}
              onChange={(e) => { setTeam(e.target.value); }}
            >
              <option value="">Todos</option>
              {teams.map((x) => <option key={x} value={x}>{x}</option>)}
            </select>
            <span className="ml-auto tabular-nums">
              {total} jugadores · {groups.length} partidos
            </span>
          </div>
        )}
      </div>

      <div className="flex-1 px-[22px]">
        {statDef.sparse && sel.rows.length > 0 && (
          <div
            className="mb-4 py-2 pl-3 text-[11.5px] leading-relaxed"
            style={{ color: "var(--accent)", borderLeft: "2px solid var(--accent)" }}
          >
            Conteo poco frecuente — la mayoría de partidos son 0. La proyección es una aproximación más basta que en yardas continuas.
          </div>
        )}

        {isLoading && (
          <p className="pt-2 text-[13px]" style={{ color: "var(--text-2)" }}>Cargando…</p>
        )}
        {isError && (
          <p className="pt-2 text-[13px]" style={{ color: "var(--side-b)" }}>
            No se pudieron cargar las proyecciones. Intenta de nuevo.
          </p>
        )}
        {noUpcoming && (
          <div className="pt-2 text-[13px] leading-relaxed" style={{ color: "var(--text-2)" }}>
            <p>Aún no hay proyecciones para la próxima semana; se publican cuando corre el ciclo semanal.</p>
            {published && (
              <p className="mt-2 text-[12px] tabular-nums" style={{ color: "var(--text-3)" }}>
                Última publicación: {published.toLocaleString("es-MX", { dateStyle: "medium", timeStyle: "short" })}
              </p>
            )}
          </div>
        )}
        {sel.rows.length > 0 && total === 0 && (
          <p className="pt-2 text-[13px]" style={{ color: "var(--text-2)" }}>
            No hay proyecciones para este filtro.
          </p>
        )}

        {groups.length > 0 && (
          <>
            <div className="mb-2 flex justify-end">
              <button
                type="button"
                onClick={toggleAll}
                className="text-[12px] font-medium underline underline-offset-2"
                style={{ color: "var(--text-2)" }}
              >
                {allOpen ? "Contraer todo" : "Expandir todo"}
              </button>
            </div>
            <PropGroups groups={groups} unit={UNITS[stat] ?? ""} open={open} onToggle={toggleGame} />
            <p className="mt-2 text-[11px]" style={{ color: "var(--text-3)" }}>
              La barra muestra el rango del 80 % y la marca la proyección. Toca una fila con + para ver la explicación.
            </p>
          </>
        )}
      </div>
      <BottomNav active="props" />
    </div>
  );
}
