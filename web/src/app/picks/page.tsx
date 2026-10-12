"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQueries, useQuery } from "@tanstack/react-query";
import { BottomNav } from "@/components/bottom-nav";
import { Builder } from "@/components/picks/builder";
import { Scorers } from "@/components/picks/scorers";
import { TicketCard } from "@/components/picks/ticket-card";
import { PicksCtx, RoundRobinPanel, ScenarioPanel } from "@/components/picks/analysis";
import { AltLinesPanel, LegChange, SuggestionsPanel, TargetOddsPanel } from "@/components/picks/optimize";
import { ShareBar } from "@/components/picks/share";
import { MyTickets } from "@/components/picks/my-tickets";
import { Teams, describeLeg, gameLabel, pct } from "@/components/picks/labels";
import { Collapsible, GameStatus, fieldStyle, label } from "@/components/picks/ui";
import { GamePrediction } from "@/lib/types";
import { upcomingWeek } from "@/lib/weeks";
import {
  GAME_ID_RE,
  Leg,
  MAX_LEGS,
  ParlayInputs,
  TicketLeg,
  addToTicket,
  buildTicketSim,
  decodeTicket,
  evaluateTicket,
  legKey,
  parseOdds,
  ticketLegKey,
} from "@/lib/parlay";
import { MOCK_ENABLED, mockParlayInputs } from "@/lib/parlay-mock";
import {
  SavedTicket,
  addSaved,
  deleteSaved,
  loadSaved,
  persistSaved,
  renameSaved,
  savedLegs,
} from "@/lib/saved-tickets";

const STORAGE_KEY = "picks:v2";
const LEGACY_STORAGE_KEY = "picks:v1";

interface InputsResult {
  data: ParlayInputs;
  mock: boolean;
}

class NotAvailable extends Error {}

async function fetchGames() {
  const res = await fetch("/api/predictions?target=winner");
  if (!res.ok) throw new Error("No se pudo cargar los partidos");
  const data = await res.json();
  return data.predictions as GamePrediction[];
}

async function fetchInputs(game: GamePrediction | undefined, gameId: string): Promise<InputsResult> {
  const res = await fetch(`/api/parlay-inputs?game_id=${encodeURIComponent(gameId)}`);
  if (res.status === 404) {
    // El mock solo existe en desarrollo y siempre se marca en pantalla.
    if (MOCK_ENABLED && game) {
      return { data: mockParlayInputs({ ...game, p_home_win: game.probability }), mock: true };
    }
    throw new NotAvailable();
  }
  if (!res.ok) throw new Error("No se pudieron cargar los datos de picks");
  return { data: (await res.json()) as ParlayInputs, mock: false };
}

function readInitial(sp: URLSearchParams): { gameId: string; legs: TicketLeg[] } {
  let gameId = sp.get("game_id") ?? "";
  if (!GAME_ID_RE.test(gameId)) gameId = "";
  let encoded: string[] = (sp.get("legs") ?? "").split(",").filter(Boolean);
  if (!gameId && encoded.length === 0) {
    for (const key of [STORAGE_KEY, LEGACY_STORAGE_KEY]) {
      try {
        const raw = window.localStorage.getItem(key);
        if (!raw) continue;
        const v = JSON.parse(raw) as { game_id?: unknown; legs?: unknown };
        if (typeof v.game_id === "string" && GAME_ID_RE.test(v.game_id)) gameId = v.game_id;
        if (Array.isArray(v.legs)) encoded = v.legs.filter((x): x is string => typeof x === "string");
        break;
      } catch {
        // almacenamiento no disponible o JSON invalido: se ignora
      }
    }
  }
  const legs = decodeTicket({ gameId, legs: encoded, add: sp.getAll("add") });
  return { gameId, legs };
}

function PicksInner() {
  const searchParams = useSearchParams();
  const [init] = useState(() => readInitial(searchParams));
  const [browseId, setBrowseId] = useState(init.gameId);
  const [legs, setLegs] = useState<TicketLeg[]>(init.legs);
  const [oddsText, setOddsText] = useState("");
  const [limitMsg, setLimitMsg] = useState("");
  const [now] = useState(() => Date.now());
  const [saved, setSaved] = useState<SavedTicket[]>(() => (typeof window === "undefined" ? [] : loadSaved()));

  const { data: all, isLoading: loadingGames, isError: gamesError } = useQuery({
    queryKey: ["predictions", "winner", undefined],
    queryFn: fetchGames,
  });
  const upcoming = useMemo(() => upcomingWeek(all, now), [all, now]);
  const activeGame = browseId || upcoming.rows[0]?.game_id || "";
  const rowOf = useMemo(() => {
    const m = new Map((all ?? []).map((g) => [g.game_id, g]));
    return (id: string) => m.get(id);
  }, [all]);

  // Un fetch por cada partido del ticket mas el que se esta explorando (en paralelo y cacheados).
  const ids = useMemo(() => {
    const out: string[] = [];
    for (const id of [activeGame, ...legs.map((l) => l.gameId)]) if (id && !out.includes(id)) out.push(id);
    return out;
  }, [activeGame, legs]);

  const statuses = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["parlay-inputs", id],
      queryFn: () => fetchInputs(rowOf(id), id),
      enabled: !MOCK_ENABLED || !loadingGames,
      retry: false,
      staleTime: 60_000,
    })),
    combine: (rs): GameStatus[] =>
      rs.map((r, i) => ({
        id: ids[i],
        inputs: r.data?.data,
        mock: r.data?.mock === true,
        loading: r.isLoading || r.isPending,
        failed: r.isError,
      })),
  });
  const statusOf = (id: string) => statuses.find((s) => s.id === id);
  const browse = statusOf(activeGame);
  const inputs = browse?.inputs;
  const anyMock = statuses.some((s) => s.mock && legs.some((l) => l.gameId === s.id || s.id === activeGame));

  const inputsByGame = useMemo(() => {
    const m: Record<string, ParlayInputs | undefined> = {};
    for (const s of statuses) m[s.id] = s.inputs;
    return m;
  }, [statuses]);

  // URL compartible + localStorage
  useEffect(() => {
    if (!activeGame && legs.length === 0) return;
    const qs = new URLSearchParams();
    if (activeGame) qs.set("game_id", activeGame);
    if (legs.length > 0) qs.set("legs", legs.map(ticketLegKey).join(","));
    try {
      window.history.replaceState(null, "", `${window.location.pathname}?${qs}`);
    } catch {
      // sin acceso a history
    }
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ game_id: activeGame, legs: legs.map(ticketLegKey) }));
    } catch {
      // modo privado / cuota llena
    }
  }, [activeGame, legs]);

  const bookDecimal = parseOdds(oddsText);
  // Una sola simulacion por partido (memoizada en simulateGame) alimenta resultado y analisis.
  const result = useMemo(
    () => (legs.length > 0 ? evaluateTicket(inputsByGame, legs, { bookDecimal }) : null),
    [inputsByGame, legs, bookDecimal]
  );
  const ts = useMemo(() => (legs.length > 0 ? buildTicketSim(inputsByGame, legs) : null), [inputsByGame, legs]);
  const failedGames = ids.filter((id) => legs.some((l) => l.gameId === id) && statusOf(id)?.failed);
  const teamsOf = (id: string): Teams => {
    const g = statusOf(id)?.inputs?.game ?? rowOf(id);
    return g ? { home: g.home_team, away: g.away_team } : { home: "local", away: "visitante" };
  };
  const labelOf = (id: string) => {
    const g = statusOf(id)?.inputs?.game ?? rowOf(id);
    return g ? gameLabel(g) : id;
  };
  const ctx: PicksCtx = {
    inputsOf: (id) => statusOf(id)?.inputs,
    teamsOf,
    labelOf,
    legText: (i) => {
      const t = legs[i];
      if (!t) return "";
      return describeLeg(t.leg, statusOf(t.gameId)?.inputs, teamsOf(t.gameId));
    },
  };

  const addLeg = (leg: Leg, gameId = activeGame) => {
    if (!gameId) return;
    const r = addToTicket(legs, { gameId, leg });
    setLimitMsg(r.error ?? "");
    if (r.ticket !== legs) setLegs(r.ticket);
  };
  const removeLeg = (k: string) => setLegs((cur) => cur.filter((x) => ticketLegKey(x) !== k));
  /** Sustituye selecciones por indice (cambio de linea); quita duplicados que resulten. */
  const replaceLegs = (changes: LegChange[]) => {
    setLegs((cur) => {
      const next = cur.map((t, i) => {
        const c = changes.find((x) => x.index === i);
        return c ? { gameId: t.gameId, leg: c.leg } : t;
      });
      const seen = new Set<string>();
      return next.filter((t) => {
        const k = ticketLegKey(t);
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
    });
  };

  const onSave = (t: SavedTicket): string | undefined => {
    const r = addSaved(saved, t);
    if (r.error) return r.error;
    setSaved(r.list);
    return persistSaved(r.list) ? undefined : "Guardado solo en esta sesión: el navegador no permite almacenamiento.";
  };
  const updateSaved = (next: SavedTicket[]) => {
    setSaved(next);
    persistSaved(next);
  };
  const loadTicket = (t: SavedTicket) => {
    const ls = savedLegs(t).reduce<TicketLeg[]>((acc, l) => addToTicket(acc, l).ticket, []);
    setLegs(ls);
    setLimitMsg("");
    if (ls[0]) setBrowseId(ls[0].gameId);
    window.scrollTo?.({ top: 0, behavior: "smooth" });
  };

  const gameOptions = upcoming.rows.slice();
  const gameInList = gameOptions.some((g) => g.game_id === activeGame);
  const countIn = (id: string) => legs.filter((l) => l.gameId === id).length;
  const browseLegs = new Set(legs.filter((l) => l.gameId === activeGame).map((l) => legKey(l.leg)));
  const calibration = inputs?.calibration ?? statuses.find((s) => s.inputs?.calibration)?.inputs?.calibration ?? null;

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <div className="mb-[14px] flex items-baseline justify-between">
          <h1 className="text-[16px] font-bold">Picks · Ticket</h1>
          <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
            {upcoming.week != null ? `Semana ${upcoming.week} · ${upcoming.season}` : "—"}
          </span>
        </div>

        <p
          className="mb-4 py-2 pl-3 text-[11.5px] leading-relaxed"
          style={{ color: "var(--text-2)", borderLeft: "2px solid var(--accent)" }}
        >
          Estimación del modelo, no es recomendación de apuesta.
          {calibration && (
            <span className="tabular-nums">
              {" "}Calibración: ECE {pct(calibration.ece)} sobre {calibration.n} predicciones.
            </span>
          )}
        </p>

        <div className="flex flex-col gap-1 pb-4">
          <label htmlFor="pk-game" className={label} style={{ color: "var(--text-3)" }}>
            Explorar partido (agrega selecciones al ticket)
          </label>
          <select
            id="pk-game"
            className="min-h-[40px] rounded-lg px-2 text-[14px]"
            style={fieldStyle}
            value={activeGame}
            onChange={(e) => {
              setBrowseId(e.target.value);
              setLimitMsg("");
            }}
            disabled={gameOptions.length === 0 && !activeGame}
          >
            {!gameInList && activeGame && <option value={activeGame}>{labelOf(activeGame)}</option>}
            {gameOptions.map((g) => (
              <option key={g.game_id} value={g.game_id}>
                {gameLabel(g)} ·{" "}
                {new Date(g.kickoff_utc).toLocaleString("es-MX", { weekday: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                {countIn(g.game_id) > 0 ? ` · ${countIn(g.game_id)} en ticket` : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex-1 px-[22px]">
        {loadingGames && <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Cargando partidos…</p>}
        {gamesError && (
          <p className="text-[13px]" style={{ color: "var(--side-b)" }}>No se pudieron cargar los partidos. Intenta de nuevo.</p>
        )}
        {!loadingGames && !gamesError && !activeGame && legs.length === 0 && (
          <p className="text-[13px]" style={{ color: "var(--text-2)" }}>
            Aún no hay partidos próximos; se publican cuando corre el ciclo semanal.
          </p>
        )}
        {activeGame && browse?.loading && <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Cargando datos de picks…</p>}
        {activeGame && browse?.failed && (
          <p className="mb-4 text-[13px]" style={{ color: "var(--text-2)" }}>
            Datos de picks no disponibles para {labelOf(activeGame)}.
          </p>
        )}

        {anyMock && (
          <p className="mb-4 text-[12px] font-semibold" style={{ color: "var(--accent)" }}>
            DATOS DE EJEMPLO (mock de desarrollo): no son proyecciones reales.
          </p>
        )}

        {limitMsg && (
          <p className="mb-3 text-[12px] font-semibold" role="status" style={{ color: "var(--side-b)" }}>{limitMsg}</p>
        )}

        {inputs && <Builder key={activeGame} inputs={inputs} inTicket={browseLegs} onAdd={(l) => addLeg(l)} />}

        <TicketCard
          legs={legs}
          result={result}
          statusOf={statusOf}
          labelOf={labelOf}
          teamsOf={teamsOf}
          failedGames={failedGames}
          oddsText={oddsText}
          onOdds={setOddsText}
          bookDecimal={bookDecimal}
          onRemove={removeLeg}
          onClear={() => setLegs([])}
        />

        {result && ts && result.activeLegs > 0 && (
          <>
            <ShareBar
              legs={legs}
              result={result}
              ctx={ctx}
              activeGame={activeGame}
              calibration={calibration}
              savedCount={saved.length}
              onSave={onSave}
            />
            <Collapsible title="Escenarios" hint="simulaciones, eslabón débil" defaultOpen>
              <ScenarioPanel ts={ts} ctx={ctx} />
            </Collapsible>
            <Collapsible title="Líneas alternativas" hint="cambia una línea">
              <AltLinesPanel ts={ts} ctx={ctx} onReplace={replaceLegs} />
            </Collapsible>
            <Collapsible title="Cuota objetivo" hint="ajusta líneas">
              <TargetOddsPanel ts={ts} ctx={ctx} onReplace={replaceLegs} />
            </Collapsible>
            <Collapsible title="Sugerencias correlacionadas" hint="mismo partido">
              <SuggestionsPanel
                ts={ts}
                ctx={ctx}
                inputsByGame={inputsByGame}
                full={legs.length >= MAX_LEGS}
                onAdd={(gameId, leg) => addLeg(leg, gameId)}
              />
            </Collapsible>
            <Collapsible title="Round robin" hint="combinaciones">
              <RoundRobinPanel ts={ts} ctx={ctx} />
            </Collapsible>
          </>
        )}

        <Collapsible title={`Mis tickets (${saved.length})`} hint="guardados y resultados" defaultOpen={saved.length > 0}>
          <MyTickets
            saved={saved}
            onLoad={loadTicket}
            onRename={(id, name) => updateSaved(renameSaved(saved, id, name))}
            onDelete={(id) => updateSaved(deleteSaved(saved, id))}
          />
        </Collapsible>

        {inputs && <Scorers inputs={inputs} legs={browseLegs} onAdd={(l) => addLeg(l)} />}
      </div>
      <BottomNav active="picks" />
    </div>
  );
}

export default function PicksPage() {
  return (
    <Suspense fallback={<div className="px-[22px] pt-[22px] text-[13px]" style={{ color: "var(--text-2)" }}>Cargando…</div>}>
      <PicksInner />
    </Suspense>
  );
}
