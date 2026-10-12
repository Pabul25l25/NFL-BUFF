"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQueries, useQuery } from "@tanstack/react-query";
import { BottomNav } from "@/components/bottom-nav";
import { GamePrediction } from "@/lib/types";
import { upcomingWeek } from "@/lib/weeks";
import {
  Leg,
  MAX_GAMES,
  MAX_LEGS,
  ParlayInputs,
  Side,
  TeamSide,
  TicketLeg,
  TicketResult,
  addToTicket,
  anytimeTdProb,
  decodeTicket,
  evaluateParlay,
  evaluateTicket,
  firstTdProb,
  formatAmerican,
  halfLine,
  hasMarket,
  legKey,
  parseOdds,
  tdRate,
  ticketLegKey,
} from "@/lib/parlay";
import { MOCK_ENABLED, mockParlayInputs } from "@/lib/parlay-mock";

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

// ------------------------------------------------------------ etiquetas

const MARKETS: { key: string; label: string }[] = [
  { key: "receiving_yards", label: "Yds recepción" },
  { key: "rushing_yards", label: "Yds carrera" },
  { key: "receptions", label: "Recepciones" },
  { key: "passing_yards", label: "Yds pase" },
  { key: "passing_tds", label: "TD pase" },
  { key: "sacks_suffered", label: "Capturas" },
  { key: "anytime_td", label: "Anota TD" },
  { key: "two_td", label: "2+ TD" },
  { key: "first_td", label: "Primer TD" },
];
const marketLabel = (k: string) => MARKETS.find((m) => m.key === k)?.label ?? k;

function playerName(inputs: ParlayInputs, id: string) {
  return inputs.players.find((p) => p.gsis_id === id)?.player_name ?? id;
}

const sideText = (s: Side) => (s === "over" ? "Más de" : "Menos de");
const fmtLine = (n: number) => (n > 0 ? `+${n}` : `${n}`);

interface Teams {
  home: string;
  away: string;
}

function describeLeg(leg: Leg, inputs: ParlayInputs | undefined, g: Teams): string {
  const name = (id: string) => (inputs ? playerName(inputs, id) : id);
  switch (leg.kind) {
    case "player":
      return `${name(leg.gsis_id)} · ${marketLabel(leg.target)} · ${sideText(leg.side)} ${leg.line}`;
    case "anytime_td":
      return `${name(leg.gsis_id)} · Anota TD`;
    case "two_td":
      return `${name(leg.gsis_id)} · 2+ TD`;
    case "first_td":
      return `${name(leg.gsis_id)} · Primer TD`;
    case "winner":
      return `Gana ${leg.team === "home" ? g.home : g.away}`;
    case "spread":
      return `Spread · ${leg.team === "home" ? g.home : g.away} ${fmtLine(leg.line)}`;
    case "total":
      return `Total · ${sideText(leg.side)} ${leg.line}`;
  }
}

/** Mercados disponibles para un jugador (segun los datos del modelo). */
function marketsFor(inputs: ParlayInputs, gsis: string): string[] {
  return MARKETS.filter((m) => {
    if (m.key === "anytime_td" || m.key === "two_td" || m.key === "first_td") return tdRate(inputs, gsis) != null;
    return hasMarket(inputs, gsis, m.key);
  }).map((m) => m.key);
}

function defaultPlayerLine(inputs: ParlayInputs, gsis: string, target: string): number {
  const m = inputs.markets.find((x) => x.gsis_id === gsis && x.target === target);
  const v = m?.point_estimate ?? m?.lambda ?? 0;
  return halfLine(v);
}

const pct = (p: number, d = 1) => `${(p * 100).toFixed(d)} %`;

// ------------------------------------------------------------ estilos

const fieldStyle = { background: "var(--surface)", color: "var(--text)", border: "1px solid var(--hair)" };
const btn = "min-h-[40px] rounded-lg px-3 py-2 text-[13px] font-semibold";
const label = "text-[11px] font-semibold uppercase tracking-wider";

function readInitial(sp: URLSearchParams): { gameId: string; legs: TicketLeg[] } {
  let gameId = sp.get("game_id") ?? "";
  let encoded: string[] = (sp.get("legs") ?? "").split(",").filter(Boolean);
  if (!gameId && encoded.length === 0) {
    for (const key of [STORAGE_KEY, LEGACY_STORAGE_KEY]) {
      try {
        const raw = window.localStorage.getItem(key);
        if (!raw) continue;
        const v = JSON.parse(raw) as { game_id?: unknown; legs?: unknown };
        if (typeof v.game_id === "string") gameId = v.game_id;
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

// ------------------------------------------------------------ pagina

interface GameStatus {
  id: string;
  inputs?: ParlayInputs;
  mock: boolean;
  loading: boolean;
  failed: boolean;
}

const gameLabel = (g: { away_team: string; home_team: string }) => `${g.away_team} @ ${g.home_team}`;

function PicksInner() {
  const searchParams = useSearchParams();
  const [init] = useState(() => readInitial(searchParams));
  const [browseId, setBrowseId] = useState(init.gameId);
  const [legs, setLegs] = useState<TicketLeg[]>(init.legs);
  const [oddsText, setOddsText] = useState("");
  const [limitMsg, setLimitMsg] = useState("");
  const [now] = useState(() => Date.now());

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
  const result = useMemo(
    () => (legs.length > 0 ? evaluateTicket(inputsByGame, legs, { bookDecimal }) : null),
    [inputsByGame, legs, bookDecimal]
  );
  const failedGames = ids.filter((id) => legs.some((l) => l.gameId === id) && statusOf(id)?.failed);
  const teamsOf = (id: string): Teams => {
    const g = statusOf(id)?.inputs?.game ?? rowOf(id);
    return g ? { home: g.home_team, away: g.away_team } : { home: "local", away: "visitante" };
  };
  const labelOf = (id: string) => {
    const g = statusOf(id)?.inputs?.game ?? rowOf(id);
    return g ? gameLabel(g) : id;
  };

  const addLeg = (leg: Leg, gameId = activeGame) => {
    if (!gameId) return;
    const r = addToTicket(legs, { gameId, leg });
    setLimitMsg(r.error ?? "");
    if (r.ticket !== legs) setLegs(r.ticket);
  };
  const removeLeg = (k: string) => setLegs((cur) => cur.filter((x) => ticketLegKey(x) !== k));

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

        {inputs && <Scorers inputs={inputs} legs={browseLegs} onAdd={(l) => addLeg(l)} />}
      </div>
      <BottomNav active="picks" />
    </div>
  );
}

// ------------------------------------------------------------ constructor

function Builder({ inputs, inTicket, onAdd }: { inputs: ParlayInputs; inTicket: Set<string>; onAdd: (l: Leg) => void }) {
  const g = inputs.game;
  const quick: { leg: Leg; text: string }[] = [
    { leg: { kind: "winner", team: "away" }, text: `Gana ${g.away_team}` },
    { leg: { kind: "winner", team: "home" }, text: `Gana ${g.home_team}` },
    { leg: { kind: "spread", team: "away", line: halfLine(g.margin_mu) }, text: `${g.away_team} ${fmtLine(halfLine(g.margin_mu))}` },
    { leg: { kind: "spread", team: "home", line: halfLine(-g.margin_mu) }, text: `${g.home_team} ${fmtLine(halfLine(-g.margin_mu))}` },
    { leg: { kind: "total", side: "over", line: halfLine(g.total_mu) }, text: `Más de ${halfLine(g.total_mu)}` },
    { leg: { kind: "total", side: "under", line: halfLine(g.total_mu) }, text: `Menos de ${halfLine(g.total_mu)}` },
  ];
  const [mode, setMode] = useState<"player" | "game">("player");
  const [playerSel, setPlayerSel] = useState("");
  const [marketSel, setMarketSel] = useState("");
  const [side, setSide] = useState<Side>("over");
  const [edit, setEdit] = useState<{ key: string; value: string } | null>(null);

  const [gKind, setGKind] = useState<"winner" | "spread" | "total">("winner");
  const [gTeam, setGTeam] = useState<TeamSide>("home");
  const [gSide, setGSide] = useState<Side>("over");
  const [gEdit, setGEdit] = useState<{ key: string; value: string } | null>(null);

  const withMarkets = inputs.players.filter((p) => marketsFor(inputs, p.gsis_id).length > 0);
  const teams = [
    { name: g.away_team, rows: withMarkets.filter((p) => !p.is_home) },
    { name: g.home_team, rows: withMarkets.filter((p) => p.is_home) },
  ].filter((t) => t.rows.length > 0);

  const playerId = withMarkets.some((p) => p.gsis_id === playerSel) ? playerSel : (withMarkets[0]?.gsis_id ?? "");
  const available = playerId ? marketsFor(inputs, playerId) : [];
  const market = available.includes(marketSel) ? marketSel : (available[0] ?? "");
  const isTd = market === "anytime_td" || market === "two_td" || market === "first_td";

  const lineKey = `${playerId}|${market}`;
  const defLine = playerId && market && !isTd ? defaultPlayerLine(inputs, playerId, market) : 0;
  const lineText = edit?.key === lineKey ? edit.value : String(defLine);
  const lineNum = Number(lineText);
  const lineOk = lineText.trim() !== "" && Number.isFinite(lineNum);

  const gDefault =
    gKind === "spread"
      ? halfLine(gTeam === "home" ? -g.margin_mu : g.margin_mu)
      : gKind === "total"
        ? halfLine(g.total_mu)
        : 0;
  const gKey = `${gKind}|${gTeam}`;
  const gLineText = gEdit?.key === gKey ? gEdit.value : String(gDefault);
  const gLineNum = Number(gLineText);
  const gLineOk = gLineText.trim() !== "" && Number.isFinite(gLineNum);

  let candidate: Leg | null = null;
  if (mode === "player" && playerId && market) {
    if (market === "anytime_td") candidate = { kind: "anytime_td", gsis_id: playerId };
    else if (market === "two_td") candidate = { kind: "two_td", gsis_id: playerId };
    else if (market === "first_td") candidate = { kind: "first_td", gsis_id: playerId };
    else if (lineOk) candidate = { kind: "player", gsis_id: playerId, target: market, side, line: lineNum };
  } else if (mode === "game") {
    if (gKind === "winner") candidate = { kind: "winner", team: gTeam };
    else if (gKind === "spread" && gLineOk) candidate = { kind: "spread", team: gTeam, line: gLineNum };
    else if (gKind === "total" && gLineOk) candidate = { kind: "total", side: gSide, line: gLineNum };
  }
  const candP = candidate ? evaluateParlay(inputs, [candidate], { sims: 6000 }).legs[0] : null;

  const seg = (active: boolean) => ({
    background: active ? "var(--text)" : "var(--surface)",
    color: active ? "var(--bg)" : "var(--text-2)",
    border: `1px solid ${active ? "var(--text)" : "var(--hair)"}`,
  });

  return (
    <section className="mb-6" aria-labelledby="pk-builder">
      <h2 id="pk-builder" className="mb-2 text-[14px] font-bold">
        {gameLabel(g)} · agregar selección
      </h2>
      <div className="mb-4 grid grid-cols-2 gap-[6px] sm:grid-cols-3" role="group" aria-label={`Selecciones rápidas ${gameLabel(g)}`}>
        {quick.map(({ leg, text }) => {
          const has = inTicket.has(legKey(leg));
          return (
            <button
              key={legKey(leg)}
              type="button"
              className={`${btn} tabular-nums`}
              style={{ background: "var(--surface)", color: has ? "var(--text-3)" : "var(--accent)", border: "1px solid var(--hair)" }}
              disabled={has}
              onClick={() => onAdd(leg)}
              aria-label={`Agregar al ticket: ${gameLabel(g)}, ${text}`}
            >
              {text}
            </button>
          );
        })}
      </div>
      <div className="mb-3 flex gap-[6px]" role="group" aria-label="Tipo de selección">
        <button type="button" className={btn} style={seg(mode === "player")} aria-pressed={mode === "player"} onClick={() => setMode("player")}>
          Jugador
        </button>
        <button type="button" className={btn} style={seg(mode === "game")} aria-pressed={mode === "game"} onClick={() => setMode("game")}>
          Juego
        </button>
      </div>

      {mode === "player" ? (
        withMarkets.length === 0 ? (
          <p className="text-[13px]" style={{ color: "var(--text-2)" }}>No hay mercados de jugadores para este partido.</p>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Field id="pk-player" text="Jugador">
              <select id="pk-player" className="min-h-[40px] w-full rounded-lg px-2 text-[14px]" style={fieldStyle} value={playerId} onChange={(e) => setPlayerSel(e.target.value)}>
                {teams.map((t) => (
                  <optgroup key={t.name} label={t.name}>
                    {t.rows.map((p) => (
                      <option key={p.gsis_id} value={p.gsis_id}>{p.player_name}{p.position ? ` (${p.position})` : ""}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </Field>
            <Field id="pk-market" text="Mercado">
              <select id="pk-market" className="min-h-[40px] w-full rounded-lg px-2 text-[14px]" style={fieldStyle} value={market} onChange={(e) => setMarketSel(e.target.value)}>
                {available.map((k) => <option key={k} value={k}>{marketLabel(k)}</option>)}
              </select>
            </Field>
            {!isTd && (
              <>
                <div className="flex flex-col gap-1">
                  <span className={label} style={{ color: "var(--text-3)" }}>Más / Menos</span>
                  <div className="flex gap-[6px]" role="group" aria-label="Más o menos">
                    {(["over", "under"] as Side[]).map((s) => (
                      <button key={s} type="button" className={`${btn} flex-1`} style={seg(side === s)} aria-pressed={side === s} onClick={() => setSide(s)}>
                        {s === "over" ? "Más" : "Menos"}
                      </button>
                    ))}
                  </div>
                </div>
                <Field id="pk-line" text="Línea">
                  <input
                    id="pk-line"
                    inputMode="decimal"
                    className="min-h-[40px] w-full rounded-lg px-3 text-[14px] tabular-nums"
                    style={fieldStyle}
                    value={lineText}
                    onChange={(e) => setEdit({ key: lineKey, value: e.target.value })}
                  />
                </Field>
              </>
            )}
          </div>
        )
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Field id="pk-gkind" text="Mercado">
            <select id="pk-gkind" className="min-h-[40px] w-full rounded-lg px-2 text-[14px]" style={fieldStyle} value={gKind} onChange={(e) => setGKind(e.target.value as typeof gKind)}>
              <option value="winner">Ganador</option>
              <option value="spread">Spread</option>
              <option value="total">Total</option>
            </select>
          </Field>
          {gKind !== "total" ? (
            <Field id="pk-gteam" text="Equipo">
              <select id="pk-gteam" className="min-h-[40px] w-full rounded-lg px-2 text-[14px]" style={fieldStyle} value={gTeam} onChange={(e) => setGTeam(e.target.value as TeamSide)}>
                <option value="home">{g.home_team} (local)</option>
                <option value="away">{g.away_team} (visitante)</option>
              </select>
            </Field>
          ) : (
            <div className="flex flex-col gap-1">
              <span className={label} style={{ color: "var(--text-3)" }}>Más / Menos</span>
              <div className="flex gap-[6px]" role="group" aria-label="Más o menos">
                {(["over", "under"] as Side[]).map((s) => (
                  <button key={s} type="button" className={`${btn} flex-1`} style={seg(gSide === s)} aria-pressed={gSide === s} onClick={() => setGSide(s)}>
                    {s === "over" ? "Más" : "Menos"}
                  </button>
                ))}
              </div>
            </div>
          )}
          {gKind !== "winner" && (
            <Field id="pk-gline" text={gKind === "spread" ? "Línea (handicap del equipo)" : "Línea"}>
              <input
                id="pk-gline"
                inputMode="decimal"
                className="min-h-[40px] w-full rounded-lg px-3 text-[14px] tabular-nums"
                style={fieldStyle}
                value={gLineText}
                onChange={(e) => setGEdit({ key: gKey, value: e.target.value })}
              />
            </Field>
          )}
        </div>
      )}

      <div className="mt-3 flex items-center justify-between gap-3">
        <span className="text-[13px] tabular-nums" style={{ color: "var(--text-2)" }} aria-live="polite">
          {candidate && candP ? (candP.available ? `Probabilidad: ${pct(candP.p)}${candP.approx ? " (aprox.)" : ""}` : candP.reason) : "—"}
        </span>
        <button
          type="button"
          className={btn}
          style={{ background: "var(--accent)", color: "var(--bg)" }}
          disabled={!candidate || !candP?.available}
          onClick={() => candidate && onAdd(candidate)}
        >
          Agregar al ticket
        </button>
      </div>
    </section>
  );
}

function Field({ id, text, children }: { id: string; text: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className={label} style={{ color: "var(--text-3)" }}>{text}</label>
      {children}
    </div>
  );
}

// ------------------------------------------------------------ ticket

function TicketCard({
  legs,
  result,
  statusOf,
  labelOf,
  teamsOf,
  failedGames,
  oddsText,
  onOdds,
  bookDecimal,
  onRemove,
  onClear,
}: {
  legs: TicketLeg[];
  result: TicketResult | null;
  statusOf: (id: string) => GameStatus | undefined;
  labelOf: (id: string) => string;
  teamsOf: (id: string) => Teams;
  failedGames: string[];
  oddsText: string;
  onOdds: (s: string) => void;
  bookDecimal: number | null;
  onRemove: (k: string) => void;
  onClear: () => void;
}) {
  const unavailable = result?.unavailableLegs ?? 0;
  return (
    <section className="mb-6" aria-labelledby="pk-parlay">
      <div className="flex items-baseline justify-between" style={{ borderBottom: "1px solid var(--hair)" }}>
        <h2 id="pk-parlay" className="py-2 text-[14px] font-bold">
          Ticket ({legs.length} {legs.length === 1 ? "selección" : "selecciones"}
          {result && result.games.length > 0 ? ` · ${result.games.length} ${result.games.length === 1 ? "partido" : "partidos"}` : ""})
        </h2>
        {legs.length > 0 && (
          <button type="button" onClick={onClear} className="min-h-[40px] text-[12px] font-medium underline underline-offset-2" style={{ color: "var(--text-2)" }}>
            Vaciar
          </button>
        )}
      </div>

      {legs.length === 0 || !result ? (
        <p className="pt-3 text-[13px]" style={{ color: "var(--text-2)" }}>
          Elige un partido arriba y agrega selecciones; puedes combinar hasta {MAX_GAMES} partidos y {MAX_LEGS} selecciones en el mismo ticket.
        </p>
      ) : (
        <>
          {result.games.map((gr) => {
            const st = statusOf(gr.gameId);
            const teams = teamsOf(gr.gameId);
            const headId = `pk-g-${gr.gameId}`;
            return (
              <div key={gr.gameId} role="group" aria-labelledby={headId} className="pt-3">
                <div className="flex items-baseline justify-between gap-3">
                  <h3 id={headId} className="text-[12px] font-bold uppercase tracking-wider" style={{ color: "var(--text-2)" }}>
                    {labelOf(gr.gameId)}
                  </h3>
                  {gr.joint != null && gr.legs.filter((l) => l.available).length >= 2 && (
                    <span className="text-[12px] tabular-nums" style={{ color: "var(--text-2)" }}>
                      Conjunta del partido <span className="font-bold" style={{ color: "var(--text)" }}>{pct(gr.joint)}</span>
                    </span>
                  )}
                </div>
                <ul>
                  {gr.legs.map((r) => (
                    <li key={r.key} className="flex items-center gap-3 py-[10px]" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
                      <span className="min-w-0 flex-1 text-[13.5px]">
                        {describeLeg(r.leg, st?.inputs, teams)}
                        {r.approx && r.available && (
                          <span className="ml-2 text-[11px]" style={{ color: "var(--text-3)" }}>aproximado</span>
                        )}
                      </span>
                      <span className="text-[14px] font-bold tabular-nums">
                        {r.available ? (
                          pct(r.p)
                        ) : (
                          <span className="text-[12px] font-medium" style={{ color: "var(--text-3)" }}>
                            {st?.loading ? "cargando…" : "no disponible"}
                          </span>
                        )}
                      </span>
                      <button
                        type="button"
                        onClick={() => onRemove(r.key)}
                        aria-label={`Quitar ${labelOf(gr.gameId)}: ${describeLeg(r.leg, st?.inputs, teams)}`}
                        className="min-h-[40px] min-w-[40px] text-[18px] leading-none"
                        style={{ color: "var(--text-3)" }}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}

          {failedGames.length > 0 && (
            <p className="pt-2 text-[12px]" role="status" style={{ color: "var(--side-b)" }}>
              Sin datos del modelo para {failedGames.map(labelOf).join(", ")}: sus selecciones no se cuentan en el ticket.
            </p>
          )}
          {unavailable > 0 && failedGames.length === 0 && (
            <p className="pt-2 text-[12px]" style={{ color: "var(--side-b)" }}>
              {unavailable} {unavailable === 1 ? "selección sin datos y no se cuenta" : "selecciones sin datos y no se cuentan"} en el ticket.
            </p>
          )}

          <div className="grid grid-cols-2 gap-4 py-4" aria-live="polite">
            <div>
              <div className={label} style={{ color: "var(--text-3)" }}>Ticket (correlación por partido)</div>
              <div className="text-[28px] font-bold tabular-nums">{pct(result.joint)}</div>
            </div>
            <div>
              <div className={label} style={{ color: "var(--text-3)" }}>Si todo fuera independiente</div>
              <div className="text-[28px] font-bold tabular-nums" style={{ color: "var(--text-2)" }}>{pct(result.independent)}</div>
            </div>
            <div>
              <div className={label} style={{ color: "var(--text-3)" }}>Cuota justa decimal</div>
              <div className="text-[18px] font-bold tabular-nums">{result.fairDecimal ? result.fairDecimal.toFixed(2) : "—"}</div>
            </div>
            <div>
              <div className={label} style={{ color: "var(--text-3)" }}>Cuota justa americana</div>
              <div className="text-[18px] font-bold tabular-nums">{formatAmerican(result.fairAmerican)}</div>
            </div>
          </div>

          <div className="flex flex-wrap items-end gap-3 pb-2">
            <Field id="pk-odds" text="Cuota del libro (opcional)">
              <input
                id="pk-odds"
                inputMode="text"
                placeholder="+450 o 5.50"
                className="min-h-[40px] w-[160px] rounded-lg px-3 text-[14px] tabular-nums"
                style={fieldStyle}
                value={oddsText}
                onChange={(e) => onOdds(e.target.value)}
              />
            </Field>
            <div aria-live="polite">
              <div className={label} style={{ color: "var(--text-3)" }}>EV por 1 apostado</div>
              <div className="text-[18px] font-bold tabular-nums">
                {result.ev != null ? `${result.ev >= 0 ? "+" : "−"}${Math.abs(result.ev).toFixed(2)}` : "—"}
              </div>
            </div>
          </div>
          {oddsText.trim() !== "" && bookDecimal == null && (
            <p className="text-[12px]" style={{ color: "var(--side-b)" }}>Cuota no válida. Usa americana (+450, -110) o decimal (5.50).</p>
          )}

          <ul className="mt-2 list-disc pl-4 text-[11.5px] leading-relaxed" style={{ color: "var(--text-3)" }}>
            {result.approx && (
              <li>Primer TD es aproximado y se trata como independiente del resto de selecciones.</li>
            )}
            {result.repairShrink > 0 && (
              <li>La matriz de correlación no era válida y se corrigió (suavizada {pct(result.repairShrink, 0)} hacia independencia).</li>
            )}
            <li>Partidos distintos se tratan como independientes entre sí; la correlación solo aplica dentro de cada partido.</li>
            <li>Simulación Monte Carlo de {result.sims.toLocaleString("es-MX")} escenarios con cópula gaussiana.</li>
          </ul>
        </>
      )}
    </section>
  );
}

// ------------------------------------------------------------ quien anotaria

function Scorers({ inputs, legs, onAdd }: { inputs: ParlayInputs; legs: Set<string>; onAdd: (l: Leg) => void }) {
  const rows = useMemo(
    () =>
      inputs.players
        .map((p) => ({ p, any: anytimeTdProb(inputs, p.gsis_id), first: firstTdProb(inputs, p.gsis_id) }))
        .filter((r): r is { p: typeof r.p; any: number; first: number | null } => r.any != null)
        .sort((a, b) => b.any - a.any),
    [inputs]
  );
  const inParlay = legs;
  return (
    <section className="mb-4" aria-labelledby="pk-scorers">
      <h2 id="pk-scorers" className="py-2 text-[14px] font-bold" style={{ borderBottom: "1px solid var(--hair)" }}>
        ¿Quién anotaría? · {gameLabel(inputs.game)}
      </h2>
      {rows.length === 0 ? (
        <p className="pt-3 text-[13px]" style={{ color: "var(--text-2)" }}>Sin proyecciones de TD para este partido.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[340px] border-collapse text-[13.5px] tabular-nums">
            <thead>
              <tr style={{ color: "var(--text-3)", borderBottom: "1px solid var(--hair)" }}>
                <th scope="col" className="py-2 text-left text-[11px] font-semibold uppercase tracking-wider">Jugador</th>
                <th scope="col" className="py-2 text-right text-[11px] font-semibold uppercase tracking-wider">Anota TD</th>
                <th scope="col" className="py-2 text-right text-[11px] font-semibold uppercase tracking-wider">Primer TD</th>
                <th scope="col" className="py-2 text-right text-[11px] font-semibold uppercase tracking-wider"><span className="sr-only">Agregar</span></th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ p, any, first }) => {
                const aKey = legKey({ kind: "anytime_td", gsis_id: p.gsis_id });
                const fKey = legKey({ kind: "first_td", gsis_id: p.gsis_id });
                return (
                  <tr key={p.gsis_id} style={{ borderBottom: "1px solid var(--hair-soft)" }}>
                    <td className="py-[9px] pr-2">
                      <span className="font-semibold">{p.player_name}</span>{" "}
                      <span className="text-[11.5px]" style={{ color: "var(--text-3)" }}>{p.team}{p.position ? ` · ${p.position}` : ""}</span>
                    </td>
                    <td className="py-[9px] text-right font-bold">{pct(any, 0)}</td>
                    <td className="py-[9px] text-right">{first != null ? pct(first, 1) : "—"}</td>
                    <td className="py-[9px] pl-2 text-right whitespace-nowrap">
                      <button
                        type="button"
                        disabled={inParlay.has(aKey)}
                        onClick={() => onAdd({ kind: "anytime_td", gsis_id: p.gsis_id })}
                        aria-label={`Agregar al ticket: ${p.player_name} anota TD`}
                        className="min-h-[40px] min-w-[44px] text-[12.5px] font-semibold disabled:opacity-40"
                        style={{ color: "var(--accent)" }}
                      >
                        + TD
                      </button>
                      <button
                        type="button"
                        disabled={first == null || inParlay.has(fKey)}
                        onClick={() => onAdd({ kind: "first_td", gsis_id: p.gsis_id })}
                        aria-label={`Agregar al ticket: ${p.player_name} primer TD`}
                        className="min-h-[40px] min-w-[44px] text-[12.5px] font-semibold disabled:opacity-40"
                        style={{ color: "var(--accent)" }}
                      >
                        + 1º
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-2 text-[11px]" style={{ color: "var(--text-3)" }}>
        Anota TD incluye recepción y carrera. Primer TD es una aproximación a partir de la tasa de TD de cada jugador.
      </p>
    </section>
  );
}

export default function PicksPage() {
  return (
    <Suspense fallback={<div className="px-[22px] pt-[22px] text-[13px]" style={{ color: "var(--text-2)" }}>Cargando…</div>}>
      <PicksInner />
    </Suspense>
  );
}
