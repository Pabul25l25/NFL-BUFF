"use client";

import { useState } from "react";
import { Leg, ParlayInputs, Side, TeamSide, evaluateParlay, halfLine, legKey } from "@/lib/parlay";
import { defaultPlayerLine, fmtLine, gameLabel, marketLabel, marketsFor, pct, playerProjection } from "./labels";
import { Field, btn, fieldStyle, label } from "./ui";

export function Builder({ inputs, inTicket, onAdd }: { inputs: ParlayInputs; inTicket: Set<string>; onAdd: (l: Leg) => void }) {
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
  const proj = playerId && market && !isTd ? playerProjection(inputs, playerId, market) : null;
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
            {proj && proj.mean != null && (
              <p className="text-[12.5px] tabular-nums sm:col-span-2" style={{ color: "var(--text-2)" }}>
                Media <span className="font-bold" style={{ color: "var(--text)" }}>{proj.mean.toFixed(1)}</span>
                {proj.median != null && (
                  <span className="text-[11px]" style={{ color: "var(--text-3)" }}>
                    {" "}· Línea 50/50 {proj.median.toFixed(1)}
                  </span>
                )}
              </p>
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
