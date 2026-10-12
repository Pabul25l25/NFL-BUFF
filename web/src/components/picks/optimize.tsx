"use client";

import { useMemo, useState } from "react";
import { Leg, ParlayInputs, TicketSim, formatAmerican, parseOdds, toAmerican } from "@/lib/parlay";
import { altLines, lineStep, optimizeToOdds, suggestCorrelated } from "@/lib/parlay-analysis";
import { describeLeg, pct } from "./labels";
import { PicksCtx } from "./analysis";
import { Field, accentBtn, btn, fieldStyle, label } from "./ui";

export interface LegChange {
  index: number;
  leg: Leg;
}

// ---------------------------------------------------------------- lineas alternativas

export function AltLinesPanel({
  ts,
  ctx,
  onReplace,
}: {
  ts: TicketSim;
  ctx: PicksCtx;
  onReplace: (changes: LegChange[]) => void;
}) {
  const adjustable = useMemo(
    () => ts.ticket.map((t, i) => i).filter((i) => lineStep(ts.ticket[i].leg) != null && ts.legs[i] && ts.games[ts.legs[i].gameIdx]?.results[ts.legs[i].localIdx].available),
    [ts]
  );
  const [sel, setSel] = useState(adjustable[0] ?? 0);
  const index = adjustable.includes(sel) ? sel : (adjustable[0] ?? -1);
  const rows = useMemo(() => (index >= 0 ? altLines(ts, index) : []), [ts, index]);

  if (adjustable.length === 0) {
    return <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Agrega una selección de jugador, total o spread para ver líneas alternativas.</p>;
  }
  return (
    <div>
      <Field id="pk-alt-leg" text="Selección">
        <select
          id="pk-alt-leg"
          className="min-h-[40px] w-full rounded-lg px-2 text-[14px]"
          style={fieldStyle}
          value={index}
          onChange={(e) => setSel(Number(e.target.value))}
        >
          {adjustable.map((i) => (
            <option key={i} value={i}>{ctx.legText(i)}</option>
          ))}
        </select>
      </Field>
      <div className="overflow-x-auto">
        <table className="mt-3 w-full min-w-[320px] border-collapse text-[13px] tabular-nums">
          <caption className="sr-only">Probabilidad a distintas líneas</caption>
          <thead>
            <tr style={{ color: "var(--text-3)", borderBottom: "1px solid var(--hair)" }}>
              <th scope="col" className="py-2 text-left text-[11px] font-semibold uppercase tracking-wider">Línea</th>
              <th scope="col" className="py-2 text-right text-[11px] font-semibold uppercase tracking-wider">Sola</th>
              <th scope="col" className="py-2 text-right text-[11px] font-semibold uppercase tracking-wider">Ticket</th>
              <th scope="col" className="py-2 text-right text-[11px] font-semibold uppercase tracking-wider"><span className="sr-only">Usar</span></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.offset} style={{ borderBottom: "1px solid var(--hair-soft)", color: r.current ? "var(--accent)" : undefined }}>
                <td className="py-2 font-semibold">{r.line}{r.current ? " (actual)" : ""}</td>
                <td className="py-2 text-right">{pct(r.marginal)}</td>
                <td className="py-2 text-right font-semibold">{pct(r.ticketP)}</td>
                <td className="py-1 text-right">
                  {!r.current && (
                    <button
                      type="button"
                      className="min-h-[40px] min-w-[56px] text-[12.5px] font-semibold"
                      style={{ color: "var(--accent)" }}
                      onClick={() => onReplace([{ index, leg: r.leg }])}
                      aria-label={`Cambiar a línea ${r.line}`}
                    >
                      Usar
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px]" style={{ color: "var(--text-3)" }}>
        &quot;Ticket&quot; es la probabilidad conjunta si solo cambia esta línea.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- cuota objetivo

export function TargetOddsPanel({
  ts,
  ctx,
  onReplace,
}: {
  ts: TicketSim;
  ctx: PicksCtx;
  onReplace: (changes: LegChange[]) => void;
}) {
  const [text, setText] = useState("");
  const target = parseOdds(text);
  const res = useMemo(() => (target ? optimizeToOdds(ts, target) : null), [ts, target]);
  return (
    <div>
      <Field id="pk-target" text="Cuota objetivo (decimal o americana)">
        <input
          id="pk-target"
          inputMode="text"
          placeholder="+500 o 6.00"
          className="min-h-[40px] w-[160px] rounded-lg px-3 text-[14px] tabular-nums"
          style={fieldStyle}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </Field>
      <p className="mt-1 text-[11.5px]" style={{ color: "var(--text-3)" }}>
        Busca el ticket más probable cuya cuota justa sea al menos la objetivo, moviendo líneas hasta 3 pasos.
      </p>
      {text.trim() !== "" && target == null && (
        <p className="mt-2 text-[12px]" style={{ color: "var(--side-b)" }}>Cuota no válida. Usa americana (+450, -110) o decimal (5.50).</p>
      )}
      {res && (
        <div className="mt-3" aria-live="polite">
          {res.status === "no-adjustable" ? (
            <p className="text-[13px]" style={{ color: "var(--text-2)" }}>
              Ninguna selección tiene línea ajustable (solo jugador, total y spread).
            </p>
          ) : (
            <>
              <p className="text-[13px] tabular-nums">
                Ahora: {pct(res.startJoint)} (cuota justa {res.startJoint > 0 ? (1 / res.startJoint).toFixed(2) : "—"}).{" "}
                Con los cambios: <span className="font-bold">{pct(res.joint)}</span> (cuota justa{" "}
                <span className="font-bold">{res.fairDecimal ? res.fairDecimal.toFixed(2) : "—"}</span>
                {res.joint > 0 && res.joint < 1 ? `, ${formatAmerican(toAmerican(res.joint))}` : ""}).
              </p>
              {res.status === "unreachable" && (
                <p className="mt-1 text-[12.5px]" style={{ color: "var(--side-b)" }}>
                  Con estas líneas no se llega a la cuota objetivo; este es el ticket más largo que se alcanzó.
                </p>
              )}
              {res.changes.length === 0 ? (
                <p className="mt-2 text-[12.5px]" style={{ color: "var(--text-2)" }}>
                  No hace falta mover líneas: el ticket ya es el más probable que cumple la cuota objetivo.
                </p>
              ) : (
                <>
                  <ul className="mt-2">
                    {res.changes.map((c) => (
                      <li key={c.index} className="py-2 text-[12.5px] tabular-nums" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
                        {ctx.legText(c.index)} <span style={{ color: "var(--text-3)" }}>→ línea {c.to}</span>
                      </li>
                    ))}
                  </ul>
                  <button
                    type="button"
                    className={`${btn} mt-3`}
                    style={accentBtn}
                    onClick={() => onReplace(res.changes.map((c) => ({ index: c.index, leg: c.leg })))}
                  >
                    Aplicar cambios al ticket
                  </button>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- sugerencias correlacionadas

export function SuggestionsPanel({
  ts,
  ctx,
  inputsByGame,
  full,
  onAdd,
}: {
  ts: TicketSim;
  ctx: PicksCtx;
  inputsByGame: Record<string, ParlayInputs | undefined>;
  full: boolean;
  onAdd: (gameId: string, leg: Leg) => void;
}) {
  const sugg = useMemo(() => suggestCorrelated(inputsByGame, ts), [inputsByGame, ts]);
  if (sugg.length === 0) {
    return (
      <p className="text-[13px]" style={{ color: "var(--text-2)" }}>
        No hay selecciones del mismo partido con correlación positiva suficiente con tu ticket.
      </p>
    );
  }
  return (
    <div>
      <p className="text-[11.5px]" style={{ color: "var(--text-3)" }}>
        Selecciones del mismo partido que tienden a cumplirse junto con las tuyas (correlación del modelo).
      </p>
      <ul className="mt-1">
        {sugg.map((s) => {
          const text = describeLeg(s.leg, ctx.inputsOf(s.gameId), ctx.teamsOf(s.gameId));
          return (
            <li key={`${s.gameId}-${text}`} className="flex items-center gap-3 py-2" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
              <span className="min-w-0 flex-1 text-[12.5px] leading-snug">
                <span className="block text-[13.5px]">{text}</span>
                <span className="tabular-nums" style={{ color: "var(--text-3)" }}>
                  {ctx.labelOf(s.gameId)} · sola {pct(s.marginal)} · correlación {s.corr.toFixed(2)}
                </span>
              </span>
              <span className="shrink-0 text-right text-[12px] tabular-nums" style={{ color: "var(--text-2)" }}>
                <span className={label} style={{ color: "var(--text-3)" }}>Ticket</span>
                <span className="block text-[14px] font-bold" style={{ color: "var(--text)" }}>{pct(s.jointAfter)}</span>
                <span>antes {pct(s.jointBefore)}</span>
              </span>
              <button
                type="button"
                className="min-h-[40px] min-w-[56px] text-[12.5px] font-semibold disabled:opacity-40"
                style={{ color: "var(--accent)" }}
                disabled={full}
                onClick={() => onAdd(s.gameId, s.leg)}
                aria-label={`Agregar al ticket: ${text}`}
              >
                Agregar
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
