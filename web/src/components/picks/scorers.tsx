"use client";

import { useMemo } from "react";
import { Leg, ParlayInputs, anytimeTdProb, firstTdProb, legKey } from "@/lib/parlay";
import { gameLabel, pct } from "./labels";

export function Scorers({ inputs, legs, onAdd }: { inputs: ParlayInputs; legs: Set<string>; onAdd: (l: Leg) => void }) {
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
