"use client";

import { useMemo, useState } from "react";
import { Leg, TicketSim, formatAmerican, toAmerican } from "@/lib/parlay";
import { legCountDistribution, roundRobin, scenarioSummary, typicalScenario, weakestLink } from "@/lib/parlay-analysis";
import { Teams, pct, playerName } from "./labels";
import { btn, label } from "./ui";
import type { ParlayInputs } from "@/lib/parlay";

export interface PicksCtx {
  inputsOf: (gameId: string) => ParlayInputs | undefined;
  teamsOf: (gameId: string) => Teams;
  labelOf: (gameId: string) => string;
  /** texto de la seleccion i del ticket */
  legText: (i: number) => string;
}

const UNIT: Record<string, string> = {
  receiving_yards: "yds",
  rushing_yards: "yds",
  passing_yards: "yds",
  receptions: "rec",
  passing_tds: "TD pase",
  sacks_suffered: "capturas",
};

const r0 = (x: number) => Math.round(x).toString();

/** Frase del valor tipico de una seleccion; null si no aporta (se agrupa con otras del mismo partido). */
function typicalPhrase(leg: Leg, gameId: string, median: number, ctx: PicksCtx): string {
  const inputs = ctx.inputsOf(gameId);
  const t = ctx.teamsOf(gameId);
  const name = (id: string) => (inputs ? playerName(inputs, id) : id);
  switch (leg.kind) {
    case "player":
      return `${name(leg.gsis_id)} ${r0(median)} ${UNIT[leg.target] ?? ""}`.trim();
    case "anytime_td":
    case "two_td":
      return `${name(leg.gsis_id)} ${r0(median)} TD`;
    case "winner":
    case "spread":
      return median >= 0 ? `${t.home} gana por ${r0(median)}` : `${t.away} gana por ${r0(-median)}`;
    case "total":
      return `total ${r0(median)}`;
    default:
      return "";
  }
}

export function ScenarioPanel({ ts, ctx }: { ts: TicketSim; ctx: PicksCtx }) {
  const summary = useMemo(() => scenarioSummary(ts), [ts]);
  const dist = useMemo(() => legCountDistribution(ts), [ts]);
  const weak = useMemo(() => weakestLink(ts), [ts]);
  const typical = useMemo(() => typicalScenario(ts), [ts]);

  if (summary.active === 0) {
    return <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Sin selecciones con datos para simular.</p>;
  }
  const fmt = (n: number) => n.toLocaleString("es-MX");
  const maxP = Math.max(...dist.probs, 1e-9);

  // frases tipicas deduplicadas por partido (margen y total se repiten entre selecciones)
  const phrases: string[] = [];
  for (const row of typical.rows) {
    const t = ts.ticket[row.index];
    const ph = typicalPhrase(t.leg, t.gameId, row.median, ctx);
    if (ph && !phrases.includes(ph)) phrases.push(ph);
  }

  return (
    <div className="flex flex-col gap-5">
      <p className="text-[14px] tabular-nums" aria-live="polite">
        Se cumple en <span className="font-bold">{fmt(summary.hits)}</span> de {fmt(summary.sims)} simulaciones{" "}
        <span style={{ color: "var(--text-2)" }}>({pct(summary.hits / summary.sims)})</span>.
        {ts.gameIds.length > 1 && (
          <span className="block text-[11.5px]" style={{ color: "var(--text-3)" }}>
            Con varios partidos se empareja la simulación i de cada uno (partidos independientes).
          </span>
        )}
      </p>

      <div>
        <h3 className={label} style={{ color: "var(--text-3)" }}>Selecciones cumplidas (de {dist.m})</h3>
        <ul className="mt-2" aria-label="Distribución de selecciones cumplidas">
          {dist.probs.map((p, k) => (
            <li key={k} className="flex items-center gap-3 py-[3px] text-[12.5px] tabular-nums">
              <span className="w-[44px] shrink-0">{k} de {dist.m}</span>
              <span className="h-[10px] flex-1" style={{ background: "var(--hair-soft)" }} aria-hidden>
                <span
                  className="block h-full"
                  style={{ width: `${(p / maxP) * 100}%`, background: k === dist.m ? "var(--accent)" : "var(--side-b)" }}
                />
              </span>
              <span className="w-[56px] shrink-0 text-right font-semibold">{pct(p)}</span>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className={label} style={{ color: "var(--text-3)" }}>Eslabón más débil</h3>
        {weak.rows.length < 2 || weak.worst == null ? (
          <p className="mt-2 text-[12.5px]" style={{ color: "var(--text-2)" }}>
            {weak.rows.length < 2 ? "Con una sola selección no hay eslabón que comparar." : "El ticket no falla en ninguna simulación."}
          </p>
        ) : (
          <>
            <p className="mt-2 text-[13px]" role="status">
              <span className="font-bold" style={{ color: "var(--accent)" }}>{ctx.legText(weak.worst)}</span>{" "}
              tumba el ticket en{" "}
              <span className="font-bold tabular-nums">
                {pct(weak.rows.find((r) => r.index === weak.worst)?.missShare ?? 0, 0)}
              </span>{" "}
              de los escenarios fallidos.
            </p>
            <div className="overflow-x-auto">
              <table className="mt-2 w-full min-w-[360px] border-collapse text-[12.5px] tabular-nums">
                <caption className="sr-only">Probabilidad del ticket sin cada selección y su peso en los fallos</caption>
                <thead>
                  <tr style={{ color: "var(--text-3)", borderBottom: "1px solid var(--hair)" }}>
                    <th scope="col" className="py-2 text-left text-[11px] font-semibold uppercase tracking-wider">Selección</th>
                    <th scope="col" className="py-2 text-right text-[11px] font-semibold uppercase tracking-wider">Sin ella</th>
                    <th scope="col" className="py-2 text-right text-[11px] font-semibold uppercase tracking-wider">Tumba</th>
                  </tr>
                </thead>
                <tbody>
                  {[...weak.rows].sort((a, b) => b.missShare - a.missShare).map((r) => (
                    <tr key={r.index} style={{ borderBottom: "1px solid var(--hair-soft)", color: r.index === weak.worst ? "var(--accent)" : undefined }}>
                      <td className="py-2 pr-2">{ctx.legText(r.index)}</td>
                      <td className="py-2 text-right font-semibold">{pct(r.pWithout)}</td>
                      <td className="py-2 text-right">{pct(r.missShare, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div>
        <h3 className={label} style={{ color: "var(--text-3)" }}>Escenario típico en que se cumple</h3>
        {typical.n === 0 ? (
          <p className="mt-2 text-[12.5px]" style={{ color: "var(--text-2)" }}>El ticket no se cumple en ninguna simulación.</p>
        ) : (
          <>
            <p className="mt-2 text-[13.5px] leading-relaxed">{phrases.join(", ")}.</p>
            <p className="mt-1 text-[11.5px]" style={{ color: "var(--text-3)" }}>
              Mediana de cada estadística en {fmt(typical.n)} simulaciones ganadoras.
              {typical.n < 100 ? " Muestra pequeña: tómalo con cautela." : ""}
            </p>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- round robin

export function RoundRobinPanel({ ts, ctx }: { ts: TicketSim; ctx: PicksCtx }) {
  const m = useMemo(() => legCountDistribution(ts).m, [ts]);
  const maxK = Math.min(4, m - 1);
  const [kSel, setK] = useState(2);
  const k = Math.min(Math.max(2, kSel), Math.max(2, maxK));
  const rr = useMemo(() => (maxK >= 2 ? roundRobin(ts, k) : null), [ts, k, maxK]);

  if (maxK < 2 || !rr) {
    return (
      <p className="text-[13px]" style={{ color: "var(--text-2)" }}>
        El round robin necesita al menos 3 selecciones con datos.
      </p>
    );
  }
  return (
    <div>
      <div className="flex items-center gap-2" role="group" aria-label="Tamaño de las combinaciones">
        <span className={label} style={{ color: "var(--text-3)" }}>Combinaciones de</span>
        {Array.from({ length: maxK - 1 }, (_, i) => i + 2).map((n) => (
          <button
            key={n}
            type="button"
            className={`${btn} min-w-[44px] tabular-nums`}
            aria-pressed={k === n}
            onClick={() => setK(n)}
            style={{
              background: k === n ? "var(--text)" : "var(--surface)",
              color: k === n ? "var(--bg)" : "var(--text-2)",
              border: `1px solid ${k === n ? "var(--text)" : "var(--hair)"}`,
            }}
          >
            {n}
          </button>
        ))}
      </div>
      <p className="mt-3 text-[13px] tabular-nums" aria-live="polite">
        {rr.total} combinaciones de {rr.k} entre {rr.m} selecciones. Ganadoras esperadas:{" "}
        <span className="font-bold">{rr.expectedWins.toFixed(2)}</span>. Al menos una gana:{" "}
        <span className="font-bold">{pct(rr.pAny)}</span>.
      </p>
      {rr.total > rr.combos.length && (
        <p className="text-[11.5px]" style={{ color: "var(--text-3)" }}>
          Se muestran las {rr.combos.length} más probables; el esperado cuenta las {rr.total}.
        </p>
      )}
      <ul className="mt-2">
        {rr.combos.map((c, n) => (
          <li key={c.indices.join("-")} className="flex items-baseline gap-3 py-2" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
            <span className="w-[22px] shrink-0 text-[11.5px] tabular-nums" style={{ color: "var(--text-3)" }}>{n + 1}</span>
            <span className="min-w-0 flex-1 text-[12.5px] leading-snug">
              {c.indices.map((i) => ctx.legText(i)).join(" + ")}
            </span>
            <span className="shrink-0 text-right text-[13px] font-bold tabular-nums">
              {pct(c.joint)}
              <span className="block text-[11px] font-medium" style={{ color: "var(--text-3)" }}>
                {c.fairDecimal ? c.fairDecimal.toFixed(2) : "—"} · {formatAmerican(toAmerican(c.joint))}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
