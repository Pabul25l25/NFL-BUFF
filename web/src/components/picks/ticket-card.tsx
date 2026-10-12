"use client";

import { MAX_GAMES, MAX_LEGS, TicketLeg, TicketResult, formatAmerican } from "@/lib/parlay";
import { Teams, describeLeg, pct } from "./labels";
import { Field, GameStatus, fieldStyle, label } from "./ui";

export function TicketCard({
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
