"use client";

import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { SavedTicket, savedGameIds, savedLegs } from "@/lib/saved-tickets";
import { LegGrade, TicketStatus, gradeTicket, summarizeHistory } from "@/lib/ticket-grading";
import type { TicketResultsResponse } from "@/lib/ticket-results-types";
import { GAME_ID_RE, formatAmerican, toAmerican } from "@/lib/parlay";
import { pct } from "./labels";
import { accentBtn, btn, fieldStyle, ghostBtn } from "./ui";

const CHUNK = 10;

async function fetchResults(ids: string[]): Promise<TicketResultsResponse> {
  const res = await fetch(`/api/ticket-results?game_ids=${encodeURIComponent(ids.join(","))}`);
  if (!res.ok) throw new Error("No se pudieron cargar los resultados");
  return (await res.json()) as TicketResultsResponse;
}

const GRADE_TEXT: Record<LegGrade, string> = {
  hit: "Acertó",
  miss: "Falló",
  push: "Empate",
  void: "No jugó (anulada)",
  pending: "Pendiente",
  unverifiable: "No verificable",
};
const gradeColor = (g: LegGrade) => (g === "hit" ? "var(--accent)" : g === "miss" ? "var(--side-b)" : "var(--text-3)");

const STATUS_COLOR: Record<TicketStatus, string> = {
  ganado: "var(--accent)",
  perdido: "var(--side-b)",
  pendiente: "var(--text-3)",
  anulado: "var(--text-3)",
  "sin verificar": "var(--text-3)",
};

export function MyTickets({
  saved,
  onLoad,
  onRename,
  onDelete,
}: {
  saved: SavedTicket[];
  onLoad: (t: SavedTicket) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
}) {
  const [editing, setEditing] = useState<{ id: string; value: string } | null>(null);
  const [confirmDel, setConfirmDel] = useState<string | null>(null);

  const chunks = useMemo(() => {
    const ids = Array.from(new Set(saved.flatMap(savedGameIds))).filter((id) => GAME_ID_RE.test(id)).sort();
    const out: string[][] = [];
    for (let i = 0; i < ids.length; i += CHUNK) out.push(ids.slice(i, i + CHUNK));
    return out;
  }, [saved]);

  const queries = useQueries({
    queries: chunks.map((ids) => ({
      queryKey: ["ticket-results", ids.join(",")],
      queryFn: () => fetchResults(ids),
      staleTime: 5 * 60_000,
      retry: 1,
    })),
  });
  const loading = queries.some((q) => q.isLoading);
  const failed = queries.some((q) => q.isError);

  const ok = queries.flatMap((q) => (q.data ? [q.data] : []));
  const merged: TicketResultsResponse | undefined =
    ok.length === 0
      ? undefined
      : {
          games: ok.flatMap((r) => r.games),
          stats: ok.flatMap((r) => r.stats),
          first_td: ok.some((r) => r.first_td == null) ? null : ok.flatMap((r) => r.first_td ?? []),
        };

  const graded = saved.map((t) => ({ t, ...gradeTicket(savedLegs(t), merged) }));
  const history = summarizeHistory(graded.map((g) => ({ probability: g.t.probability, status: g.status })));

  if (saved.length === 0) {
    return (
      <p className="text-[13px]" style={{ color: "var(--text-2)" }}>
        Aún no has guardado tickets. Arma uno arriba y usa &quot;Guardar ticket&quot;.
      </p>
    );
  }

  return (
    <div>
      <div className="mb-3 text-[12.5px] leading-relaxed tabular-nums" aria-live="polite">
        {history.graded === 0 ? (
          <p style={{ color: "var(--text-2)" }}>
            Aún no hay tickets calificados ({saved.length} guardados). Se califican solos cuando terminan sus partidos.
          </p>
        ) : (
          <>
            <p>
              {history.graded} {history.graded === 1 ? "ticket calificado" : "tickets calificados"}: se cumplieron{" "}
              <span className="font-bold">{history.wins}</span> ({pct(history.hitRate ?? 0, 0)}). Probabilidad media predicha{" "}
              <span className="font-bold">{pct(history.avgPredicted ?? 0, 0)}</span>.
            </p>
            <ul className="mt-1" style={{ color: "var(--text-2)" }}>
              {history.bands.map((b) => (
                <li key={b.label}>
                  Tus tickets de ~{(b.avgPredicted * 100).toFixed(0)} % se cumplieron {b.wins} de {b.n}.
                </li>
              ))}
            </ul>
            {history.graded < 20 && (
              <p className="text-[11.5px]" style={{ color: "var(--text-3)" }}>Muestra pequeña: la calibración aún dice poco.</p>
            )}
          </>
        )}
        {loading && <p style={{ color: "var(--text-3)" }}>Cargando resultados…</p>}
        {failed && !loading && (
          <p style={{ color: "var(--side-b)" }}>No se pudieron cargar algunos resultados; esos tickets figuran como pendientes.</p>
        )}
      </div>

      <ul>
        {graded.map(({ t, grades, status }) => (
          <li key={t.id} className="py-3" style={{ borderBottom: "1px solid var(--hair)" }}>
            <div className="flex items-baseline justify-between gap-3">
              {editing?.id === t.id ? (
                <form
                  className="flex flex-1 gap-2"
                  onSubmit={(e) => {
                    e.preventDefault();
                    onRename(t.id, editing.value);
                    setEditing(null);
                  }}
                >
                  <label htmlFor={`pk-rn-${t.id}`} className="sr-only">Nuevo nombre</label>
                  <input
                    id={`pk-rn-${t.id}`}
                    autoFocus
                    maxLength={80}
                    className="min-h-[40px] min-w-0 flex-1 rounded-lg px-3 text-[14px]"
                    style={fieldStyle}
                    value={editing.value}
                    onChange={(e) => setEditing({ id: t.id, value: e.target.value })}
                  />
                  <button type="submit" className={btn} style={accentBtn}>Guardar</button>
                  <button type="button" className={btn} style={ghostBtn} onClick={() => setEditing(null)}>Cancelar</button>
                </form>
              ) : (
                <h3 className="min-w-0 truncate text-[14px] font-bold">{t.name || "Ticket sin nombre"}</h3>
              )}
              <span className="shrink-0 text-[12px] font-bold uppercase tracking-wide" style={{ color: STATUS_COLOR[status] }}>
                {status}
              </span>
            </div>
            <p className="text-[11.5px] tabular-nums" style={{ color: "var(--text-3)" }}>
              {new Date(t.created).toLocaleDateString("es-MX", { day: "numeric", month: "short", year: "numeric" })} · {t.legs.length}{" "}
              {t.legs.length === 1 ? "selección" : "selecciones"} · {pct(t.probability)} · cuota justa{" "}
              {t.fairDecimal ? t.fairDecimal.toFixed(2) : "—"} ({formatAmerican(toAmerican(t.probability))})
              {t.calibration ? ` · ECE ${pct(t.calibration.ece)} (n=${t.calibration.n})` : ""}
            </p>
            <ul className="mt-1">
              {t.legs.map((l, i) => {
                const gameId = l.split("~")[0];
                return (
                  <li key={l} className="flex items-baseline justify-between gap-3 py-[5px] text-[12.5px]">
                    <span className="min-w-0 flex-1">
                      {t.legText[i]}
                      <span className="block text-[11px]" style={{ color: "var(--text-3)" }}>{t.games[gameId] ?? gameId}</span>
                    </span>
                    <span className="shrink-0 text-[12px] font-semibold" style={{ color: gradeColor(grades[i] ?? "pending") }}>
                      {GRADE_TEXT[grades[i] ?? "pending"]}
                    </span>
                  </li>
                );
              })}
            </ul>
            <div className="mt-1 flex flex-wrap gap-1" role="group" aria-label={`Acciones de ${t.name}`}>
              <button type="button" className="min-h-[40px] px-2 text-[12.5px] font-semibold" style={{ color: "var(--accent)" }} onClick={() => onLoad(t)}>
                Cargar
              </button>
              <button
                type="button"
                className="min-h-[40px] px-2 text-[12.5px] font-semibold"
                style={{ color: "var(--text-2)" }}
                onClick={() => setEditing({ id: t.id, value: t.name })}
              >
                Renombrar
              </button>
              {confirmDel === t.id ? (
                <>
                  <button
                    type="button"
                    className="min-h-[40px] px-2 text-[12.5px] font-bold"
                    style={{ color: "var(--side-b)" }}
                    onClick={() => {
                      onDelete(t.id);
                      setConfirmDel(null);
                    }}
                  >
                    Confirmar eliminar
                  </button>
                  <button type="button" className="min-h-[40px] px-2 text-[12.5px]" style={{ color: "var(--text-2)" }} onClick={() => setConfirmDel(null)}>
                    Cancelar
                  </button>
                </>
              ) : (
                <button type="button" className="min-h-[40px] px-2 text-[12.5px] font-semibold" style={{ color: "var(--text-2)" }} onClick={() => setConfirmDel(t.id)}>
                  Eliminar
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[11px]" style={{ color: "var(--text-3)" }}>
        Guardados en este navegador (máximo 50). Primer TD no se puede verificar todavía; si un jugador no jugó, su selección se anula.
      </p>
    </div>
  );
}
