"use client";

import { useState } from "react";
import { TicketLeg, TicketResult, ticketLegKey } from "@/lib/parlay";
import { MAX_SAVED, SavedTicket, newId } from "@/lib/saved-tickets";
import { ExportData, buildShareUrl, copyText, formatTicketText } from "@/lib/ticket-export";
import { PicksCtx } from "./analysis";
import { accentBtn, btn, fieldStyle, ghostBtn, label } from "./ui";
import { downloadCanvasPng, renderTicketCanvas } from "./ticket-image";

/** Datos exportables del ticket actual. */
export function buildExportData(
  legs: TicketLeg[],
  result: TicketResult,
  ctx: PicksCtx,
  calibration: { ece: number; n: number } | null,
  name?: string,
  url?: string
): ExportData {
  const pOf = new Map<string, number | null>();
  for (const g of result.games) for (const l of g.legs) pOf.set(l.key, l.available ? l.p : null);
  return {
    name,
    legs: legs.map((t, i) => ({ game: ctx.labelOf(t.gameId), text: ctx.legText(i), p: pOf.get(ticketLegKey(t)) ?? null })),
    joint: result.joint,
    independent: result.independent,
    fairDecimal: result.fairDecimal,
    fairAmerican: result.fairAmerican,
    calibration,
    url,
  };
}

export function ShareBar({
  legs,
  result,
  ctx,
  activeGame,
  calibration,
  savedCount,
  onSave,
}: {
  legs: TicketLeg[];
  result: TicketResult;
  ctx: PicksCtx;
  activeGame: string;
  calibration: { ece: number; n: number } | null;
  savedCount: number;
  onSave: (t: SavedTicket) => string | undefined;
}) {
  const [name, setName] = useState("");
  const [msg, setMsg] = useState("");
  const keys = legs.map(ticketLegKey);
  const url = () => buildShareUrl(window.location.origin, window.location.pathname, activeGame, keys);
  const say = (m: string) => setMsg(m);

  const save = () => {
    const games: Record<string, string> = {};
    for (const t of legs) games[t.gameId] = ctx.labelOf(t.gameId);
    const fallback = `Ticket ${new Date().toLocaleDateString("es-MX", { day: "numeric", month: "short" })}`;
    const t: SavedTicket = {
      id: newId(),
      name: name.trim().slice(0, 80) || fallback,
      created: new Date().toISOString(),
      legs: keys,
      legText: legs.map((_, i) => ctx.legText(i)),
      games,
      probability: Math.min(1, Math.max(0, result.joint)),
      fairDecimal: result.fairDecimal,
      calibration,
    };
    const err = onSave(t);
    if (err) say(err);
    else {
      setName("");
      say("Ticket guardado en Mis tickets.");
    }
  };

  return (
    <div className="mb-4">
      <div className="flex flex-wrap items-end gap-2">
        <div className="flex min-w-[180px] flex-1 flex-col gap-1">
          <label htmlFor="pk-save-name" className={label} style={{ color: "var(--text-3)" }}>Nombre para guardar</label>
          <input
            id="pk-save-name"
            maxLength={80}
            placeholder="Mi ticket"
            className="min-h-[40px] w-full rounded-lg px-3 text-[14px]"
            style={fieldStyle}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <button type="button" className={btn} style={accentBtn} onClick={save} disabled={savedCount >= MAX_SAVED || result.activeLegs === 0}>
          Guardar ticket
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label="Compartir y exportar">
        <button
          type="button"
          className={btn}
          style={ghostBtn}
          onClick={async () => say((await copyText(url())) ? "Enlace copiado." : "No se pudo copiar el enlace.")}
        >
          Copiar enlace
        </button>
        <button
          type="button"
          className={btn}
          style={ghostBtn}
          onClick={async () =>
            say((await copyText(formatTicketText(buildExportData(legs, result, ctx, calibration, name.trim() || undefined, url())))) ? "Texto copiado." : "No se pudo copiar el texto.")
          }
        >
          Copiar como texto
        </button>
        <button
          type="button"
          className={btn}
          style={ghostBtn}
          onClick={async () => {
            try {
              const canvas = renderTicketCanvas(buildExportData(legs, result, ctx, calibration, name.trim() || undefined));
              say((await downloadCanvasPng(canvas, "ticket.png")) ? "Imagen descargada." : "No se pudo generar la imagen.");
            } catch {
              say("No se pudo generar la imagen.");
            }
          }}
        >
          Descargar imagen
        </button>
      </div>
      <p className="mt-2 min-h-[18px] text-[12px]" role="status" style={{ color: "var(--text-2)" }}>{msg}</p>
    </div>
  );
}
