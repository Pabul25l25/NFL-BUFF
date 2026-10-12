"use client";

import { useId, useState } from "react";
import { ParlayInputs } from "@/lib/parlay";

export const fieldStyle = { background: "var(--surface)", color: "var(--text)", border: "1px solid var(--hair)" };
export const btn = "min-h-[40px] rounded-lg px-3 py-2 text-[13px] font-semibold";
export const label = "text-[11px] font-semibold uppercase tracking-wider";
export const ghostBtn = {
  background: "var(--surface)",
  color: "var(--text)",
  border: "1px solid var(--hair)",
};
export const accentBtn = { background: "var(--accent)", color: "var(--bg)" };

export interface GameStatus {
  id: string;
  inputs?: ParlayInputs;
  mock: boolean;
  loading: boolean;
  failed: boolean;
}

export function Field({ id, text, children }: { id: string; text: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className={label} style={{ color: "var(--text-3)" }}>{text}</label>
      {children}
    </div>
  );
}

/** Seccion plegable: el contenido (y su calculo) solo existe mientras esta abierta. */
export function Collapsible({
  title,
  hint,
  defaultOpen = false,
  children,
}: {
  title: string;
  hint?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <section className="mb-4" aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`} className="m-0" style={{ borderBottom: "1px solid var(--hair)" }}>
        <button
          type="button"
          className="flex min-h-[44px] w-full items-baseline justify-between gap-3 text-left text-[14px] font-bold"
          aria-expanded={open}
          aria-controls={`${id}-p`}
          onClick={() => setOpen((o) => !o)}
        >
          <span>{title}</span>
          <span className="text-[12px] font-medium" style={{ color: "var(--text-3)" }}>
            {hint ? `${hint} · ` : ""}{open ? "Ocultar" : "Ver"}
          </span>
        </button>
      </h2>
      {open && (
        <div id={`${id}-p`} className="pt-3">
          {children}
        </div>
      )}
    </section>
  );
}
