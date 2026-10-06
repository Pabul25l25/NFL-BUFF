"use client";

import { useSyncExternalStore } from "react";

type Theme = "light" | "dark";

const KEY = "theme";
const META_COLOR: Record<Theme, string> = { light: "#F3F5F8", dark: "#0F1216" };

function current(): Theme {
  const attr = document.documentElement.dataset.theme;
  if (attr === "light" || attr === "dark") return attr;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function subscribe(cb: () => void) {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const obs = new MutationObserver(cb);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  mq.addEventListener("change", cb);
  return () => {
    obs.disconnect();
    mq.removeEventListener("change", cb);
  };
}

function apply(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  // Fija el color de la barra del navegador a la elección (quita el media del meta).
  document.querySelectorAll('meta[name="theme-color"]').forEach((m) => {
    m.removeAttribute("media");
    m.setAttribute("content", META_COLOR[theme]);
  });
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* sin almacenamiento: la elección vale solo para esta sesión */
  }
}

export function ThemeToggle() {
  const theme = useSyncExternalStore<Theme>(subscribe, current, () => "light");
  const isDark = theme === "dark";
  return (
    <button
      type="button"
      onClick={() => apply(isDark ? "light" : "dark")}
      // Etiqueta fija: el estado lo comunica aria-pressed, asi el aria-label no cambia
      // entre el HTML del servidor (siempre "light") y la hidratacion.
      aria-label="Modo oscuro"
      aria-pressed={isDark}
      suppressHydrationWarning
      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[12px] border outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
      style={{
        background: "var(--surface)",
        borderColor: "var(--hair)",
        color: "var(--text)",
        // anillo de foco y su separación con los tokens del tema
        ["--tw-ring-color" as string]: "var(--accent)",
        ["--tw-ring-offset-color" as string]: "var(--bg)",
      }}
    >
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5" aria-hidden>
        {isDark ? (
          <>
            <circle cx={10} cy={10} r={3.4} />
            <path d="M10 2.5v1.8M10 15.7v1.8M2.5 10h1.8M15.7 10h1.8M4.7 4.7l1.3 1.3M14 14l1.3 1.3M4.7 15.3L6 14M14 6l1.3-1.3" />
          </>
        ) : (
          <path d="M16.5 11.8A7 7 0 0 1 8.2 3.5a7 7 0 1 0 8.3 8.3z" />
        )}
      </svg>
    </button>
  );
}
