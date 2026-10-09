import Link from "next/link";
import { ThemeToggle } from "./theme-toggle";

type NavKey = "predicciones" | "resultados" | "calibracion" | "produccion" | "props" | "picks" | "modelos";

const ITEMS: { key: NavKey; href: string; label: string; icon: React.ReactNode }[] = [
  {
    key: "predicciones",
    href: "/",
    label: "Predic.",
    icon: (
      <path d="M2 4h12M2 8h8M2 12h10" strokeWidth={1.5} strokeLinecap="round" />
    ),
  },
  {
    key: "resultados",
    href: "/resultados",
    label: "Result.",
    icon: (
      <>
        <circle cx={8} cy={8} r={6} strokeWidth={1.4} />
        <path d="M5.4 8.2l1.8 1.8L10.6 6.2" strokeWidth={1.4} strokeLinecap="round" strokeLinejoin="round" />
      </>
    ),
  },
  {
    key: "calibracion",
    href: "/calibracion",
    label: "Calib.",
    icon: (
      <>
        <circle cx={8} cy={8} r={6} strokeWidth={1.3} />
        <circle cx={8} cy={8} r={3} strokeWidth={1.3} />
        <circle cx={8} cy={8} r={0.7} fill="currentColor" stroke="none" />
      </>
    ),
  },
  {
    key: "produccion",
    href: "/produccion",
    label: "Prod.",
    icon: (
      <>
        <path d="M2 11l3.5-4 2.5 2.5L13 4" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
        <path d="M9.7 4h3.3v3.3" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      </>
    ),
  },
  {
    key: "props",
    href: "/props",
    label: "Props",
    icon: (
      <>
        <circle cx={8} cy={5} r={2.3} strokeWidth={1.4} />
        <path d="M3.2 13c0.6-2.6 2.6-4 4.8-4s4.2 1.4 4.8 4" strokeWidth={1.4} strokeLinecap="round" />
      </>
    ),
  },
  {
    key: "picks",
    href: "/picks",
    label: "Picks",
    icon: (
      <>
        <path d="M3 3.5h10v9H3z" strokeWidth={1.3} strokeLinejoin="round" />
        <path d="M5.5 6.5h5M5.5 9.5h3" strokeWidth={1.3} strokeLinecap="round" />
      </>
    ),
  },
  {
    key: "modelos",
    href: "/modelos",
    label: "Modelos",
    icon: (
      <>
        <path d="M8 2.5l5.5 2.75L8 8l-5.5-2.75L8 2.5z" strokeWidth={1.3} strokeLinejoin="round" />
        <path d="M2.5 8.5L8 11.25l5.5-2.75" strokeWidth={1.3} strokeLinejoin="round" />
        <path d="M2.5 11.5L8 14.25l5.5-2.75" strokeWidth={1.3} strokeLinejoin="round" />
      </>
    ),
  },
];

export function BottomNav({ active }: { active: NavKey }) {
  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-10 flex border-t"
      style={{
        background: "var(--surface)",
        borderColor: "var(--hair)",
        boxShadow: "0 -1px 8px color-mix(in srgb, var(--text) 6%, transparent)",
        height: "calc(62px + env(safe-area-inset-bottom, 0px))",
        paddingBottom: "env(safe-area-inset-bottom, 0px)",
      }}
    >
      {ITEMS.map((item) => {
        const isActive = item.key === active;
        return (
          <Link
            key={item.key}
            href={item.href}
            aria-current={isActive ? "page" : undefined}
            className="relative flex flex-1 flex-col items-center justify-center gap-1 text-[10px] font-medium"
            style={{ color: isActive ? "var(--text)" : "var(--text-2)", fontWeight: isActive ? 600 : 500 }}
          >
            {isActive && (
              <span className="absolute -top-px left-1/2 h-[2px] w-8 -translate-x-1/2" style={{ background: "var(--accent)" }} />
            )}
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" className="h-[18px] w-[18px]">
              {item.icon}
            </svg>
            <span>{item.label}</span>
          </Link>
        );
      })}
      <div className="flex w-[52px] shrink-0 items-center justify-center">
        <ThemeToggle />
      </div>
    </nav>
  );
}
