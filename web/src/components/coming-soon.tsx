import { BottomNav } from "@/components/bottom-nav";

type NavKey = "predicciones" | "resultados" | "calibracion" | "produccion" | "props" | "modelos";

export function ComingSoon({ title, active }: { title: string; active: NavKey }) {
  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="flex flex-1 flex-col items-center justify-center gap-2 px-[22px] text-center">
        <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
          {title}
        </span>
        <p className="text-[15px] font-semibold">Todavía no está disponible</p>
        <p className="max-w-[280px] text-[13px] leading-relaxed" style={{ color: "var(--text-2)" }}>
          Esta sección se está construyendo. Por ahora podés ver Predicciones y Props.
        </p>
      </div>
      <BottomNav active={active} />
    </div>
  );
}
