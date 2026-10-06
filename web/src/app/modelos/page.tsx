"use client";

import { useQuery } from "@tanstack/react-query";
import { BottomNav } from "@/components/bottom-nav";

interface ModelRow {
  model_id: string;
  target: string;
  family: string;
  version: number;
  status: string;
  metric_name: string | null;
  train_metric: number | null;
  val_metric: number | null;
  train_val_gap: number | null;
  n_train_rows: number | null;
  train_seasons: string | null;
  created_at: string;
  promotion_blocked_reason: string | null;
}

interface LogRow {
  occurred_at: string;
  target: string;
  action: string;
  model_id: string | null;
  previous_model_id: string | null;
  reason: string | null;
}

const STATUS_LABEL: Record<string, string> = {
  production: "producción",
  shadow: "shadow",
  rejected: "rechazado",
  archived: "archivado",
};

async function fetchModels() {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error("No se pudo cargar modelos");
  const data = await res.json();
  return data.models as ModelRow[];
}

async function fetchLog() {
  const res = await fetch("/api/models/log");
  if (!res.ok) throw new Error("No se pudo cargar la bitácora");
  const data = await res.json();
  return data.log as LogRow[];
}

export default function ModelosPage() {
  const { data: models, isLoading } = useQuery({ queryKey: ["models"], queryFn: fetchModels });
  const { data: log } = useQuery({ queryKey: ["models-log"], queryFn: fetchLog });

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <div className="text-[16px] font-bold">Modelos y trazabilidad</div>
      </div>

      <div className="flex-1 px-[22px] pt-4">
        {isLoading && <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Cargando…</p>}
        {!isLoading && (models ?? []).length === 0 && (
          <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Sin modelos registrados todavía.</p>
        )}

        {(models ?? []).map((m) => (
          <div key={m.model_id} className="flex items-center justify-between gap-[10px] py-3" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
            <div>
              <div className="text-[13.5px] font-semibold">
                {m.target} <span style={{ color: "var(--text-3)", fontWeight: 500 }}>· {m.family} v{m.version}</span>
              </div>
              <div className="text-[11px] uppercase tracking-wide" style={{ color: m.status === "production" ? "var(--text)" : "var(--text-2)", fontWeight: m.status === "production" ? 600 : 400 }}>
                {STATUS_LABEL[m.status] ?? m.status}
                {m.train_val_gap != null && ` · brecha ${(m.train_val_gap * 100).toFixed(1)}%`}
              </div>
            </div>
            <div className="shrink-0 text-right">
              <div className="text-[13px] font-semibold tabular-nums">
                {m.val_metric != null ? m.val_metric.toFixed(4) : "—"}
              </div>
              <div className="text-[10px] uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
                {m.metric_name ?? ""}
              </div>
            </div>
          </div>
        ))}

        <div className="mb-1 mt-8 text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
          Bitácora de reentrenamiento
        </div>
        {(log ?? []).map((l, i) => (
          <div key={i} className="flex flex-col gap-1 py-3" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
            <div className="flex items-baseline justify-between">
              <span className="text-[12.5px] font-semibold">{l.target} · {l.action}</span>
              <span className="text-[10.5px] tabular-nums" style={{ color: "var(--text-2)" }}>
                {new Date(l.occurred_at).toLocaleDateString("es-MX")}
              </span>
            </div>
            {l.reason && (
              <p className="text-[11.5px] leading-relaxed" style={{ color: "var(--text-2)" }}>{l.reason}</p>
            )}
          </div>
        ))}
      </div>
      <BottomNav active="modelos" />
    </div>
  );
}
