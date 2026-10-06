"use client";

import { useQuery } from "@tanstack/react-query";
import { BottomNav } from "@/components/bottom-nav";

interface CalibrationBin {
  model_label: string;
  bin_index: number;
  bin_lower: number;
  bin_upper: number;
  mean_predicted: number;
  observed_frequency: number;
  n_samples: number;
}

interface BacktestOverallRow {
  model_label: string;
  ece: number | null;
  log_loss: number | null;
  brier_score: number | null;
}

async function fetchCalibration() {
  const res = await fetch("/api/calibration?target=winner");
  if (!res.ok) throw new Error("No se pudo cargar calibración");
  const data = await res.json();
  return data.bins as CalibrationBin[];
}

async function fetchBacktest() {
  const res = await fetch("/api/backtest?target=winner");
  if (!res.ok) throw new Error("No se pudo cargar el backtest");
  const data = await res.json();
  return data.overall as BacktestOverallRow[];
}

export default function CalibracionPage() {
  const { data: bins, isLoading } = useQuery({ queryKey: ["calibration"], queryFn: fetchCalibration });
  const { data: overall } = useQuery({ queryKey: ["backtest", "winner"], queryFn: fetchBacktest });

  const modelBins = (bins ?? []).filter((b) => b.model_label !== "baseline_mercado");
  const primaryLabel = modelBins[0]?.model_label;
  const primaryBins = modelBins.filter((b) => b.model_label === primaryLabel);

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <div className="text-[16px] font-bold">Calibración</div>
        <p className="mt-2 text-[12.5px] leading-relaxed" style={{ color: "var(--text-2)" }}>
          Si el modelo dice &ldquo;70%&rdquo;, debería acertar el 70% de las veces. La barra azul es lo predicho, la dorada lo observado.
        </p>
      </div>

      <div className="flex-1 px-[22px] pt-4">
        {isLoading && <p className="text-[13px]" style={{ color: "var(--text-2)" }}>Cargando…</p>}
        {!isLoading && primaryBins.length === 0 && (
          <p className="text-[13px]" style={{ color: "var(--text-2)" }}>
            Sin datos de calibración todavía.
          </p>
        )}

        {primaryBins.map((b) => (
          <div key={b.bin_index} className="flex flex-col gap-[6px] py-3" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
            <div className="flex items-baseline justify-between text-[12px]">
              <span style={{ color: "var(--text-2)" }}>
                {(b.bin_lower * 100).toFixed(0)}–{(b.bin_upper * 100).toFixed(0)}%
              </span>
              <span className="tabular-nums" style={{ color: "var(--text-2)" }}>n={b.n_samples}</span>
            </div>
            <div className="relative h-[5px] rounded-[3px]" style={{ background: "var(--hair-soft)", boxShadow: "inset 0 0 0 1px var(--hair)" }}>
              <div className="absolute inset-y-0 left-0 rounded-[3px]" style={{ width: `${b.mean_predicted * 100}%`, background: "var(--side-b)" }} />
            </div>
            <div className="relative h-[5px] rounded-[3px]" style={{ background: "var(--hair-soft)", boxShadow: "inset 0 0 0 1px var(--hair)" }}>
              <div className="absolute inset-y-0 left-0 rounded-[3px]" style={{ width: `${b.observed_frequency * 100}%`, background: "var(--accent)" }} />
            </div>
          </div>
        ))}

        <div className="mt-3 flex gap-[18px] text-[11px] uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
          <span className="flex items-center gap-[6px]">
            <span className="inline-block h-[5px] w-4 rounded-[3px]" style={{ background: "var(--side-b)" }} />
            predicho
          </span>
          <span className="flex items-center gap-[6px]">
            <span className="inline-block h-[5px] w-4 rounded-[3px]" style={{ background: "var(--accent)" }} />
            observado
          </span>
        </div>

        {overall && overall.length > 0 && (
          <>
            <div className="mb-1 mt-8 text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
              Error de calibración (ECE) — menor es mejor
            </div>
            {overall.map((row) => (
              <div key={row.model_label} className="flex items-center justify-between py-3" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
                <span className="text-[13px]">{row.model_label}</span>
                <span className="text-[13px] font-semibold tabular-nums">
                  {row.ece != null ? row.ece.toFixed(4) : "—"}
                </span>
              </div>
            ))}
          </>
        )}
      </div>
      <BottomNav active="calibracion" />
    </div>
  );
}
