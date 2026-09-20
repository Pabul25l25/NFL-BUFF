"use client";

import { use } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { BottomNav } from "@/components/bottom-nav";
import { GamePrediction, Target, displayProb, pickLabel } from "@/lib/types";

async function fetchGame(gameId: string, target: Target) {
  const params = new URLSearchParams({ game_id: gameId, target });
  const res = await fetch(`/api/predictions?${params}`);
  if (!res.ok) throw new Error("No se pudo cargar la predicción");
  const data = await res.json();
  return (data.predictions as GamePrediction[])[0] ?? null;
}

function friendlyFactorLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\br(\d+)\b/g, "últimas $1 semanas")
    .replace(/\bewm\b/g, "promedio ponderado")
    .replace(/\belo\b/gi, "Elo")
    .replace(/\bepa\b/gi, "EPA")
    .replace(/^./, (c) => c.toUpperCase());
}

export default function DetallePage({ params }: { params: Promise<{ gameId: string }> }) {
  const { gameId } = use(params);
  const searchParams = useSearchParams();
  const target = (searchParams.get("target") as Target) ?? "winner";

  const { data: g, isLoading } = useQuery({
    queryKey: ["game", gameId, target],
    queryFn: () => fetchGame(gameId, target),
  });

  let factors: { label: string; value: number }[] = [];
  if (g?.shap_values) {
    try {
      const parsed = JSON.parse(g.shap_values) as Record<string, number>;
      factors = Object.entries(parsed)
        .map(([k, v]) => ({ label: friendlyFactorLabel(k), value: v }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 6);
    } catch {
      factors = [];
    }
  }
  const maxAbs = Math.max(1e-6, ...factors.map((f) => Math.abs(f.value)));

  return (
    <div className="flex min-h-screen flex-col pb-[78px]">
      <div className="px-[22px] pt-[calc(env(safe-area-inset-top,0px)+22px)]">
        <Link href="/" className="text-[12px]" style={{ color: "var(--text-2)" }}>
          ‹ Volver
        </Link>

        {isLoading && (
          <p className="pt-6 text-[13px]" style={{ color: "var(--text-2)" }}>
            Cargando…
          </p>
        )}

        {g && (
          <>
            <div className="flex flex-col gap-1 pb-[18px] pt-1" style={{ borderBottom: "1px solid var(--hair)" }}>
              <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
                Semana {g.week} · {new Date(g.kickoff_utc).toLocaleString("es-MX", { dateStyle: "short", timeStyle: "short" })} · {g.stadium}
              </span>
              <span className="text-[21px] font-bold">
                {g.away_team} <span className="font-normal" style={{ color: "var(--text-3)" }}>en</span> {g.home_team}
              </span>
            </div>

            <div className="pt-[18px]">
              <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--text-3)" }}>
                Predicción · {target === "winner" ? "Ganador" : target === "spread" ? "Spread" : "Total"}
              </span>
              <div className="my-1 text-[42px] font-bold leading-none tabular-nums">
                {pickLabel(g)}
              </div>
              <span className="text-[10.5px] font-medium uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
                Confianza {g.confidence_level}
              </span>
              {g.explanation_es && (
                <p className="mb-[26px] mt-4 text-[13.5px] leading-relaxed" style={{ color: "var(--text-2)" }}>
                  {g.explanation_es}
                </p>
              )}

              {factors.length > 0 && (
                <>
                  <div className="mb-4 flex gap-[18px] text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-2)" }}>
                    <span>— {g.home_team}</span>
                    <span style={{ color: "var(--side-b)" }}>— {g.away_team}</span>
                  </div>
                  {factors.map((f) => {
                    const side = f.value >= 0 ? "home" : "away";
                    const color = side === "home" ? "var(--accent)" : "var(--side-b)";
                    const widthPct = (Math.abs(f.value) / maxAbs) * 50;
                    return (
                      <div key={f.label} className="flex flex-col gap-[5px] py-[10px]" style={{ borderBottom: "1px solid var(--hair-soft)" }}>
                        <div className="flex items-baseline justify-between">
                          <span className="text-[12.5px]">{f.label}</span>
                          <span className="text-[11.5px] font-semibold tabular-nums" style={{ color }}>
                            {f.value >= 0 ? "+" : ""}{f.value.toFixed(3)}
                          </span>
                        </div>
                        <div className="relative h-[3px]" style={{ background: "var(--hair)" }}>
                          <div
                            className="absolute top-0 bottom-0"
                            style={{
                              [side === "home" ? "left" : "right"]: "50%",
                              width: `${widthPct}%`,
                              background: color,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </>
              )}

              <div className="mt-[22px] pt-4 text-[10.5px] leading-loose" style={{ color: "var(--text-3)", borderTop: "1px solid var(--hair)" }}>
                Modelo {g.model_family ?? "—"} v{g.model_version ?? "—"} · predicho{" "}
                {g.predicted_at ? new Date(g.predicted_at).toLocaleString("es-MX") : "—"}
                <br />
                Corte as-of {g.as_of_ts ? new Date(g.as_of_ts).toLocaleString("es-MX") : "—"}
              </div>
            </div>
          </>
        )}
      </div>
      <BottomNav active="predicciones" />
    </div>
  );
}
