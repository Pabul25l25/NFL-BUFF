import { formatAmerican } from "@/lib/parlay";
import type { ExportData } from "@/lib/ticket-export";

const FALLBACK = {
  bg: "#0f1216",
  surface: "#171c23",
  hair: "#2a313b",
  text: "#eceff3",
  text2: "#a9b2bf",
  text3: "#8a94a3",
  accent: "#d4a24c",
};

function palette(): typeof FALLBACK {
  try {
    const cs = getComputedStyle(document.documentElement);
    const g = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb;
    return {
      bg: g("--bg", FALLBACK.bg),
      surface: g("--surface", FALLBACK.surface),
      hair: g("--hair", FALLBACK.hair),
      text: g("--text", FALLBACK.text),
      text2: g("--text-2", FALLBACK.text2),
      text3: g("--text-3", FALLBACK.text3),
      accent: g("--accent", FALLBACK.accent),
    };
  } catch {
    return FALLBACK;
  }
}

const FONT = '"Work Sans", system-ui, -apple-system, "Segoe UI", sans-serif';
const pct = (p: number) => `${(p * 100).toFixed(1)} %`;

function wrap(ctx: CanvasRenderingContext2D, text: string, maxW: number): string[] {
  const words = text.split(" ");
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    const t = cur ? `${cur} ${w}` : w;
    if (ctx.measureText(t).width > maxW && cur) {
      lines.push(cur);
      cur = w;
    } else cur = t;
  }
  if (cur) lines.push(cur);
  return lines;
}

/** Dibuja la tarjeta del ticket en un canvas (sin dependencias). Ancho fijo 1080 px, alto segun contenido. */
export function renderTicketCanvas(d: ExportData): HTMLCanvasElement {
  const W = 1080;
  const PAD = 56;
  const c = palette();
  const canvas = document.createElement("canvas");
  const probe = canvas.getContext("2d") as CanvasRenderingContext2D;

  // medicion: construye primero el layout
  type Row = { kind: "game"; text: string } | { kind: "leg"; lines: string[]; p: string };
  const rows: Row[] = [];
  probe.font = `500 30px ${FONT}`;
  const games: string[] = [];
  for (const l of d.legs) if (!games.includes(l.game)) games.push(l.game);
  for (const g of games) {
    rows.push({ kind: "game", text: g.toUpperCase() });
    for (const l of d.legs.filter((x) => x.game === g)) {
      rows.push({ kind: "leg", lines: wrap(probe, l.text, W - PAD * 2 - 170), p: l.p == null ? "—" : pct(l.p) });
    }
  }
  const legH = (r: Row) => (r.kind === "game" ? 62 : 24 + r.lines.length * 38);
  const bodyH = rows.reduce((a, r) => a + legH(r), 0);
  const H = 150 + bodyH + 330;

  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
  ctx.fillStyle = c.bg;
  ctx.fillRect(0, 0, W, H);
  ctx.textBaseline = "alphabetic";

  // cabecera
  ctx.fillStyle = c.accent;
  ctx.fillRect(PAD, 56, 6, 54);
  ctx.fillStyle = c.text;
  ctx.font = `700 44px ${FONT}`;
  ctx.fillText(d.name || "Ticket", PAD + 24, 96);
  ctx.fillStyle = c.text3;
  ctx.font = `500 22px ${FONT}`;
  ctx.fillText(`${d.legs.length} ${d.legs.length === 1 ? "selección" : "selecciones"}`, PAD + 24, 128);

  let y = 150;
  for (const r of rows) {
    if (r.kind === "game") {
      ctx.fillStyle = c.text2;
      ctx.font = `700 22px ${FONT}`;
      ctx.fillText(r.text, PAD, y + 40);
      y += 62;
      continue;
    }
    ctx.fillStyle = c.text;
    ctx.font = `500 30px ${FONT}`;
    r.lines.forEach((ln, i) => ctx.fillText(ln, PAD, y + 38 + i * 38));
    ctx.font = `700 32px ${FONT}`;
    ctx.textAlign = "right";
    ctx.fillText(r.p, W - PAD, y + 38);
    ctx.textAlign = "left";
    const h = legH(r);
    ctx.fillStyle = c.hair;
    ctx.fillRect(PAD, y + h - 1, W - PAD * 2, 1);
    y += h;
  }

  // resumen
  y += 36;
  ctx.fillStyle = c.text3;
  ctx.font = `600 20px ${FONT}`;
  ctx.fillText("PROBABILIDAD CONJUNTA", PAD, y);
  ctx.fillText("CUOTA JUSTA", W / 2 + 20, y);
  ctx.fillStyle = c.accent;
  ctx.font = `700 72px ${FONT}`;
  ctx.fillText(pct(d.joint), PAD, y + 80);
  ctx.fillStyle = c.text;
  ctx.fillText(d.fairDecimal ? d.fairDecimal.toFixed(2) : "—", W / 2 + 20, y + 80);
  ctx.fillStyle = c.text2;
  ctx.font = `500 24px ${FONT}`;
  ctx.fillText(`Si todo fuera independiente: ${pct(d.independent)}`, PAD, y + 124);
  ctx.fillText(`Americana ${formatAmerican(d.fairAmerican)}`, W / 2 + 20, y + 124);
  if (d.calibration) {
    ctx.fillText(`Calibración del modelo: ECE ${pct(d.calibration.ece)} sobre ${d.calibration.n} predicciones`, PAD, y + 160);
  }
  ctx.fillStyle = c.text3;
  ctx.font = `500 20px ${FONT}`;
  ctx.fillText("Estimación del modelo, no es recomendación de apuesta.", PAD, H - 44);
  return canvas;
}

/** Descarga el canvas como PNG. Devuelve false si el navegador no pudo generarlo. */
export function downloadCanvasPng(canvas: HTMLCanvasElement, filename: string): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      canvas.toBlob((blob) => {
        if (!blob) return resolve(false);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 2000);
        resolve(true);
      }, "image/png");
    } catch {
      resolve(false);
    }
  });
}
