import { ImageResponse } from "next/og";

export const alt = "NFL Predictor — Proyecciones";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 56,
          background: "#F3F5F8",
          color: "#0F1216",
        }}
      >
        <svg width="260" height="260" viewBox="0 0 64 64">
          <rect width="64" height="64" rx="14" fill="#0b6b4f" />
          <path d="M20 12v18a12 12 0 0 0 24 0V12" fill="none" stroke="#fff" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M32 42v12" stroke="#fff" strokeWidth="5" strokeLinecap="round" />
          <circle cx="32" cy="22" r="5.5" fill="#f4b41a" />
          <circle cx="32" cy="22" r="2" fill="#0b6b4f" />
        </svg>
        <div style={{ fontSize: 68, fontWeight: 700, letterSpacing: -1 }}>NFL Predictor — Proyecciones</div>
      </div>
    ),
    size,
  );
}
