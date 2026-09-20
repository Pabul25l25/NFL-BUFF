import type { Metadata } from "next";
import { Work_Sans } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const workSans = Work_Sans({
  variable: "--font-work-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "NFL-Predictor",
  description: "Predicciones NFL: ganador, spread, total y props de jugador, con justificación en lenguaje natural.",
  robots: { index: false, follow: false },
};

export const viewport = {
  themeColor: "#0B0D10",
  viewportFit: "cover" as const,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="es" className={`${workSans.variable} h-full antialiased dark`}>
      <body className="min-h-full flex flex-col bg-[var(--bg)] text-[var(--text)]">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
