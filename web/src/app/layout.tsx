import type { Metadata } from "next";
import { Barlow_Condensed, Work_Sans } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const workSans = Work_Sans({
  variable: "--font-work-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const barlow = Barlow_Condensed({
  variable: "--font-barlow",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

export const metadata: Metadata = {
  title: "NFL-Predictor",
  description: "Predicciones NFL de los partidos que vienen: ganador, spread, total y proyección de rendimiento de jugadores, con justificación en lenguaje natural.",
  robots: { index: false, follow: false },
};

export const viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#F3F5F8" },
    { media: "(prefers-color-scheme: dark)", color: "#0F1216" },
  ],
  viewportFit: "cover" as const,
};

// Fija data-theme antes del primer pintado y sincroniza los meta theme-color. Si los meta
// aun no existen (el script corre en <head> antes que ellos) reintenta en DOMContentLoaded
// y, si siguen sin existir, crea uno.
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("theme");if(t!=="light"&&t!=="dark")return;document.documentElement.dataset.theme=t;var c=t==="dark"?"#0F1216":"#F3F5F8";function s(){var m=document.querySelectorAll('meta[name="theme-color"]');if(!m.length){var n=document.createElement("meta");n.name="theme-color";document.head.appendChild(n);m=[n]}for(var i=0;i<m.length;i++){m[i].removeAttribute("media");m[i].setAttribute("content",c)}}s();document.addEventListener("DOMContentLoaded",s)}catch(e){}})();`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="es" className={`${workSans.variable} ${barlow.variable} h-full antialiased`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-full flex flex-col bg-[var(--bg)] text-[var(--text)]">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
