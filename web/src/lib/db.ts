import { neon } from "@neondatabase/serverless";

// Se crea de forma perezosa (no al importar el módulo) para que Next.js no
// intente construir el cliente durante "collecting page data" en el build,
// paso en el que las variables de entorno "sensitive" de Vercel no están
// disponibles (solo lo están en runtime, dentro de la función serverless).
export function getSql() {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error("DATABASE_URL no está configurada (ver web/.env.local)");
  }
  return neon(url);
}
