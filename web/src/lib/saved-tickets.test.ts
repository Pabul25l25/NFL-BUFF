import { describe, expect, it } from "vitest";
import {
  MAX_SAVED,
  SavedTicket,
  addSaved,
  deleteSaved,
  parseSavedTicket,
  parseSavedTickets,
  renameSaved,
  savedGameIds,
} from "./saved-tickets";
import { formatTicketText } from "./ticket-export";

const good = (id = "x1"): SavedTicket => ({
  id,
  name: "Mi ticket",
  created: "2026-10-11T12:00:00.000Z",
  legs: ["2026_05_BUF_KC~w:home", "2026_05_BUF_KC~p:A:receiving_yards:over:70.5", "2026_05_DEN_LV~t:over:44.5"],
  legText: ["Gana KC", "A · Yds recepción · Más de 70.5", "Total · Más de 44.5"],
  games: { "2026_05_BUF_KC": "BUF @ KC" },
  probability: 0.12,
  fairDecimal: 8.3,
  calibration: { ece: 0.04, n: 500 },
});

describe("tickets guardados", () => {
  it("acepta un ticket valido y conserva sus campos", () => {
    const t = parseSavedTicket(good());
    expect(t).toEqual(good());
    expect(savedGameIds(good())).toEqual(["2026_05_BUF_KC", "2026_05_DEN_LV"]);
  });

  it("rechaza formas malformadas", () => {
    const bad: unknown[] = [
      null,
      "texto",
      42,
      [],
      {},
      { ...good(), id: "" },
      { ...good(), id: 7 },
      { ...good(), name: 5 },
      { ...good(), created: "no es fecha" },
      { ...good(), legs: [] },
      { ...good(), legs: "w:home" },
      { ...good(), legs: ["sin-partido-valido"] },
      { ...good(), legs: ["g~p:A:receiving_yards:sideways:1"] },
      { ...good(), legs: Array.from({ length: 21 }, () => "g~w:home") },
      { ...good(), probability: 1.5 },
      { ...good(), probability: NaN },
      { ...good(), probability: "0.1" },
      { ...good(), fairDecimal: -2 },
      { ...good(), calibration: { ece: "x", n: 1 } },
      { ...good(), calibration: "alta" },
    ];
    for (const b of bad) expect(parseSavedTicket(b)).toBeNull();
  });

  it("tolera campos opcionales ausentes", () => {
    const t = parseSavedTicket({ ...good(), legText: undefined, games: undefined, fairDecimal: undefined, calibration: undefined });
    expect(t).not.toBeNull();
    expect(t?.legText).toEqual(good().legs);
    expect(t?.games).toEqual({});
    expect(t?.fairDecimal).toBeNull();
    expect(t?.calibration).toBeNull();
  });

  it("parseSavedTickets descarta invalidos y duplicados, y limita a 50", () => {
    expect(parseSavedTickets("x")).toEqual([]);
    expect(parseSavedTickets({ a: 1 })).toEqual([]);
    const mixed = parseSavedTickets([good("a"), { nope: true }, good("a"), good("b")]);
    expect(mixed.map((t) => t.id)).toEqual(["a", "b"]);
    const many = parseSavedTickets(Array.from({ length: 80 }, (_, i) => good(`id${i}`)));
    expect(many).toHaveLength(MAX_SAVED);
  });

  it("agrega con tope, renombra y elimina", () => {
    let list: SavedTicket[] = [];
    for (let i = 0; i < MAX_SAVED; i++) list = addSaved(list, good(`id${i}`)).list;
    const r = addSaved(list, good("extra"));
    expect(r.error).toBeTruthy();
    expect(r.list).toHaveLength(MAX_SAVED);
    expect(renameSaved(list, "id1", "  Nuevo  ").find((t) => t.id === "id1")?.name).toBe("Nuevo");
    expect(renameSaved(list, "id1", "   ")).toBe(list);
    expect(deleteSaved(list, "id1")).toHaveLength(MAX_SAVED - 1);
  });
});

describe("texto exportado", () => {
  it("incluye selecciones, conjunta y cuota justa", () => {
    const txt = formatTicketText({
      name: "Prueba",
      legs: [
        { game: "BUF @ KC", text: "Gana KC", p: 0.6 },
        { game: "BUF @ KC", text: "A · Más de 70.5", p: 0.4 },
        { game: "DEN @ LV", text: "Total · Más de 44.5", p: null },
      ],
      joint: 0.2,
      independent: 0.24,
      fairDecimal: 5,
      fairAmerican: 400,
      calibration: { ece: 0.04, n: 500 },
      url: "https://x.test/picks?legs=a",
    });
    expect(txt).toContain("Ticket: Prueba");
    expect(txt).toContain("Gana KC: 60.0 %");
    expect(txt).toContain("sin datos");
    expect(txt).toContain("20.0 %");
    expect(txt).toContain("5.00 (+400)");
    expect(txt).toContain("no es recomendación de apuesta");
  });
});
