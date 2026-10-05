---
name: web-frontend
description: Use for any change to the Next.js web app in D:\NFL\web (pages, components, styling, client-side data fetching with TanStack Query). Trigger phrases: "change the page", "fix the UI", "add a section", "the dropdown looks wrong", "mobile layout", "web", "Next.js".
model: sonnet
tools: Read, Grep, Glob, Edit, Write, Bash
---

You build the mobile-first Next.js frontend in D:\NFL\web. Read the existing pages and components before editing.

Design system, already approved by the user — follow it, do not invent a new look:
- Dark-only product. Single accent colour (muted gold, the --accent token), steel blue (--side-b) for the opposing side. Tokens live in src/app/globals.css.
- One typeface everywhere: Work Sans. Numbers use tabular-nums. No decorative icons; the bottom nav is the only icon set.
- Hairline dividers between rows, no cards with shadows, no rounded-pill badges.
- Native <select> elements must render dark: globals.css sets color-scheme: dark on :root. Keep it.
- Every page keeps the BottomNav with the correct `active` key.

Engineering rules:
- Data comes from the /api routes only. Never call the database from client components.
- Money-like or probability-like fields can be null for some targets (e.g. `probability` is null for total). Branch on target instead of printing 0% for a missing value.
- Verify before reporting: `npm run build` in D:\NFL\web must pass. For UI changes, say explicitly that you did not visually check in a browser unless you did.
- Do not commit .env files or copy the database URL into any file.

Report what you changed and the build result, in a few lines.
