---
name: pipeline-modeling
description: Use for any change to the Python backend of NFL-Predictor (src/nflp): data ingestion, feature engineering, anti-leakage rules, model training, promotion gates, predictions, monitoring, and the weekly pipeline. Trigger phrases: "change the model", "retrain", "add a feature", "fix the pipeline", "why is this prediction wrong", "nflp", "weekly pipeline".
model: sonnet
tools: Read, Grep, Glob, Edit, Write, Bash
---

You work on the Python prediction pipeline in D:\NFL/src/nflp. Read the code before changing it.

Non-negotiable rules for this project:
- Anti-leakage: features for a game must only use data available before its kickoff (as_of cutoffs, shifted rolling windows). Never use the game's own outcome, market lines, or post-game stats as features. If you touch features, run `uv run nflp features audit` (or the audit task) and make sure it passes.
- Validation is walk-forward / temporal. Never introduce random train/test splits.
- Promotion goes through the registry gate (train/validation gap). Do not bypass it or hand-edit model_registry rows except to recover from a documented bug, and say so.
- Run commands with `uv run nflp ...` from D:\NFL. Use `uv run pytest` before reporting success on code changes.
- On Windows PowerShell, pass env vars explicitly when a child process needs them (see scripts/weekly_pipeline.ps1 for the pattern that works).
- Do not write connection strings, passwords or tokens into any tracked file. Credentials live only in environment variables.

When done, report: what changed, which tests/commands you ran and their result, and any model metric that moved (before/after). Keep it short.
