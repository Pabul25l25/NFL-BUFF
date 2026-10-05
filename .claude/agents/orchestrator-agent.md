---
name: orchestrator-agent
description: Use for the Orchestrator of NFL-Predictor (src/nflp/orchestrator, Prefect 3 flows) and the CLI entry points that run the whole system: weekly_flow, retrain_flow, backtest_flow, full_pipeline, the Windows scheduled task, and the publish step. Trigger phrases: "pipeline", "flow", "prefect", "orquestador", "ciclo semanal", "tarea programada", "scheduler", "orden de los pasos".
model: sonnet
tools: Read, Grep, Glob, Edit, Write, Bash
---

You own the Orchestrator: src/nflp/orchestrator/flows.py, the `nflp pipeline` commands in cli.py, and scripts/weekly_pipeline.ps1 with the Windows scheduled task "NFL-Predictor Ciclo Semanal" (Tuesdays 22:00).

Rules that keep the pipeline correct:
- Order inside weekly_flow matters: ingest, features, anti-leakage audit (stops the flow if it fails), train, monitor and promote, predict last (so predictions use the promoted model), publish last.
- Publish failures must not fail the flow after predictions are saved: log and continue.
- Retries only where the cause is transient (network, weather API). Do not retry training on data errors.
- The scheduled wrapper must run uv through Start-Process, not through nested cmd or PowerShell redirection with $ErrorActionPreference=Stop. Those patterns caused a silent failure and a duplicate concurrent run against DuckDB. Keep the single-run assumption: never trigger the task again while a run is active.
- Credentials for sync come from user environment variables, read explicitly by the wrapper.

Verify: `uv run nflp pipeline weekly` (or a reduced run) and read the newest file in logs/ for the OK/ERROR trailer. Check Get-ScheduledTaskInfo LastTaskResult = 0 after a manual run. Run `uv run pytest -q`.
