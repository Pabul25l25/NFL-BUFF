"""Orden de las tareas del orquestador y política de fallos del paso de parlay.

Las tareas pesadas se sustituyen por funciones que anotan su nombre; los flows
se ejecutan con ``.fn`` para no levantar el motor de Prefect.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from nflp.orchestrator import flows

_REAL_PARLAY_FN = flows.predict_parlay_task.fn


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    log: list[str] = []
    monkeypatch.setattr(flows, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(flows, "get_run_logger", lambda: logging.getLogger("test"))

    def fake(name: str, result: Any = 1):
        def _inner(*args: Any, **kwargs: Any) -> Any:
            log.append(name if not args else f"{name}:{args[0]}")
            return result

        return _inner

    class _Submitted:
        def result(self) -> None:
            return None

    for attr in (
        "ingest_data_task", "ingest_stats_task",
    ):
        task = fake(attr)
        monkeypatch.setattr(
            flows, attr,
            type("T", (), {"submit": staticmethod(lambda *a, _t=task, **k: (_t(*a, **k), _Submitted())[1]),
                           "__call__": staticmethod(task)})(),
        )
    for attr in (
        "ingest_weather_task", "build_game_features_task", "build_player_features_task",
        "audit_features_task", "train_model_task", "train_props_task", "monitor_task",
        "predict_task", "predict_props_task", "predict_parlay_task", "train_parlay_task",
        "publish_task",
    ):
        monkeypatch.setattr(flows, attr, fake(attr))
    return log


def _first(log: list[str], prefix: str) -> int:
    return next(i for i, c in enumerate(log) if c.startswith(prefix))


def test_weekly_order_props_parlay_publish(calls: list[str]) -> None:
    flows.weekly_flow.fn()
    assert calls.index("audit_features_task") < _first(calls, "train_model_task")
    assert _first(calls, "monitor_task") < _first(calls, "predict_task")
    last_props = max(i for i, c in enumerate(calls) if c.startswith("predict_props_task"))
    parlay = _first(calls, "predict_parlay_task")
    assert last_props < parlay < calls.index("publish_task")
    assert calls[-1] == "publish_task"


def test_full_pipeline_parlay_after_props(calls: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flows, "backtest_task", lambda *a, **k: {})
    flows.full_pipeline.fn(seasons=[2024])
    last_props = max(i for i, c in enumerate(calls) if c.startswith("predict_props_task"))
    assert _first(calls, "train_parlay_task") < _first(calls, "predict_parlay_task")
    assert last_props < _first(calls, "predict_parlay_task")


def test_retrain_refits_parlay_after_props(calls: list[str]) -> None:
    flows.retrain_flow.fn()
    last_props = max(i for i, c in enumerate(calls) if c.startswith("train_props_task"))
    assert last_props < _first(calls, "train_parlay_task")


def test_parlay_failure_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    import nflp.agents.modeling.parlay as parlay

    class Boom:
        def predict(self, **kwargs: Any) -> None:
            raise RuntimeError("sin modelo")

    monkeypatch.setattr(parlay, "ParlayAgent", Boom)
    monkeypatch.setattr(flows, "get_run_logger", lambda: logging.getLogger("test"))
    out = _REAL_PARLAY_FN(2025, 1)
    assert out["skipped"] is True and "sin modelo" in out["error"]


def test_parlay_failure_does_not_block_publish(calls: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import nflp.agents.modeling.parlay as parlay

    class Boom:
        def predict(self, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(parlay, "ParlayAgent", Boom)
    monkeypatch.setattr(flows, "predict_parlay_task", _REAL_PARLAY_FN)
    result = flows.weekly_flow.fn()
    assert calls[-1] == "publish_task"
    assert result["predicciones"]["parlay"]["skipped"] is True
