"""Contrato anti-leakage.

Este módulo es el guardián metodológico del proyecto. Un backtest con leakage
produce métricas espectaculares y pérdidas reales; todo lo demás del sistema
depende de que estos invariantes se cumplan.

**Invariantes:**

1. Toda fila de feature lleva ``as_of_ts`` = ``kickoff - offset``.
2. Las agregaciones móviles usan ``shift(1)``: solo semanas estrictamente
   anteriores, nunca el partido objetivo.
3. Las lesiones se resuelven al último parte con ``date_modified < as_of_ts``.
4. Las columnas de la *denylist* no pueden aparecer nunca en la matriz de
   features. Si aparecen, el entrenamiento se aborta.
5. Las líneas de mercado viven en otra tabla y el Feature Agent no las consulta.

La comprobación no es decorativa: ``assert_no_leakage`` se llama desde el
Modeling Agent antes de cada entrenamiento y desde las pruebas automáticas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from nflp.core.logging import get_logger

log = get_logger(__name__)


class LeakageError(RuntimeError):
    """Se detectó información no disponible antes del kickoff."""


# --------------------------------------------------------------------- denylist

# Resultados del partido: conocerlos antes del kickoff es imposible.
OUTCOME_COLUMNS: frozenset[str] = frozenset(
    {
        "result", "total", "home_score", "away_score", "overtime",
        "margin", "home_win", "actual_total", "actual_margin",
        "points_scored", "points_allowed_actual",
    }
)

# Datos de mercado: excluidos por decisión metodológica explícita. El mercado es
# el BENCHMARK del backtest, no una feature. Si se usara como feature, el modelo
# se limitaría a reaprender la línea y dejaría de ser medible si aporta valor.
MARKET_COLUMNS: frozenset[str] = frozenset(
    {
        "spread_line", "total_line", "home_moneyline", "away_moneyline",
        "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
        "implied_prob", "vegas_spread", "vegas_total", "closing_line",
        "market_prob", "line_movement",
    }
)

# Identificadores y metadatos: no son leakage, pero tampoco son features.
# Si entran al modelo, este memoriza partidos concretos en vez de aprender.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "game_id", "gsis_id", "player_id", "feature_set_version", "as_of_ts",
        "feature_hash", "kickoff_utc", "gameday", "gametime", "stadium",
        "home_qb_name", "away_qb_name", "home_coach", "away_coach", "referee",
        "player_name", "prediction_id", "model_id",
    }
)

DENYLIST: frozenset[str] = OUTCOME_COLUMNS | MARKET_COLUMNS

# Patrones que delatan una feature construida con información futura.
SUSPICIOUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^actual_", re.IGNORECASE),
    re.compile(r"_actual$", re.IGNORECASE),
    re.compile(r"^final_", re.IGNORECASE),
    re.compile(r"^post_game", re.IGNORECASE),
    re.compile(r"^outcome", re.IGNORECASE),
    re.compile(r"vegas", re.IGNORECASE),
    re.compile(r"moneyline", re.IGNORECASE),
    re.compile(r"spread_line", re.IGNORECASE),
    re.compile(r"total_line", re.IGNORECASE),
)


@dataclass
class LeakageReport:
    """Resultado de una auditoría anti-leakage."""

    denied_columns: list[str]
    suspicious_columns: list[str]
    rows_after_as_of: int
    total_rows: int
    missing_as_of: int

    @property
    def is_clean(self) -> bool:
        return (
            not self.denied_columns
            and not self.suspicious_columns
            and self.rows_after_as_of == 0
            and self.missing_as_of == 0
        )

    def describe(self) -> str:
        if self.is_clean:
            return f"Auditoría anti-leakage superada ({self.total_rows:,} filas)."
        problems: list[str] = []
        if self.denied_columns:
            problems.append(f"columnas prohibidas: {', '.join(sorted(self.denied_columns))}")
        if self.suspicious_columns:
            problems.append(f"columnas sospechosas: {', '.join(sorted(self.suspicious_columns))}")
        if self.missing_as_of:
            problems.append(f"{self.missing_as_of:,} filas sin as_of_ts")
        if self.rows_after_as_of:
            problems.append(f"{self.rows_after_as_of:,} filas con as_of_ts posterior al kickoff")
        return "Auditoría anti-leakage FALLIDA — " + "; ".join(problems)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columnas utilizables como features: todo menos identificadores y denylist."""
    excluded = DENYLIST | NON_FEATURE_COLUMNS
    return [column for column in frame.columns if column not in excluded]


def audit_features(
    frame: pd.DataFrame, *, kickoff: pd.Series | None = None
) -> LeakageReport:
    """Audita una matriz de features y devuelve el informe.

    Args:
        frame: matriz a auditar.
        kickoff: serie opcional de timestamps de kickoff, para verificar que
            ``as_of_ts`` es anterior en todas las filas.
    """
    columns = set(frame.columns)

    denied = sorted(columns & DENYLIST)
    suspicious = sorted(
        column
        for column in columns - DENYLIST - NON_FEATURE_COLUMNS
        if any(pattern.search(column) for pattern in SUSPICIOUS_PATTERNS)
    )

    missing_as_of = 0
    rows_after = 0
    if "as_of_ts" in frame.columns:
        as_of = pd.to_datetime(frame["as_of_ts"], errors="coerce")
        missing_as_of = int(as_of.isna().sum())
        if kickoff is not None:
            kickoff_ts = pd.to_datetime(kickoff, errors="coerce")
            rows_after = int((as_of > kickoff_ts).sum())
    else:
        missing_as_of = len(frame)

    return LeakageReport(
        denied_columns=denied,
        suspicious_columns=suspicious,
        rows_after_as_of=rows_after,
        total_rows=len(frame),
        missing_as_of=missing_as_of,
    )


def assert_no_leakage(frame: pd.DataFrame, *, kickoff: pd.Series | None = None) -> None:
    """Aborta si la matriz de features contiene leakage.

    Se llama antes de cada entrenamiento. Fallar aquí es preferible por mucho a
    entrenar un modelo con métricas infladas.
    """
    report = audit_features(frame, kickoff=kickoff)
    if not report.is_clean:
        raise LeakageError(report.describe())
    log.debug(report.describe())


def shifted_rolling(
    frame: pd.DataFrame,
    group_keys: list[str],
    value_column: str,
    window: int,
    *,
    min_periods: int = 1,
) -> pd.Series:
    """Media móvil que **excluye la fila actual**.

    Es la primitiva que garantiza el invariante 2. El ``shift(1)`` dentro de cada
    grupo es lo que impide que el partido objetivo entre en su propia media.

    El DataFrame debe venir ordenado cronológicamente.
    """
    grouped = frame.groupby(group_keys, observed=True, sort=False)[value_column]
    return grouped.transform(
        lambda serie: serie.shift(1).rolling(window=window, min_periods=min_periods).mean()
    )


def shifted_ewm(
    frame: pd.DataFrame,
    group_keys: list[str],
    value_column: str,
    halflife: float,
) -> pd.Series:
    """Media exponencial que excluye la fila actual (forma reciente ponderada)."""
    grouped = frame.groupby(group_keys, observed=True, sort=False)[value_column]
    return grouped.transform(
        lambda serie: serie.shift(1).ewm(halflife=halflife, min_periods=1).mean()
    )


def shifted_expanding(
    frame: pd.DataFrame,
    group_keys: list[str],
    value_column: str,
    *,
    min_periods: int = 1,
) -> pd.Series:
    """Media acumulada que excluye la fila actual."""
    grouped = frame.groupby(group_keys, observed=True, sort=False)[value_column]
    return grouped.transform(
        lambda serie: serie.shift(1).expanding(min_periods=min_periods).mean()
    )
