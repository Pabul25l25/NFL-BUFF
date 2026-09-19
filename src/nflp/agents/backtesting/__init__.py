"""Backtesting Agent — validación walk-forward y comparación contra baselines."""

from nflp.agents.backtesting.metrics import (
    BREAKEVEN_ATS,
    CalibrationCurve,
    ats_win_pct,
    calibration_curve,
    classification_metrics,
    crps_gaussian,
    expected_calibration_error,
    regression_metrics,
)

__all__ = [
    "BREAKEVEN_ATS",
    "CalibrationCurve",
    "ats_win_pct",
    "calibration_curve",
    "classification_metrics",
    "crps_gaussian",
    "expected_calibration_error",
    "regression_metrics",
]
