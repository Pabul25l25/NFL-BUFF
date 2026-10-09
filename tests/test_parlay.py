"""Modelo de correlación de parlays del mismo partido (cópula gaussiana).

* Proyección a la correlación semidefinida positiva más cercana.
* Contracción ρ·n/(n+k) y elección de k por validación temporal.
* PIT aleatorizado: uniforme con el modelo correcto, también en conteos.
* Sin fuga: las correlaciones para evaluar la temporada S solo usan S' < S.
* Roles por uso previo y signo del margen para el equipo visitante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest, norm, poisson

from nflp.agents.modeling.distributions import QUANTILE_LEVELS
from nflp.agents.modeling.parlay import (
    GAME_ROLE,
    KIND_IDENTITY,
    CorrelationModel,
    assign_roles,
    build_correlation_matrix,
    canonical_key,
    choose_shrinkage,
    ece_score,
    fit_before,
    fit_correlations,
    grid_cdf,
    nearest_psd_correlation,
    normal_scores,
    pair_stats,
    randomized_pit,
    round_psd,
    shrink,
    simulate_parlays,
)

_QCOLS = [f"q{i:02d}" for i in range(len(QUANTILE_LEVELS))]


# ------------------------------------------------------------------ PSD


def test_nearest_psd_fixes_indefinite_matrix() -> None:
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(bad).min() < 0
    fixed = nearest_psd_correlation(bad)
    assert np.allclose(fixed, fixed.T)
    assert np.allclose(np.diag(fixed), 1.0)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-10
    assert np.all(np.abs(fixed) <= 1.0 + 1e-12)
    # Conserva los signos de la estructura original.
    assert fixed[0, 1] > 0 and fixed[1, 2] > 0 and fixed[0, 2] < 0


def test_nearest_psd_keeps_valid_correlation() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 40))
    corr = np.corrcoef(a)
    assert np.allclose(nearest_psd_correlation(corr), corr, atol=1e-6)


def test_round_psd_stays_psd() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(60, 3))
    corr = np.corrcoef(base @ rng.normal(size=(3, 60)) + 1e-3 * rng.normal(size=(60, 60)))
    near_singular = nearest_psd_correlation(corr)
    rounded = round_psd(near_singular)
    assert np.allclose(np.diag(rounded), 1.0)
    assert np.linalg.eigvalsh(rounded).min() >= 0.0
    assert np.allclose(rounded, np.round(rounded, 3))


# -------------------------------------------------------------- shrinkage


def test_shrink_formula() -> None:
    rho = np.array([0.5, 0.5, -0.4])
    n = np.array([100.0, 900.0, 0.0])
    assert np.allclose(shrink(rho, n, 100), [0.25, 0.45, 0.0])
    assert np.allclose(shrink(rho, n, 0), rho)


def _synthetic_stats(rho_by_key: dict[str, float], seasons: list[int], n: int, seed: int) -> pd.DataFrame:
    """Estadísticos por (clave, temporada) de pares N(0, ρ)."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, rho in rho_by_key.items():
        for season in seasons:
            rho_s = rho(season) if callable(rho) else rho
            cov = [[1.0, rho_s], [rho_s, 1.0]]
            xy = rng.multivariate_normal([0, 0], cov, size=n)
            x, y = xy[:, 0], xy[:, 1]
            rows.append({
                "relation": "teammate", "role_a": key, "target_a": "t", "role_b": "WR1",
                "target_b": "u", "season": season, "n": n, "sx": x.sum(), "sy": y.sum(),
                "sxx": (x * x).sum(), "syy": (y * y).sum(), "sxy": (x * y).sum(),
            })
    return pd.DataFrame(rows)


def test_choose_shrinkage_prefers_strong_shrink_for_noise() -> None:
    # 40 claves sin correlación real y pocas observaciones: conviene contraer mucho.
    stats = _synthetic_stats({f"R{i}": 0.0 for i in range(40)}, [2019, 2020, 2021, 2022], n=30, seed=2)
    k, table = choose_shrinkage(stats, [2019, 2020, 2021, 2022])
    assert k >= 100
    assert set(table.columns) == {"k", "loglik"}


def test_choose_shrinkage_prefers_little_shrink_for_stable_signal() -> None:
    stats = _synthetic_stats({f"R{i}": 0.6 for i in range(10)}, [2019, 2020, 2021, 2022], n=400, seed=3)
    k, _ = choose_shrinkage(stats, [2019, 2020, 2021, 2022])
    assert k <= 50


# ------------------------------------------------------------ PIT / z


def test_randomized_pit_poisson_is_uniform() -> None:
    rng = np.random.default_rng(4)
    lam = rng.uniform(0.05, 1.5, size=20000)  # muchos ceros, como los TDs
    y = rng.poisson(lam).astype(float)
    u = randomized_pit(np.full(y.size, "poisson"), y, rng, lam=lam)
    assert kstest(u, "uniform").pvalue > 0.01
    z = normal_scores(u)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03
    # Un PIT no aleatorizado (F(y)) estaría claramente sesgado hacia arriba.
    assert poisson.cdf(y, lam).mean() > 0.6


def test_randomized_pit_grid_with_floor_is_uniform() -> None:
    """Yardas enteras con masa en 0 y rejilla p05..p95 de la distribución verdadera."""
    rng = np.random.default_rng(5)
    n = 20000
    scale = rng.uniform(5, 40, size=n)
    # Verdad: max(0, round(N(μ, σ))) — masa en 0 y valores enteros.
    mu = 1.2 * scale
    taus = np.asarray(QUANTILE_LEVELS)
    # Cuantiles de la variable discreta censurada vía la latente continua + 0,5.
    grid = np.maximum(mu[:, None] + scale[:, None] * norm.ppf(taus)[None, :], 0.0)
    y = np.maximum(np.round(mu + scale * rng.normal(size=n)), 0.0)
    params = {"grid": grid, "lower_bound": np.zeros(n)}
    u = randomized_pit(np.full(n, "grid"), y, rng, **params)
    # La aproximación por rejilla no es exacta; basta con una uniformidad razonable.
    hist, _ = np.histogram(u, bins=10, range=(0, 1))
    assert np.all(np.abs(hist / n - 0.1) < 0.02)


def test_grid_cdf_interpolates_and_extrapolates() -> None:
    taus = np.asarray(QUANTILE_LEVELS)
    grid = norm.ppf(taus)[None, :] * 10 + 50  # N(50, 10)
    x = np.array([50.0, 60.0, 30.0, 75.0, -1.0])
    cdf = grid_cdf(x, np.repeat(grid, len(x), axis=0), lower_bound=0.0)
    assert cdf[0] == pytest.approx(0.5, abs=1e-9)
    assert cdf[1] == pytest.approx(norm.cdf(1.0), abs=0.02)
    assert cdf[2] == pytest.approx(norm.cdf(-2.0), abs=0.005)   # cola extrapolada
    assert cdf[3] == pytest.approx(norm.cdf(2.5), abs=0.005)
    assert cdf[4] == 0.0                                        # bajo el suelo


# ----------------------------------------------------------- sin fuga


def test_correlations_for_test_season_use_only_prior_seasons() -> None:
    # Correlación 0 hasta 2022 y 0,9 en 2023-2024: el modelo para evaluar 2023
    # no puede ver la señal de 2023 ni de 2024.
    stats = _synthetic_stats(
        {"QB": lambda s: 0.9 if s >= 2023 else 0.0}, [2019, 2020, 2021, 2022, 2023, 2024], n=500, seed=6
    )
    model = fit_before(stats, 2023)
    assert model.seasons == [2019, 2020, 2021, 2022]
    assert abs(model.rho("teammate", "QB", "t", "WR1", "u")) < 0.1

    later = fit_before(stats, 2025)
    assert max(later.seasons) == 2024
    assert later.rho("teammate", "QB", "t", "WR1", "u") > 0.3


def test_choose_shrinkage_only_reads_given_seasons() -> None:
    stats = _synthetic_stats({f"R{i}": 0.0 for i in range(20)}, [2019, 2020, 2021], n=50, seed=7)
    poisoned = stats.copy()
    poisoned = pd.concat([poisoned, poisoned.assign(season=2030, sxy=1e9)], ignore_index=True)
    k1, t1 = choose_shrinkage(stats, [2019, 2020, 2021])
    k2, t2 = choose_shrinkage(poisoned, [2019, 2020, 2021])
    assert k1 == k2
    assert np.allclose(t1["loglik"], t2["loglik"])


# ------------------------------------------------- roles y ensamblado


def test_assign_roles_by_pre_game_usage() -> None:
    rows = []
    for gsis, pos, share in [("w1", "WR", 0.10), ("w2", "WR", 0.28), ("w3", "WR", 0.20), ("w4", "WR", 0.05),
                             ("t1", "TE", 0.15), ("t2", "TE", 0.02)]:
        rows.append({"game_id": "g", "gsis_id": gsis, "team": "AAA", "position": pos,
                     "target_share_ewm": share, "snap_pct_r5": 0.8, "injury_status_code": 0})
    rows.append({"game_id": "g", "gsis_id": "r1", "team": "AAA", "position": "RB",
                 "carries_ewm": 15.0, "targets_ewm": 3.0, "snap_pct_r5": 0.6, "injury_status_code": 0})
    rows.append({"game_id": "g", "gsis_id": "r2", "team": "AAA", "position": "RB",
                 "carries_ewm": 20.0, "targets_ewm": 1.0, "snap_pct_r5": 0.5, "injury_status_code": 3})  # baja
    rows.append({"game_id": "g", "gsis_id": "q1", "team": "AAA", "position": "QB",
                 "passing_attempts_ewm": 33.0, "snap_pct_r5": 1.0, "injury_status_code": 0})
    roles = assign_roles(pd.DataFrame(rows)).set_index("gsis_id")["role"]
    assert roles["w2"] == "WR1" and roles["w3"] == "WR2" and roles["w1"] == "WR3"
    assert roles["w4"] == "OTH" and roles["t1"] == "TE1" and roles["t2"] == "OTH"
    assert roles["r1"] == "RB1" and roles["r2"] == "OTH"
    assert roles["q1"] == "QB"


def _model(entries: dict[tuple[str, ...], float]) -> CorrelationModel:
    table = pd.DataFrame(
        [{"relation": k[0], "role_a": k[1], "target_a": k[2], "role_b": k[3], "target_b": k[4],
          "n": 1000, "rho_raw": v, "rho": v} for k, v in entries.items()]
    )
    # Claves canónicas, como las produce pair_stats.
    canon = [canonical_key(*row) for row in table[["relation", "role_a", "target_a", "role_b", "target_b"]]
             .itertuples(index=False)]
    table[["relation", "role_a", "target_a", "role_b", "target_b"]] = pd.DataFrame(canon)
    return CorrelationModel(table=table, k=0.0, seasons=[2020])


def test_matrix_signs_margin_for_away_player_and_is_psd() -> None:
    model = _model({
        ("game", "RB1", "rushing_yards", GAME_ROLE, "margin"): 0.3,
        ("teammate", "QB", "passing_yards", "WR1", "receiving_yards"): 0.4,
    })
    legs = pd.DataFrame({
        "leg": ["p:h_rb:rushing_yards", "p:a_rb:rushing_yards", "p:h_qb:passing_yards",
                "p:h_wr:receiving_yards", "g:margin", "g:total"],
        "gsis_id": ["h_rb", "a_rb", "h_qb", "h_wr", None, None],
        "team": ["HOM", "AWY", "HOM", "HOM", None, None],
        "is_home": [True, False, True, True, True, True],
        "role": ["RB1", "RB1", "QB", "WR1", GAME_ROLE, GAME_ROLE],
        "target": ["rushing_yards", "rushing_yards", "passing_yards", "receiving_yards", "margin", "total"],
    })
    matrix = build_correlation_matrix(legs, model, psd=False)
    assert matrix[0, 4] == pytest.approx(0.3)      # RB1 local × margen local
    assert matrix[1, 4] == pytest.approx(-0.3)     # RB1 visitante × margen local
    assert matrix[2, 3] == pytest.approx(0.4)      # orden de la clave indiferente
    assert matrix[3, 2] == pytest.approx(0.4)
    assert matrix[0, 1] == 0.0                     # clave no estimada → 0
    psd = build_correlation_matrix(legs, model)
    assert np.linalg.eigvalsh(psd).min() >= -1e-10
    identity = CorrelationModel(table=model.table, k=0.0, seasons=[2020], kind=KIND_IDENTITY)
    assert np.allclose(build_correlation_matrix(legs, identity), np.eye(len(legs)))


def test_pair_stats_recover_correlation_and_sign() -> None:
    """Datos simulados con cópula conocida: pair_stats + fit recuperan ρ y el signo del margen."""
    rng = np.random.default_rng(8)
    rows, games = [], []
    for g in range(1500):
        margin = rng.normal()
        qb = rng.normal()
        wr = 0.5 * qb + np.sqrt(1 - 0.25) * rng.normal()
        rb_home = 0.4 * margin + np.sqrt(1 - 0.16) * rng.normal()
        rb_away = -0.4 * margin + np.sqrt(1 - 0.16) * rng.normal()
        gid = f"g{g}"
        season = 2020 + g % 3
        for gsis, team, home, role, target, z in [
            ("qb", "H", True, "QB", "passing_yards", qb), ("wr", "H", True, "WR1", "receiving_yards", wr),
            ("rbh", "H", True, "RB1", "rushing_yards", rb_home),
            ("rba", "A", False, "RB1", "rushing_yards", rb_away),
        ]:
            rows.append({"game_id": gid, "season": season, "leg": f"p:{gsis}{g}:{target}",
                         "gsis_id": f"{gsis}{g}", "team": team, "is_home": home, "role": role,
                         "target": target, "z": z})
        games.append({"game_id": gid, "season": season, "target": "margin", "z": margin})
        games.append({"game_id": gid, "season": season, "target": "total", "z": rng.normal()})
    stats = pair_stats(pd.DataFrame(rows), pd.DataFrame(games))
    model = fit_correlations(stats, [2020, 2021, 2022], k=0)
    assert model.rho("teammate", "WR1", "receiving_yards", "QB", "passing_yards") == pytest.approx(0.5, abs=0.06)
    assert model.rho("game", "RB1", "rushing_yards", GAME_ROLE, "margin") == pytest.approx(0.4, abs=0.05)
    assert abs(model.rho("game", GAME_ROLE, "margin", GAME_ROLE, "total")) < 0.08


def test_simulated_parlays_copula_beats_independence_on_correlated_data() -> None:
    """Con dependencia real la cópula calibra mejor que el producto de marginales."""
    rng = np.random.default_rng(9)
    rho = 0.6
    legs = []
    for g in range(300):
        z1 = rng.normal()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal()
        z3 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal()
        for gsis, role, z in (("a", "QB", z1), ("b", "WR1", z2), ("c", "WR2", z3), ("d", "TE1", rng.normal())):
            legs.append({"game_id": f"g{g}", "season": 2024, "leg": f"p:{gsis}:x", "gsis_id": gsis,
                         "team": "H", "is_home": True, "role": role, "target": "x",
                         "kind": "normal", "mu": 0.0, "sigma": 1.0, "y": z, "line": 0.0})
    frame = pd.DataFrame(legs)
    frame["p_over"] = 0.5
    frame["t"] = 0.0
    frame["over_hit"] = frame["y"] > 0
    model = _model({
        ("teammate", "QB", "x", "WR1", "x"): rho, ("teammate", "QB", "x", "WR2", "x"): rho,
        ("teammate", "WR1", "x", "WR2", "x"): rho**2,
    })
    parlays = simulate_parlays(frame, model, n_per_game=10, n_sims=4000, rng=np.random.default_rng(10))
    y = parlays["hit"].to_numpy(float)
    assert ece_score(parlays["p_copula"], y) < ece_score(parlays["p_indep"], y)
