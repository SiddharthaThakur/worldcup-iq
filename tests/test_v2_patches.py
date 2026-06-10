"""Tests for the v2 patches: fitted params, simulator, odds, lock-in."""

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import (
    DixonColesParams, predict_match, scoreline_matrix, outcome_probs,
)
from src.models.fit_params import build_training_rows, fit_goal_model, fit_rho
from src.simulation.tournament import (
    SimulationConfig, play_group, rank_third_place, resolve_knockout,
    run_simulation, GroupResult,
)
from src.data.odds_loader import (
    devig_proportional, devig_power, closing_line_value,
)


@pytest.fixture
def fitted_params(tmp_path):
    """Synthetic fitted params for testing (NOT for production use)."""
    return DixonColesParams(
        intercept=0.30, elo_coef=0.18, home_adv=0.25, rho=-0.10,
        fitted_on="synthetic-test", n_matches=1000,
    )


@pytest.fixture
def synthetic_results():
    """Generate synthetic international results where a known Elo gap
    drives goals — lets us check the fitter recovers sane parameters."""
    rng = np.random.default_rng(0)
    teams = [f"T{i:02d}" for i in range(20)]
    true_strength = {t: 1500 + (i - 10) * 30 for i, t in enumerate(teams)}
    rows = []
    dates = pd.date_range("2015-01-01", periods=2000, freq="D")
    for k in range(2000):
        a, b = rng.choice(teams, size=2, replace=False)
        diff = (true_strength[a] - true_strength[b]) / 100.0
        lam_a = np.exp(0.3 + 0.15 * diff)
        lam_b = np.exp(0.3 - 0.15 * diff)
        rows.append({
            "date": dates[k], "home_code": a, "away_code": b,
            "home_score": rng.poisson(lam_a), "away_score": rng.poisson(lam_b),
            "tournament": "Friendly", "neutral": True,
        })
    return pd.DataFrame(rows)


class TestFitting:
    def test_no_lookahead_in_training_rows(self, synthetic_results):
        """First match must use default 1500 ratings for both teams."""
        train = build_training_rows(synthetic_results.head(50))
        assert train.iloc[0]["home_elo_pre"] == 1500.0
        assert train.iloc[0]["away_elo_pre"] == 1500.0

    def test_goal_model_recovers_positive_elo_coef(self, synthetic_results):
        train = build_training_rows(synthetic_results)
        intercept, elo_coef, home_adv = fit_goal_model(train)
        assert elo_coef > 0, "Stronger teams must be fitted to score more"
        assert 0.0 < np.exp(intercept) < 3.0, "Avg goals must be plausible"

    def test_rho_in_bounds(self, synthetic_results):
        train = build_training_rows(synthetic_results)
        intercept, elo_coef, home_adv = fit_goal_model(train)
        rho = fit_rho(train, intercept, elo_coef, home_adv)
        assert -0.5 <= rho <= 0.5

    def test_predict_refuses_missing_params(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="refuses"):
            DixonColesParams.load(tmp_path / "nonexistent.json")

    def test_params_roundtrip(self, fitted_params, tmp_path):
        path = tmp_path / "params.json"
        fitted_params.save(path)
        loaded = DixonColesParams.load(path)
        assert loaded.elo_coef == fitted_params.elo_coef


class TestDixonColesV2:
    def test_probabilities_sum_to_one(self, fitted_params):
        pred = predict_match("ARG", "BRA", 1800, 1700, fitted_params)
        total = pred.prob_home_win + pred.prob_draw + pred.prob_away_win
        assert abs(total - 1.0) < 1e-9

    def test_stronger_team_favored(self, fitted_params):
        pred = predict_match("ARG", "PAN", 1900, 1500, fitted_params)
        assert pred.prob_home_win > 0.5
        assert pred.prob_home_win > pred.prob_away_win

    def test_symmetric_when_equal(self, fitted_params):
        pred = predict_match("A", "B", 1600, 1600, fitted_params)
        assert abs(pred.prob_home_win - pred.prob_away_win) < 1e-9

    def test_home_advantage_applies_only_when_not_neutral(self, fitted_params):
        neutral = predict_match("USA", "X", 1600, 1600, fitted_params, neutral=True)
        home = predict_match("USA", "X", 1600, 1600, fitted_params, neutral=False)
        assert home.prob_home_win > neutral.prob_home_win


class TestSimulator:
    @pytest.fixture
    def small_world(self, fitted_params):
        groups = {
            g: [f"{g}{k}" for k in range(4)] for g in "ABCDEFGHIJKL"
        }
        rng = np.random.default_rng(1)
        strengths = {t: float(rng.normal(1600, 120))
                     for ts in groups.values() for t in ts}
        return groups, strengths, fitted_params

    def test_group_play_awards_correct_total_points(self, fitted_params):
        rng = np.random.default_rng(2)
        teams = ["W", "X", "Y", "Z"]
        strengths = {t: 1600.0 for t in teams}
        gr = play_group(teams, strengths, set(), fitted_params, rng)
        total_points = sum(r["points"] for r in gr.standings)
        # 6 matches: each gives 3 (decisive) or 2 (draw) points
        assert 12 <= total_points <= 18
        # Goal differences must sum to zero
        assert sum(r["gd"] for r in gr.standings) == 0

    def test_third_place_ranking_selects_eight(self, fitted_params):
        rng = np.random.default_rng(3)
        results = []
        for g in "ABCDEFGHIJKL":
            teams = [f"{g}{k}" for k in range(4)]
            gr = play_group(teams, {t: 1600.0 for t in teams}, set(),
                            fitted_params, rng)
            gr.group = g
            results.append(gr)
        thirds = rank_third_place(results, rng)
        assert len(thirds) == 8
        assert len(set(thirds)) == 8

    def test_knockout_always_produces_winner(self, fitted_params):
        rng = np.random.default_rng(4)
        config = SimulationConfig(seed=4)
        outcomes = [resolve_knockout(1600, 1600, fitted_params, rng, config)
                    for _ in range(200)]
        assert all(isinstance(o, bool) for o in outcomes)
        # Equal teams should split roughly evenly
        rate = np.mean(outcomes)
        assert 0.35 < rate < 0.65

    def test_stronger_team_advances_more_often(self, fitted_params):
        rng = np.random.default_rng(5)
        config = SimulationConfig(seed=5)
        wins = sum(resolve_knockout(1850, 1450, fitted_params, rng, config)
                   for _ in range(500))
        assert wins / 500 > 0.65

    def test_full_simulation_probabilities_valid(self, small_world):
        groups, strengths, params = small_world
        df = run_simulation(groups, strengths, host_teams=set(),
                            params=params,
                            config=SimulationConfig(n_sims=200, seed=42))
        assert len(df) == 48
        assert abs(df["p_champion"].sum() - 1.0) < 0.01
        # Monotone: P(champion) <= P(final) <= ... <= P(r32)
        for _, r in df.iterrows():
            assert r["p_champion"] <= r["p_final"] + 1e-9
            assert r["p_final"] <= r["p_sf"] + 1e-9
            assert r["p_sf"] <= r["p_qf"] + 1e-9

    def test_simulation_reproducible_with_seed(self, small_world):
        groups, strengths, params = small_world
        cfg = SimulationConfig(n_sims=100, seed=7)
        df1 = run_simulation(groups, strengths, set(), params, cfg)
        df2 = run_simulation(groups, strengths, set(), params, cfg)
        pd.testing.assert_frame_equal(df1, df2)

    def test_strongest_team_has_highest_champion_prob(self, small_world):
        groups, strengths, params = small_world
        # Make one team overwhelmingly strong
        strengths = dict(strengths)
        strengths["A0"] = 2200.0
        df = run_simulation(groups, strengths, set(), params,
                            SimulationConfig(n_sims=300, seed=8))
        assert df.iloc[0]["team"] == "A0"


class TestOdds:
    def test_devig_proportional_sums_to_one(self):
        ph, pd_, pa, ov = devig_proportional(2.10, 3.40, 3.60)
        assert abs(ph + pd_ + pa - 1.0) < 1e-12
        assert ov > 0

    def test_devig_power_sums_to_one(self):
        ph, pd_, pa, ov = devig_power(2.10, 3.40, 3.60)
        assert abs(ph + pd_ + pa - 1.0) < 1e-9

    def test_power_shades_longshots_more(self):
        """Power method should assign relatively less probability to the
        longshot than proportional — that's its entire purpose."""
        _, _, pa_prop, _ = devig_proportional(1.20, 6.50, 15.0)
        _, _, pa_pow, _ = devig_power(1.20, 6.50, 15.0)
        assert pa_pow < pa_prop

    def test_clv_positive_when_market_moves_toward_model(self):
        # Model said 0.60, market opened 0.50, closed 0.55 → moved toward us
        assert closing_line_value(0.60, 0.50, 0.55) > 0

    def test_clv_negative_when_market_moves_away(self):
        assert closing_line_value(0.60, 0.50, 0.45) < 0


class TestLockIn:
    def test_refuses_past_kickoffs(self, tmp_path, monkeypatch):
        from src.lockin import lock_predictions as lp
        monkeypatch.setattr(lp, "PREDICTIONS_DIR", tmp_path)
        with pytest.raises(ValueError, match="REFUSED"):
            lp.lock_predictions(
                predictions=[{"match_id": "m1", "model_name": "elo",
                              "prob_home_win": 0.5, "prob_draw": 0.3,
                              "prob_away_win": 0.2}],
                matchday="2020-01-01",
                kickoff_times={"m1": "2020-01-01T15:00:00+00:00"},
            )
