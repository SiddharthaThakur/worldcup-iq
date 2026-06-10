"""
Tests for the leak-free tournament backtest harness.

The backtest must:
  1. Fit Elo + Dixon-Coles params ONLY on matches before the tournament starts
  2. Produce one PredictionRecord per tournament match
  3. Favor the historically stronger team
  4. Output valid probability distributions
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import backtest_world_cup


def _synthetic_results() -> pd.DataFrame:
    """Build a small world: AAA is strong, DDD is weak, BBB/CCC middling.

    ~96 pre-tournament matches in 2016-2017, then a 4-match
    'FIFA World Cup' in June 2018.
    """
    teams = ["AAA", "BBB", "CCC", "DDD"]
    strength = {"AAA": 3, "BBB": 2, "CCC": 1, "DDD": 0}
    rows = []
    rng_dates = pd.date_range("2016-01-15", "2017-12-15", periods=96)
    i = 0
    for date in rng_dates:
        home = teams[i % 4]
        away = teams[(i + 1 + i // 4) % 4]
        if home == away:
            away = teams[(i + 2) % 4]
        # Deterministic scores from strength gap: stronger side scores more
        gap = strength[home] - strength[away]
        home_score = max(0, 1 + gap)
        away_score = max(0, 1 - gap)
        rows.append({
            "date": date,
            "home_code": home,
            "away_code": away,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": "Friendly",
            "neutral": False,
            "result": "H" if home_score > away_score else ("D" if home_score == away_score else "A"),
            "match_id": f"{date:%Y-%m-%d}_{home}_{away}",
        })
        i += 1

    wc = [
        ("2018-06-14", "AAA", "DDD", 2, 0),
        ("2018-06-15", "BBB", "CCC", 1, 1),
        ("2018-06-20", "AAA", "BBB", 1, 0),
        ("2018-06-21", "CCC", "DDD", 2, 1),
    ]
    for date, home, away, hs, as_ in wc:
        rows.append({
            "date": pd.Timestamp(date),
            "home_code": home,
            "away_code": away,
            "home_score": hs,
            "away_score": as_,
            "tournament": "FIFA World Cup",
            "neutral": True,
            "result": "H" if hs > as_ else ("D" if hs == as_ else "A"),
            "match_id": f"{date}_{home}_{away}",
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def backtest_result():
    return backtest_world_cup(_synthetic_results(), wc_year=2018)


def test_one_record_per_tournament_match(backtest_result):
    assert len(backtest_result.records) == 4


def test_training_excludes_tournament_matches(backtest_result):
    # 96 pre-tournament matches; the 4 WC matches must NOT be in training
    assert backtest_result.n_train_matches == 96
    assert backtest_result.params.n_matches == 96


def test_probabilities_are_valid_distributions(backtest_result):
    for rec in backtest_result.records:
        probs = np.array([rec.prob_home, rec.prob_draw, rec.prob_away])
        assert np.all(probs >= 0)
        assert abs(probs.sum() - 1.0) < 1e-6


def test_strong_team_is_favored(backtest_result):
    # First WC match is AAA (strong) vs DDD (weak)
    rec = backtest_result.records[0]
    assert rec.prob_home > rec.prob_away
    assert rec.prob_home > 0.5


def test_actual_results_recorded(backtest_result):
    assert [r.actual_result for r in backtest_result.records] == ["H", "D", "H", "H"]


def test_scorecard_present(backtest_result):
    sc = backtest_result.scorecard
    assert sc["n_matches"] == 4
    assert 0.0 <= sc["brier_score"] <= 0.667 * 1.01


def test_fitted_params_not_saved_to_disk(tmp_path, monkeypatch):
    """Backtest fitting must never overwrite the live params file."""
    from src.models import dixon_coles
    sentinel = tmp_path / "dixon_coles_params.json"
    monkeypatch.setattr(dixon_coles, "PARAMS_PATH", sentinel)
    backtest_world_cup(_synthetic_results(), wc_year=2018)
    assert not sentinel.exists()
