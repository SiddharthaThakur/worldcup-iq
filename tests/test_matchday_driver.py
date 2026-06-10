"""
Tests for the matchday prediction driver (fixtures → model predictions
ready for lock-in).
"""

from pathlib import Path

import pytest

from src.lockin.matchday import build_matchday_predictions

RAW = Path("data/raw/international_results.csv")
PARAMS = Path("models/dixon_coles_params.json")

pytestmark = pytest.mark.skipif(
    not (RAW.exists() and PARAMS.exists()),
    reason="needs downloaded data + fitted params",
)


@pytest.fixture(scope="module")
def opening_day():
    return build_matchday_predictions("2026-06-11")


def test_opening_day_has_fixtures(opening_day):
    preds, kickoffs = opening_day
    # 2 matches (MEX-RSA opener + KOR-CZE) x 2 models (champion + baseline)
    assert len(preds) == 4
    assert {p["model_name"] for p in preds} == {
        "composition_champion_v3", "elo_dixon_coles_v2"}


def test_both_models_predict_each_match(opening_day):
    preds, _ = opening_day
    for mid in ("2026-06-11_MEX_RSA", "2026-06-11_KOR_CZE"):
        models = {p["model_name"] for p in preds if p["match_id"] == mid}
        assert models == {"composition_champion_v3", "elo_dixon_coles_v2"}


def test_predictions_have_required_fields(opening_day):
    preds, _ = opening_day
    for p in preds:
        for field in ("match_id", "model_name", "prob_home_win", "prob_draw",
                      "prob_away_win", "home_xg", "away_xg"):
            assert field in p, field
        total = p["prob_home_win"] + p["prob_draw"] + p["prob_away_win"]
        assert abs(total - 1.0) < 1e-3


def test_conservative_kickoff_times(opening_day):
    # Pseudo-kickoff = midnight UTC of the match date: earlier than any
    # real kickoff, so the lock script's refusal logic is strictly safe
    _, kickoffs = opening_day
    assert kickoffs["2026-06-11_MEX_RSA"] == "2026-06-11T00:00:00+00:00"


def test_host_gets_home_advantage(opening_day):
    preds, _ = opening_day
    mex = next(p for p in preds if p["match_id"] == "2026-06-11_MEX_RSA")
    # Mexico at the Azteca: model must favor them over South Africa
    assert mex["prob_home_win"] > mex["prob_away_win"]


def test_unknown_matchday_raises():
    with pytest.raises(ValueError):
        build_matchday_predictions("2026-01-01")
