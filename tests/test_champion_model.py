"""
Tests for the champion model: per-team blended (Elo + player) ratings on
the Elo scale, consumed identically to raw Elo by predictions/simulation.
"""

import pandas as pd

from src.models.champion_model import CHAMPION_LAMBDA, build_champion_ratings
from src.models.composition_model import StrengthBridge


class _FakeElo:
    def __init__(self, ratings):
        self._r = ratings

    def get_rating(self, t):
        return self._r.get(t, 1500.0)


def _strengths():
    # high-confidence strong-player team, a high-conf weak team, a data-poor team
    return {
        "STRONG": {"overall": 0.9, "confidence": 1.0},
        "WEAK": {"overall": 0.3, "confidence": 1.0},
        "POOR": {"overall": 0.2, "confidence": 0.1},
    }


def test_champion_rating_between_elo_and_player():
    elo = _FakeElo({"STRONG": 1600})
    bridge = StrengthBridge(intercept=1400, slope=600)  # player_elo(0.9)=1940
    ratings = build_champion_ratings(elo, _strengths(), bridge, lam=CHAMPION_LAMBDA)
    # blended = 1600 + 0.6*1.0*(1940-1600) = 1804
    assert 1600 < ratings["STRONG"] < 1940
    assert abs(ratings["STRONG"] - 1804) < 1.0


def test_data_poor_team_stays_near_elo():
    elo = _FakeElo({"POOR": 1700})
    bridge = StrengthBridge(intercept=1400, slope=600)  # player_elo(0.2)=1520
    ratings = build_champion_ratings(elo, _strengths(), bridge, lam=CHAMPION_LAMBDA)
    # confidence 0.1 -> barely moves from 1700
    assert abs(ratings["POOR"] - 1700) < 20


def test_all_teams_get_a_rating():
    elo = _FakeElo({})
    bridge = StrengthBridge(intercept=1400, slope=600)
    ratings = build_champion_ratings(elo, _strengths(), bridge, lam=CHAMPION_LAMBDA)
    assert set(ratings) == {"STRONG", "WEAK", "POOR"}
    assert all(isinstance(v, float) for v in ratings.values())
