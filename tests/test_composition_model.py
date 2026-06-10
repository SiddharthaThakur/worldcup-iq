"""
Tests for the Phase 3 composition model: convert player strength to a
rating on the Elo scale, blend with Elo by confidence, predict matches.
"""

import numpy as np

from src.models.composition_model import (
    StrengthBridge,
    blended_rating,
    fit_strength_to_elo_bridge,
    predict_match_composition,
)
from src.models.dixon_coles import DixonColesParams

PARAMS = DixonColesParams(intercept=0.12, elo_coef=0.24, home_adv=0.26,
                          rho=-0.03, fitted_on="test", n_matches=1)


def test_bridge_recovers_linear_relationship():
    # Elo = 1500 + 600*strength exactly -> bridge must recover it
    strengths = [0.2, 0.4, 0.6, 0.8]
    elos = [1500 + 600 * s for s in strengths]
    bridge = fit_strength_to_elo_bridge(strengths, elos)
    assert abs(bridge.intercept - 1500) < 1.0
    assert abs(bridge.slope - 600) < 1.0


def test_player_implied_elo_monotonic():
    bridge = StrengthBridge(intercept=1500, slope=600)
    assert bridge.to_elo(0.9) > bridge.to_elo(0.5)


def test_blend_endpoints():
    # lam=0 -> pure Elo; lam=1,conf=1 -> pure player; conf=0 -> pure Elo
    assert blended_rating(elo=1800, player_elo=1600, confidence=1.0, lam=0.0) == 1800
    assert blended_rating(elo=1800, player_elo=1600, confidence=1.0, lam=1.0) == 1600
    assert blended_rating(elo=1800, player_elo=1600, confidence=0.0, lam=1.0) == 1800


def test_blend_is_confidence_weighted():
    # Half confidence -> halfway nudge toward player rating
    r = blended_rating(elo=1800, player_elo=1600, confidence=0.5, lam=1.0)
    assert abs(r - 1700) < 1e-9


def test_low_confidence_team_stays_near_elo():
    # A data-poor team (confidence 0.1) barely moves from its Elo
    r = blended_rating(elo=1750, player_elo=1300, confidence=0.1, lam=0.5)
    assert abs(r - 1750) < 25


def test_prediction_valid_distribution_and_favors_stronger():
    bridge = StrengthBridge(intercept=1500, slope=600)
    pred = predict_match_composition(
        home="AAA", away="BBB",
        home_elo=1700, away_elo=1500,
        home_strength={"overall": 0.9, "confidence": 1.0},
        away_strength={"overall": 0.4, "confidence": 1.0},
        bridge=bridge, params=PARAMS, lam=0.5, neutral=True)
    total = pred.prob_home_win + pred.prob_draw + pred.prob_away_win
    assert abs(total - 1.0) < 1e-6
    assert pred.prob_home_win > pred.prob_away_win


def test_player_signal_can_override_elo():
    # Elo says even, but players strongly favor home -> home favored
    bridge = StrengthBridge(intercept=1500, slope=600)
    pred = predict_match_composition(
        home="AAA", away="BBB",
        home_elo=1600, away_elo=1600,
        home_strength={"overall": 0.95, "confidence": 1.0},
        away_strength={"overall": 0.45, "confidence": 1.0},
        bridge=bridge, params=PARAMS, lam=1.0, neutral=True)
    assert pred.prob_home_win > pred.prob_away_win
