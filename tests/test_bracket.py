"""Tests for the Round-of-32 bracket projection."""

import numpy as np

from src.models.dixon_coles import DixonColesParams
from src.simulation.bracket import (
    R32_BRACKET,
    project_bracket,
    simulate_positions,
)

PARAMS = DixonColesParams(intercept=0.2, elo_coef=0.4, home_adv=0.2,
                          rho=-0.03, fitted_on="test", n_matches=1)


def _groups():
    # 12 groups of 4, strengths so the first team is clearly strongest
    groups, strengths = {}, {}
    for gi, letter in enumerate("ABCDEFGHIJKL"):
        teams = [f"{letter}{i}" for i in range(4)]
        groups[letter] = teams
        for i, t in enumerate(teams):
            strengths[t] = 1700 - i * 100  # T0 strongest
    return groups, strengths


def test_bracket_has_16_matches():
    assert len(R32_BRACKET) == 16


def test_every_group_winner_and_runnerup_slot_present():
    slots = [s for _, a, b in R32_BRACKET for s in (a, b)]
    winners = {s[1] for s in slots if s[0] == "W"}
    runners = {s[1] for s in slots if s[0] == "R"}
    assert winners == set("ABCDEFGHIJKL")
    assert runners == set("ABCDEFGHIJKL")


def test_projection_fills_slots_with_likely_teams():
    groups, strengths = _groups()
    pos = simulate_positions(groups, strengths, set(), PARAMS, n_sims=2000)
    bracket = project_bracket(pos, groups)
    assert len(bracket) == 16
    # Winner-E slot (match 74) should be topped by E0 (strongest in group E)
    m74 = next(m for m in bracket if m["match"] == 74)
    assert m74["a_label"] == "Winner E"
    assert m74["a"][0]["team"] == "E0"
    assert m74["a"][0]["p"] > 0.5


def test_third_place_slot_lists_eligible_groups():
    groups, strengths = _groups()
    pos = simulate_positions(groups, strengths, set(), PARAMS, n_sims=1000)
    bracket = project_bracket(pos, groups)
    m74 = next(m for m in bracket if m["match"] == 74)
    assert m74["b_label"].startswith("3rd:")
    # candidates must come from the eligible groups A/B/C/D/F
    for c in m74["b"]:
        assert c["team"][0] in "ABCDF"
