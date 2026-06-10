"""
Tests for the Phase 3 player -> team strength aggregator.

Design: every player gets a 0-1 quality score from the BEST available
signal (EA rating > market value > caps floor). A team's strength is the
quality of its likely best XI, with a confidence weight from how much of
that XI rests on real signal vs the caps-only floor.
"""

import numpy as np
import pandas as pd

from src.models.player_strength import (
    best_xi,
    player_quality_score,
    team_strength,
)


def _p(name, pos, **kw):
    base = dict(name=name, position=pos, ea_overall=None, tm_value_eur=None,
                us_npxg90=None, us_xa90=None, caps=0, goals=0,
                data_quality="tier4_caps_only")
    base.update(kw)
    return base


def test_quality_monotonic_in_ea_rating():
    low = player_quality_score(_p("a", "MF", ea_overall=65))[0]
    high = player_quality_score(_p("b", "MF", ea_overall=88))[0]
    assert 0 <= low < high <= 1


def test_quality_falls_back_to_value_then_caps():
    by_value = player_quality_score(_p("a", "FW", tm_value_eur=40_000_000))
    by_caps = player_quality_score(_p("b", "FW", caps=80))
    floor = player_quality_score(_p("c", "FW"))
    assert by_value[1] == "ea_or_value"
    assert by_caps[1] == "caps"
    assert floor[1] == "floor"
    # A €40M player outranks an 80-cap journeyman outranks an unknown
    assert by_value[0] > by_caps[0] > floor[0]


def test_ea_preferred_over_value():
    # When both exist, EA rating is the basis (cross-league calibrated)
    _, basis = player_quality_score(_p("a", "FW", ea_overall=80, tm_value_eur=5_000_000))
    assert basis == "ea_or_value"


def test_best_xi_returns_eleven_with_a_keeper():
    squad = pd.DataFrame(
        [_p(f"gk{i}", "GK", ea_overall=70 + i) for i in range(3)]
        + [_p(f"df{i}", "DF", ea_overall=70 + i) for i in range(8)]
        + [_p(f"mf{i}", "MF", ea_overall=70 + i) for i in range(8)]
        + [_p(f"fw{i}", "FW", ea_overall=70 + i) for i in range(7)]
    )
    xi = best_xi(squad)
    assert len(xi) == 11
    assert (xi["pos_group"] == "GK").sum() == 1
    # Picks the strongest keeper available
    assert xi[xi["pos_group"] == "GK"].iloc[0]["name"] == "gk2"


def test_stronger_squad_has_higher_strength():
    strong = pd.DataFrame(
        [_p("gk", "GK", ea_overall=85)]
        + [_p(f"d{i}", "DF", ea_overall=84) for i in range(4)]
        + [_p(f"m{i}", "MF", ea_overall=86) for i in range(4)]
        + [_p(f"f{i}", "FW", ea_overall=88) for i in range(2)]
    )
    weak = pd.DataFrame(
        [_p("gk", "GK", ea_overall=64)]
        + [_p(f"d{i}", "DF", ea_overall=63) for i in range(4)]
        + [_p(f"m{i}", "MF", ea_overall=65) for i in range(4)]
        + [_p(f"f{i}", "FW", ea_overall=66) for i in range(2)]
    )
    s_strong = team_strength(strong)
    s_weak = team_strength(weak)
    assert s_strong["overall"] > s_weak["overall"]
    assert 0 <= s_weak["overall"] <= 1 and 0 <= s_strong["overall"] <= 1


def test_confidence_reflects_signal_coverage():
    full_signal = pd.DataFrame(
        [_p("gk", "GK", ea_overall=80)]
        + [_p(f"d{i}", "DF", ea_overall=80) for i in range(4)]
        + [_p(f"m{i}", "MF", ea_overall=80) for i in range(4)]
        + [_p(f"f{i}", "FW", ea_overall=80) for i in range(2)]
    )
    caps_only = pd.DataFrame(
        [_p("gk", "GK", caps=30)]
        + [_p(f"d{i}", "DF", caps=30) for i in range(4)]
        + [_p(f"m{i}", "MF", caps=30) for i in range(4)]
        + [_p(f"f{i}", "FW", caps=30) for i in range(2)]
    )
    assert team_strength(full_signal)["confidence"] > team_strength(caps_only)["confidence"]
    assert team_strength(caps_only)["confidence"] < 0.5


def test_attack_defense_split_responds_to_where_quality_sits():
    attack_heavy = pd.DataFrame(
        [_p("gk", "GK", ea_overall=68)]
        + [_p(f"d{i}", "DF", ea_overall=68) for i in range(4)]
        + [_p(f"m{i}", "MF", ea_overall=82) for i in range(4)]
        + [_p(f"f{i}", "FW", ea_overall=90) for i in range(2)]
    )
    s = team_strength(attack_heavy)
    assert s["attack"] > s["defense"]
