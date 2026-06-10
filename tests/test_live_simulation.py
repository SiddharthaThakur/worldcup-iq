"""
Tests for fixing completed group games in the tournament simulator.
A played match must use its real score every time, not be re-simulated.
"""

import numpy as np

from src.models.dixon_coles import DixonColesParams
from src.simulation.tournament import play_group

PARAMS = DixonColesParams(intercept=0.2, elo_coef=0.3, home_adv=0.2,
                          rho=-0.03, fitted_on="test", n_matches=1)


def test_completed_game_uses_real_score():
    teams = ["AAA", "BBB", "CCC", "DDD"]
    strengths = {t: 1500.0 for t in teams}
    # AAA beat BBB 5-0 (real result), everything else simulated
    completed = {frozenset({"AAA", "BBB"}): {"AAA": 5, "BBB": 0}}
    rng = np.random.default_rng(1)
    gr = play_group(teams, strengths, set(), PARAMS, rng, completed=completed)
    table = {r["team"]: r for r in gr.standings}
    # AAA got at least 3 points and +5 GD from that one game alone
    assert table["AAA"]["points"] >= 3
    assert table["AAA"]["gf"] >= 5
    assert table["BBB"]["gd"] <= -5


def test_completed_result_is_deterministic():
    teams = ["AAA", "BBB", "CCC", "DDD"]
    strengths = {t: 1500.0 for t in teams}
    completed = {
        frozenset({"AAA", "BBB"}): {"AAA": 5, "BBB": 0},
        frozenset({"AAA", "CCC"}): {"AAA": 3, "CCC": 0},
        frozenset({"AAA", "DDD"}): {"AAA": 2, "DDD": 0},
        frozenset({"BBB", "CCC"}): {"BBB": 1, "CCC": 1},
        frozenset({"BBB", "DDD"}): {"BBB": 0, "DDD": 0},
        frozenset({"CCC", "DDD"}): {"CCC": 2, "DDD": 1},
    }
    # All games fixed -> standings identical across seeds
    a = play_group(teams, strengths, set(), PARAMS, np.random.default_rng(1), completed=completed)
    b = play_group(teams, strengths, set(), PARAMS, np.random.default_rng(99), completed=completed)
    assert [r["team"] for r in a.standings] == [r["team"] for r in b.standings]
    # AAA won all 3 -> 9 points, top
    assert a.standings[0]["team"] == "AAA"
    assert a.standings[0]["points"] == 9


def test_no_completed_games_still_works():
    teams = ["AAA", "BBB", "CCC", "DDD"]
    strengths = {t: 1500.0 for t in teams}
    gr = play_group(teams, strengths, set(), PARAMS, np.random.default_rng(1))
    assert len(gr.standings) == 4
