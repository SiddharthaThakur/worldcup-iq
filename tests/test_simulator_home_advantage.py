"""
Home advantage must go to the HOST team, not to whichever team happens
to be passed first to the match simulator.
"""

import numpy as np

from src.models.dixon_coles import DixonColesParams
from src.simulation.tournament import simulate_match_oriented

# Exaggerated home advantage so the statistical test is unambiguous
PARAMS = DixonColesParams(intercept=0.2, elo_coef=0.4, home_adv=1.5, rho=0.0,
                          fitted_on="test", n_matches=1)


def _mean_goals(team_a, team_b, n=800):
    rng = np.random.default_rng(7)
    strengths = {"MEX": 1500.0, "RSA": 1500.0}
    totals = {team_a: 0, team_b: 0}
    for _ in range(n):
        ga, gb = simulate_match_oriented(
            team_a, team_b, strengths, host_teams={"MEX"},
            params=PARAMS, rng=rng,
        )
        totals[team_a] += ga
        totals[team_b] += gb
    return totals[team_a] / n, totals[team_b] / n


def test_host_first_gets_boost():
    mex_goals, rsa_goals = _mean_goals("MEX", "RSA")
    assert mex_goals > rsa_goals * 1.5


def test_host_second_still_gets_boost():
    # Same fixture, host passed SECOND — boost must follow the host
    rsa_goals, mex_goals = _mean_goals("RSA", "MEX")
    assert mex_goals > rsa_goals * 1.5


def test_no_host_means_neutral():
    rng = np.random.default_rng(7)
    strengths = {"FRA": 1500.0, "GER": 1500.0}
    totals = np.zeros(2)
    for _ in range(800):
        g = simulate_match_oriented("FRA", "GER", strengths, host_teams={"MEX"},
                                    params=PARAMS, rng=rng)
        totals += g
    # Equal strength, neutral venue: goal averages within 15% of each other
    assert abs(totals[0] - totals[1]) / max(totals) < 0.15
