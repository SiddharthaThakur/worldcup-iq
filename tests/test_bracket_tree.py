"""Tests for the full knockout-bracket tree projection."""

from src.models.dixon_coles import DixonColesParams
from src.simulation.bracket_tree import (
    THIRD_SLOTS,
    assign_thirds,
    simulate_bracket_tree,
)

PARAMS = DixonColesParams(intercept=0.2, elo_coef=0.4, home_adv=0.2,
                          rho=-0.03, fitted_on="test", n_matches=1)


def test_eight_third_place_slots():
    assert len(THIRD_SLOTS) == 8


def test_assign_thirds_respects_eligibility_and_is_complete():
    # All 8 best-third groups qualify from a plausible set
    qual = {"A", "B", "C", "D", "E", "F", "G", "H"}
    out = assign_thirds(qual)
    assert len(out) == 8  # every slot filled
    # each assigned group must be eligible for its slot
    elig = dict(THIRD_SLOTS)
    for match_no, grp in out.items():
        assert grp in elig[match_no]
    # one group per slot, no repeats
    assert len(set(out.values())) == 8


def _groups():
    groups, strengths = {}, {}
    for letter in "ABCDEFGHIJKL":
        teams = [f"{letter}{i}" for i in range(4)]
        groups[letter] = teams
        for i, t in enumerate(teams):
            strengths[t] = 1700 - i * 100
    return groups, strengths


def test_tree_has_all_rounds_and_match_counts():
    groups, strengths = _groups()
    tree = simulate_bracket_tree(groups, strengths, set(), PARAMS, n_sims=500)
    by_round = {r["round"]: r["matches"] for r in tree}
    assert len(by_round["Round of 32"]) == 16
    assert len(by_round["Round of 16"]) == 8
    assert len(by_round["Quarter-finals"]) == 4
    assert len(by_round["Semi-finals"]) == 2
    assert len(by_round["Final"]) == 1


def test_final_slots_have_candidates():
    groups, strengths = _groups()
    tree = simulate_bracket_tree(groups, strengths, set(), PARAMS, n_sims=500)
    final = next(r for r in tree if r["round"] == "Final")["matches"][0]
    assert len(final["a"]) >= 1 and len(final["b"]) >= 1
