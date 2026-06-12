"""
Tests for extracting 2026 World Cup fixtures and deriving groups.

The martj42 raw CSV already contains the 72 group-stage fixtures (with
NA scores). Groups are recoverable from the fixture graph: in the group
stage, teams only play within their own group, so the 12 groups are the
connected components of the who-plays-whom graph.
"""

from pathlib import Path

import pytest

from src.data.wc2026 import load_wc2026

RAW = Path("data/raw/international_results.csv")

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw results not downloaded")


@pytest.fixture(scope="module")
def wc():
    return load_wc2026()


def test_72_group_stage_fixtures(wc):
    assert len(wc.fixtures) == 72


def test_48_teams_in_12_groups_of_4(wc):
    assert len(wc.groups) == 12
    for teams in wc.groups.values():
        assert len(teams) == 4
    all_teams = [t for teams in wc.groups.values() for t in teams]
    assert len(set(all_teams)) == 48


def test_every_team_plays_exactly_3_group_matches(wc):
    from collections import Counter
    c = Counter()
    for _, row in wc.fixtures.iterrows():
        c[row["home_code"]] += 1
        c[row["away_code"]] += 1
    assert all(v == 3 for v in c.values())


def test_host_groups_assigned_correctly(wc):
    assert "MEX" in wc.groups["A"]
    assert "CAN" in wc.groups["B"]
    assert "USA" in wc.groups["D"]


def test_official_group_letters(wc):
    # Verified against the FIFA draw (5 Dec 2025). I and J were once swapped.
    assert "ARG" in wc.groups["J"] and "ARG" not in wc.groups["I"]
    assert "FRA" in wc.groups["I"] and "FRA" not in wc.groups["J"]
    assert "ESP" in wc.groups["H"]
    assert "ENG" in wc.groups["L"]


def test_fixtures_have_required_columns(wc):
    for col in ("match_id", "date", "home_code", "away_code", "neutral", "group"):
        assert col in wc.fixtures.columns


def test_group_matches_are_within_groups(wc):
    team_group = {t: g for g, teams in wc.groups.items() for t in teams}
    for _, row in wc.fixtures.iterrows():
        assert team_group[row["home_code"]] == team_group[row["away_code"]]
        assert row["group"] == team_group[row["home_code"]]


def test_host_matches_not_neutral(wc):
    """USA/CAN/MEX playing in their own country get home advantage (D005)."""
    opener = wc.fixtures[(wc.fixtures["home_code"] == "MEX")
                         & (wc.fixtures["country"] == "Mexico")]
    assert len(opener) > 0
    assert not opener["neutral"].any()
