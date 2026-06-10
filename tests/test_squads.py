"""
Tests for parsing the 2026 World Cup squads from the cached Wikipedia page.
"""

from pathlib import Path

import pytest

from src.data.squads import parse_squads

HTML = Path("data/raw/wc2026_squads.html")

pytestmark = pytest.mark.skipif(not HTML.exists(), reason="squads HTML not downloaded")


@pytest.fixture(scope="module")
def squads():
    return parse_squads(HTML)


def test_48_teams(squads):
    assert len(squads) == 48


def test_team_keys_are_fifa_codes(squads):
    for code in ("MEX", "CAN", "USA", "FRA", "BRA", "CUW"):
        assert code in squads


def test_squad_sizes_in_range(squads):
    # FIFA allows up to 26; a few squads show fewer on Wikipedia when a
    # withdrawn player hasn't been replaced yet (e.g. Argentina's vacant #2)
    for code, players in squads.items():
        assert 23 <= len(players) <= 26, f"{code} has {len(players)}"


def test_player_fields(squads):
    p = squads["MEX"][0]
    for field in ("name", "position", "birth_date", "caps", "goals", "club", "number"):
        assert field in p
    assert p["position"] in ("GK", "DF", "MF", "FW")
    assert isinstance(p["caps"], int)
    # birth_date is ISO formatted
    assert len(p["birth_date"]) == 10 and p["birth_date"][4] == "-"


def test_positions_cover_all_roles(squads):
    for code, players in squads.items():
        positions = {p["position"] for p in players}
        assert {"GK", "DF", "MF", "FW"} <= positions, f"{code}: {positions}"


def test_no_captain_annotation_in_names(squads):
    for players in squads.values():
        for p in players:
            assert "captain" not in p["name"].lower()
            assert "(c)" not in p["name"].lower()
