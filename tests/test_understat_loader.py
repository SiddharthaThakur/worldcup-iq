"""
Tests for the Understat player-stats loader (parsing layer — network
fetching is tested implicitly by the cached raw files existing).
"""

from pathlib import Path

import numpy as np
import pytest

from src.data.understat_loader import LEAGUES, load_understat_players

RAW_DIR = Path("data/raw/understat")

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "EPL_2025.json").exists(),
    reason="no cached Understat data",
)


@pytest.fixture(scope="module")
def players():
    return load_understat_players(season=2025)


def test_known_leagues_constant():
    assert LEAGUES == ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]


def test_loads_available_leagues(players):
    # Loads whatever league files exist (at least EPL)
    assert "EPL" in set(players["league"])
    assert len(players[players["league"] == "EPL"]) > 400


def test_numeric_types(players):
    for col in ("minutes", "goals", "xg", "xa", "npxg", "shots"):
        assert np.issubdtype(players[col].dtype, np.number), col


def test_per90_columns(players):
    haaland = players[players["player_name"] == "Erling Haaland"].iloc[0]
    # 28.795 xG over 2979 minutes -> ~0.87 xG/90
    assert abs(haaland["xg90"] - 28.795 / 2979 * 90) < 0.01
    assert haaland["club"] == "Manchester City"


def test_low_minutes_players_have_nan_per90(players):
    # Per-90 rates from tiny samples are noise; below 270 min they're NaN
    low = players[players["minutes"] < 270]
    if len(low):
        assert low["xg90"].isna().all()


def test_season_column(players):
    assert (players["season"] == 2025).all()
