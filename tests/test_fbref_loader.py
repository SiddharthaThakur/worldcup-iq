"""
Tests for parsing the FBRef Big-5 standard stats page (Wayback snapshot).
"""

from pathlib import Path

import numpy as np
import pytest

from src.data.fbref_loader import load_fbref_big5

HTML = Path("data/raw/fbref_big5_standard_20260309.html")

pytestmark = pytest.mark.skipif(not HTML.exists(), reason="FBRef snapshot not downloaded")


@pytest.fixture(scope="module")
def fbref():
    return load_fbref_big5(HTML)


def test_substantial_player_count(fbref):
    assert len(fbref) > 2500


def test_no_repeated_header_rows(fbref):
    # FBRef repeats the header mid-table; those rows must be dropped
    assert not (fbref["player"] == "Player").any()


def test_known_player(fbref):
    haaland = fbref[fbref["player"] == "Erling Haaland"].iloc[0]
    assert haaland["born"] == 2000
    assert haaland["nation"] == "NOR"
    assert haaland["squad"] == "Manchester City"


def test_numeric_columns(fbref):
    for col in ("minutes", "goals", "assists", "born"):
        assert np.issubdtype(fbref[col].dtype, np.number), col


def test_snapshot_date_recorded(fbref):
    # Partial-season data must carry its as-of date so downstream code
    # and the write-up can state the boundary honestly
    assert (fbref["as_of"] == "2026-03-09").all()
