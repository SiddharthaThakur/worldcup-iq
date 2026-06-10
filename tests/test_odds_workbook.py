"""
Tests for ingesting the football-data.co.uk World Cup workbook
(data/raw/WorldCup2026.xlsx — contains sheets for 2014/2018/2022/2026).
"""

from pathlib import Path

import numpy as np
import pytest

from src.data.odds_loader import load_wc_odds_workbook

WORKBOOK = Path("data/raw/WorldCup2026.xlsx")

pytestmark = pytest.mark.skipif(not WORKBOOK.exists(), reason="odds workbook not downloaded")


@pytest.fixture(scope="module")
def odds_2018():
    return load_wc_odds_workbook(WORKBOOK, "WorldCup2018")


def test_all_64_matches_loaded(odds_2018):
    assert len(odds_2018) == 64


def test_match_ids_use_fifa_codes(odds_2018):
    # match_id format: YYYY-MM-DD_HOME_AWAY with 3-letter codes
    final = odds_2018[odds_2018["match_id"].str.startswith("2018-07-15")]
    assert final["match_id"].iloc[0] == "2018-07-15_FRA_CRO"


def test_devigged_probs_sum_to_one(odds_2018):
    totals = odds_2018[["prob_home", "prob_draw", "prob_away"]].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)


def test_overround_was_present_and_removed(odds_2018):
    # Bookmaker margins on WC matches are typically 2-8%
    assert (odds_2018["overround"] > 0).all()
    assert (odds_2018["overround"] < 0.15).all()


def test_result_is_90_minute_result(odds_2018):
    # Croatia beat England 2-1 AFTER extra time in the 2018 semi —
    # at 90 minutes it was 1-1, so the 1X2 result must be a DRAW.
    semi = odds_2018[odds_2018["match_id"] == "2018-07-11_CRO_ENG"]
    assert len(semi) == 1
    assert semi["result_90"].iloc[0] == "D"


def test_2022_sheet_also_loads():
    df = load_wc_odds_workbook(WORKBOOK, "WorldCup2022")
    assert len(df) == 64
    # 2022 final: Argentina v France, 3-3 after ET → 2-2 at 90 → draw
    final = df[df["match_id"] == "2022-12-18_ARG_FRA"]
    assert final["result_90"].iloc[0] == "D"
