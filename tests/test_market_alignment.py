"""
Tests for aligning market odds rows to our results rows when the two
sources disagree about which team is 'home' (FIFA designations differ).
"""

import pandas as pd

from src.evaluation.market_compare import align_market_to_results


def test_alignment_handles_flipped_home_away():
    results = pd.DataFrame([
        {"match_id": "2018-06-25_RUS_URU", "date": pd.Timestamp("2018-06-25"),
         "home_code": "RUS", "away_code": "URU"},
        {"match_id": "2018-06-14_RUS_KSA", "date": pd.Timestamp("2018-06-14"),
         "home_code": "RUS", "away_code": "KSA"},
    ])
    odds = pd.DataFrame([
        # Flipped orientation: workbook says Uruguay was home
        {"match_id": "2018-06-25_URU_RUS", "date": "2018-06-25",
         "home_code": "URU", "away_code": "RUS",
         "prob_home": 0.5, "prob_draw": 0.3, "prob_away": 0.2, "result_90": "H"},
        # Same orientation
        {"match_id": "2018-06-14_RUS_KSA", "date": "2018-06-14",
         "home_code": "RUS", "away_code": "KSA",
         "prob_home": 0.6, "prob_draw": 0.25, "prob_away": 0.15, "result_90": "H"},
    ])

    aligned = align_market_to_results(results, odds)
    assert len(aligned) == 2

    flipped = aligned[aligned["match_id"] == "2018-06-25_RUS_URU"].iloc[0]
    # Uruguay won 3-0 as workbook-home → from Russia's perspective it's an away win
    assert flipped["prob_home"] == 0.2
    assert flipped["prob_draw"] == 0.3
    assert flipped["prob_away"] == 0.5
    assert flipped["result_90"] == "A"

    direct = aligned[aligned["match_id"] == "2018-06-14_RUS_KSA"].iloc[0]
    assert direct["prob_home"] == 0.6
    assert direct["result_90"] == "H"


def test_unmatched_rows_are_reported_not_silently_dropped():
    results = pd.DataFrame([
        {"match_id": "2018-06-14_RUS_KSA", "date": pd.Timestamp("2018-06-14"),
         "home_code": "RUS", "away_code": "KSA"},
        {"match_id": "2018-06-15_EGY_URU", "date": pd.Timestamp("2018-06-15"),
         "home_code": "EGY", "away_code": "URU"},
    ])
    odds = pd.DataFrame([
        {"match_id": "2018-06-14_RUS_KSA", "date": "2018-06-14",
         "home_code": "RUS", "away_code": "KSA",
         "prob_home": 0.6, "prob_draw": 0.25, "prob_away": 0.15, "result_90": "H"},
    ])
    aligned = align_market_to_results(results, odds)
    assert len(aligned) == 1
    assert aligned.attrs["unmatched_match_ids"] == ["2018-06-15_EGY_URU"]
