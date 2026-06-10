"""
Tests for the core matcher behind the unified player table: matching a
squad player to a stats source on name + birth year, with nationality
disambiguation and a fuzzy fallback for name romanization.
"""

import pandas as pd

from src.data.unified_players import assign_tier, match_player_to_source


def _source(rows):
    return pd.DataFrame(rows, columns=["idx", "name", "birth_year", "nationality"])


def test_exact_name_and_birthyear():
    src = _source([(0, "Kylian Mbappe", 1998, "France")])
    i, method, score = match_player_to_source(
        "Kylian Mbappé", 1998, "FRA", src)
    assert i == 0
    assert method == "exact"


def test_birthyear_disambiguates_same_name():
    src = _source([(0, "Danilo", 1991, "Brazil"),
                   (1, "Danilo", 2001, "Portugal")])
    i, _, _ = match_player_to_source("Danilo", 2001, "POR", src)
    assert i == 1


def test_nationality_disambiguates_same_name_and_year():
    src = _source([(0, "Danilo", 1991, "Brazil"),
                   (1, "Danilo", 1991, "Portugal")])
    i, _, _ = match_player_to_source("Danilo", 1991, "POR", src)
    assert i == 1


def test_fuzzy_recovers_romanized_name_order():
    # Korean name order flipped between sources
    src = _source([(0, "Heung-Min Son", 1992, "Korea Republic")])
    i, method, score = match_player_to_source(
        "Son Heung-min", 1992, "KOR", src)
    assert i == 0
    assert method.startswith("fuzzy")


def test_fuzzy_requires_nationality_when_name_is_weak():
    # A loose name match to a DIFFERENT nationality must be rejected
    src = _source([(0, "Sonny", 1992, "England")])
    i, _, _ = match_player_to_source("Son Heung-min", 1992, "KOR", src)
    assert i is None


def test_no_match_returns_none():
    src = _source([(0, "Erling Haaland", 2000, "Norway")])
    i, method, _ = match_player_to_source("Lionel Messi", 1987, "ARG", src)
    assert i is None
    assert method == "none"


def test_wrong_birthyear_blocks_match():
    # Same name, but birth year off by more than a year -> no match
    src = _source([(0, "James Rodriguez", 1991, "Colombia")])
    i, _, _ = match_player_to_source("James Rodríguez", 1985, "COL", src)
    assert i is None


def test_tier_assignment_priority():
    assert assign_tier(has_understat=True, has_stats=True,
                       has_ea=True, has_value=True) == "tier1_xg"
    assert assign_tier(has_understat=False, has_stats=True,
                       has_ea=True, has_value=True) == "tier2_stats"
    assert assign_tier(has_understat=False, has_stats=False,
                       has_ea=True, has_value=True) == "tier3_rating"
    assert assign_tier(has_understat=False, has_stats=False,
                       has_ea=False, has_value=True) == "tier3_value"
    assert assign_tier(has_understat=False, has_stats=False,
                       has_ea=False, has_value=False) == "tier4_caps_only"
