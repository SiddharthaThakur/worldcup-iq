"""
Tests for matching WC squad players (Wikipedia) to Understat players.

Understat has no birth dates, so club agreement is the validator:
a name match is trusted when the clubs agree, distrusted otherwise.
"""

import pandas as pd

from src.data.squad_resolution import resolve_squads_to_understat


def _understat(rows):
    return pd.DataFrame(rows, columns=["id", "player_name", "club", "league"])


def _squads(rows):
    return pd.DataFrame(rows, columns=["team", "name", "club", "position", "birth_date"])


def test_exact_name_and_club_match():
    squads = _squads([("FRA", "Kylian Mbappé", "Real Madrid", "FW", "1998-12-20")])
    us = _understat([("1", "Kylian Mbappe", "Real Madrid", "La_liga")])
    out = resolve_squads_to_understat(squads, us)
    row = out.iloc[0]
    assert row["understat_id"] == "1"
    assert row["match_method"] == "exact+club"


def test_accents_and_hyphens_handled():
    squads = _squads([("GER", "İlkay Gündoğan", "Galatasaray", "MF", "1990-10-24"),
                      ("ENG", "Trent Alexander-Arnold", "Real Madrid", "DF", "1998-10-07")])
    us = _understat([("7", "Ilkay Gundogan", "Galatasaray", "EPL"),
                     ("8", "Trent Alexander Arnold", "Real Madrid", "La_liga")])
    out = resolve_squads_to_understat(squads, us)
    assert out.iloc[0]["understat_id"] == "7"
    assert out.iloc[1]["understat_id"] == "8"


def test_same_name_disambiguated_by_club():
    squads = _squads([("BRA", "Danilo", "Flamengo", "DF", "1991-07-15"),
                      ("POR", "Danilo", "Al-Ittihad", "MF", "1991-09-09")])
    us = _understat([("21", "Danilo", "Juventus", "Serie_A")])
    out = resolve_squads_to_understat(squads, us)
    # Neither squad Danilo plays at Juventus — both must stay unmatched
    assert out["understat_id"].isna().all()


def test_club_disagreement_blocks_short_ambiguous_names():
    squads = _squads([("ESP", "Rodri", "Manchester City", "MF", "1996-06-22")])
    us = _understat([("31", "Rodri", "Manchester City", "EPL"),
                     ("32", "Rodri", "Real Betis", "La_liga")])
    out = resolve_squads_to_understat(squads, us)
    # Two Understat Rodris: club must decide, and it can
    assert out.iloc[0]["understat_id"] == "31"


def test_fuzzy_match_within_same_club():
    squads = _squads([("MAR", "Achraf Hakimi Mouh", "Paris Saint-Germain", "DF", "1998-11-04")])
    us = _understat([("41", "Achraf Hakimi", "Paris Saint Germain", "Ligue_1")])
    out = resolve_squads_to_understat(squads, us)
    row = out.iloc[0]
    assert row["understat_id"] == "41"
    assert row["match_method"].startswith("fuzzy")


def test_unmatched_player_flagged_not_dropped():
    squads = _squads([("UZB", "Eldor Shomurodov", "Istanbul Basaksehir", "FW", "1995-06-29")])
    us = _understat([("51", "Erling Haaland", "Manchester City", "EPL")])
    out = resolve_squads_to_understat(squads, us)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["understat_id"])
    assert out.iloc[0]["data_quality"] == "none"


def test_matched_players_get_tier1_flag():
    squads = _squads([("NOR", "Erling Haaland", "Manchester City", "FW", "2000-07-21")])
    us = _understat([("61", "Erling Haaland", "Manchester City", "EPL")])
    out = resolve_squads_to_understat(squads, us)
    assert out.iloc[0]["data_quality"] == "tier1_xg"
