"""
Load StatsBomb open data for 2018 and 2022 World Cups.

What this does in simple English:
    StatsBomb provides free event-level data for past World Cups —
    every pass, shot, tackle, and dribble with exact pitch coordinates.
    We use this for backtesting (did our model predict 2022 correctly?)
    and for enriching player profiles with international tournament data.
"""

from pathlib import Path

import pandas as pd

PROCESSED_PATH = Path("data/processed/statsbomb_wc_events.parquet")


def load_world_cup_matches(competition_id: int = 43, season_id: int = 106) -> pd.DataFrame:
    """Load match data for a World Cup from StatsBomb open data.

    Default: 2022 World Cup (competition_id=43, season_id=106)
    2018 World Cup: competition_id=43, season_id=3

    Requires: pip install statsbombpy
    """
    try:
        from statsbombpy import sb
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        return matches
    except ImportError:
        raise ImportError("Install statsbombpy: pip install statsbombpy")


def load_match_events(match_id: int) -> pd.DataFrame:
    """Load all events for a single match."""
    try:
        from statsbombpy import sb
        events = sb.events(match_id=match_id)
        return events
    except ImportError:
        raise ImportError("Install statsbombpy: pip install statsbombpy")


def build_wc_player_stats(competition_id: int = 43, season_id: int = 106) -> pd.DataFrame:
    """Aggregate player-level stats from World Cup event data.

    Returns per-player stats: goals, assists, shots, passes, tackles, etc.
    """
    # TODO: Implement aggregation from event data
    # 1. Load all matches for the competition/season
    # 2. Load events for each match
    # 3. Aggregate per-player counts
    # 4. Normalize to per-90 where applicable
    raise NotImplementedError("Phase 2: implement after data pipeline is working")
