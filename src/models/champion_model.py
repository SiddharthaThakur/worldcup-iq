"""
The champion model: Elo blended with player composition (D022).

What this does in simple English:
    As of D022 the headline model is no longer pure Elo — it's Elo nudged
    by player strength, by how much we trust each team's player data. This
    module turns that into one Elo-scale rating per team, so the existing
    prediction and tournament-simulation code can use it in place of raw
    Elo without any other change.

    rating = Elo + lam * confidence * (player_implied_Elo - Elo)

    lam=0.6 is the blend weight validated on club data (D021). Data-poor
    teams (low confidence) stay at their Elo. The Elo baseline is still
    computed and scored separately — promotion did not end the comparison.
"""

import pandas as pd

from src.models.composition_model import (
    StrengthBridge,
    blended_rating,
    fit_strength_to_elo_bridge,
)
from src.models.elo import EloSystem
from src.models.player_strength import team_strength

CHAMPION_LAMBDA = 0.6  # validated optimum (D021)
UNIFIED_PATH = "data/processed/unified_players.parquet"


def build_champion_ratings(elo: EloSystem, strengths: dict,
                           bridge: StrengthBridge,
                           lam: float = CHAMPION_LAMBDA) -> dict[str, float]:
    """Blended Elo-scale rating per team. Keys are the teams in `strengths`."""
    out = {}
    for team, s in strengths.items():
        player_elo = bridge.to_elo(s["overall"])
        out[team] = float(blended_rating(elo.get_rating(team), player_elo,
                                         s["confidence"], lam))
    return out


def load_wc_strengths_and_bridge(elo: EloSystem):
    """Compute WC team strengths + the strength->Elo bridge (shared helper)."""
    df = pd.read_parquet(UNIFIED_PATH)
    strengths = {t: team_strength(sq) for t, sq in df.groupby("team")}
    teams = list(strengths)
    bridge = fit_strength_to_elo_bridge(
        [strengths[t]["overall"] for t in teams],
        [elo.get_rating(t) for t in teams])
    return strengths, bridge


def champion_ratings_for_wc(elo: EloSystem, lam: float = CHAMPION_LAMBDA) -> dict[str, float]:
    """Convenience: blended champion ratings for all 48 WC teams."""
    strengths, bridge = load_wc_strengths_and_bridge(elo)
    return build_champion_ratings(elo, strengths, bridge, lam)
