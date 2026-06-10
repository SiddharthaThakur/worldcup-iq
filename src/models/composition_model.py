"""
Phase 3 composition model: player strength -> match probabilities.

What this does in simple English:
    Step 1 gave each team a 0-1 player-strength number. To predict a
    match we must turn that into win/draw/loss probabilities on the SAME
    footing as the Elo baseline — otherwise we can't fairly compare them.

    Two moves:
    1. BRIDGE — convert player strength to "Elo-equivalent points" with a
       simple straight-line fit (player strength 0.9 ≈ such-and-such Elo).
       This lets the player rating reuse the already-fitted, already-
       validated Elo->goals mapping. We change WHAT the rating is, not how
       a rating becomes a prediction (the project's apples-to-apples rule).
    2. BLEND — the final rating is Elo nudged toward the player-implied
       rating, by how much we trust the player signal:
           rating = Elo + lam * confidence * (player_elo - Elo)
       For a data-rich team the nudge is real; for a data-poor team
       (confidence ~0) it stays at Elo. `lam` is the overall trust dial,
       set by validation (Step 3), NOT guessed. Until validated it is
       provisional and the model does not enter the live scorecard.

    With the blended rating in hand, prediction is identical to the
    baseline: same Dixon-Coles, same everything.
"""

from dataclasses import dataclass

import numpy as np

from src.models.dixon_coles import DixonColesParams, predict_match


@dataclass
class StrengthBridge:
    """Linear map from 0-1 player strength to Elo-equivalent points."""

    intercept: float
    slope: float

    def to_elo(self, strength: float) -> float:
        return self.intercept + self.slope * strength


def fit_strength_to_elo_bridge(strengths, elos) -> StrengthBridge:
    """Least-squares line: elo ≈ intercept + slope * strength.

    Fitted across teams so player strength lands in Elo units. This only
    rescales — the disagreement between a team's player strength and its
    Elo (the informative part) is preserved in the residual and survives
    into the blend.
    """
    s = np.asarray(strengths, dtype=float)
    e = np.asarray(elos, dtype=float)
    slope, intercept = np.polyfit(s, e, 1)
    return StrengthBridge(intercept=float(intercept), slope=float(slope))


def blended_rating(elo: float, player_elo: float,
                   confidence: float, lam: float) -> float:
    """Elo nudged toward the player-implied rating by lam * confidence.

    lam=0 -> pure Elo (baseline). lam=1, confidence=1 -> pure player.
    Low confidence (data-poor team) keeps the rating near Elo.
    """
    weight = lam * confidence
    return (1.0 - weight) * elo + weight * player_elo


def predict_match_composition(
    home: str, away: str,
    home_elo: float, away_elo: float,
    home_strength: dict, away_strength: dict,
    bridge: StrengthBridge, params: DixonColesParams,
    lam: float, neutral: bool = True,
):
    """Predict a match from blended (Elo + player) ratings."""
    home_blend = blended_rating(
        home_elo, bridge.to_elo(home_strength["overall"]),
        home_strength["confidence"], lam)
    away_blend = blended_rating(
        away_elo, bridge.to_elo(away_strength["overall"]),
        away_strength["confidence"], lam)
    return predict_match(home, away, home_blend, away_blend, params, neutral=neutral)
