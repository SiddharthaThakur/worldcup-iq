"""
Dixon-Coles model for football score prediction — with FITTED parameters.

What this does in simple English:
    Given two teams' strength ratings, this model predicts the probability
    of every possible scoreline (0-0, 1-0, ... up to 5-5) using a Poisson
    distribution with the Dixon-Coles (1997) correction for low-scoring draws.

    CRITICAL DESIGN POINT: nothing here is hand-tuned. The mapping from
    Elo difference to expected goals, and the rho correction parameter,
    are both FITTED on historical international results via
    src/models/fit_params.py. Hardcoded constants would silently
    miscalibrate every probability — and calibration is this project's
    entire deliverable.

    Fitted parameters are loaded from models/dixon_coles_params.json.
    If that file doesn't exist, prediction raises an error rather than
    falling back to made-up defaults. Fail loudly, not wrongly.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 8  # consider scorelines up to 7-7 (tail matters for calibration)
PARAMS_PATH = Path("models/dixon_coles_params.json")

# Goal-level calibration (D025). The model fitted on all internationals,
# combined with the champion+ rating blend compressing team gaps, produced
# ~2.46 total goals/game vs ~2.6 in real recent World Cups. This multiplier
# scales both teams' expected goals to match WC scoring. Affects absolute
# goal levels (and slightly lowers draw rates); the win/loss split is
# nearly unchanged since it depends on the DIFFERENCE in expected goals.
GOAL_CALIBRATION = 1.06


@dataclass
class DixonColesParams:
    """Fitted model parameters. Produced by fit_params.py, never hand-set.

    Attributes:
        intercept: log expected goals for an average team vs average team
        elo_coef: coefficient on (elo_diff / 100) in the log-linear goal model
        home_adv: log-scale home advantage (applied only when not neutral)
        rho: Dixon-Coles low-score correction parameter
        fitted_on: description of the training data (for provenance)
        n_matches: number of matches used in fitting
    """

    intercept: float
    elo_coef: float
    home_adv: float
    rho: float
    fitted_on: str = ""
    n_matches: int = 0

    def save(self, path: Path = PARAMS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> "DixonColesParams":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.models.fit_params` first. "
                "This model refuses to predict with unfitted parameters."
            )
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class MatchPrediction:
    """Full prediction output for a single match."""

    home_team: str
    away_team: str
    home_expected_goals: float
    away_expected_goals: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    scoreline_probs: np.ndarray  # shape (MAX_GOALS, MAX_GOALS); [i,j] = P(home=i, away=j)
    most_likely_score: tuple[int, int]

    def to_dict(self) -> dict:
        """JSON-serializable summary (for prediction lock-in files)."""
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_xg": round(self.home_expected_goals, 3),
            "away_xg": round(self.away_expected_goals, 3),
            "prob_home_win": round(self.prob_home_win, 4),
            "prob_draw": round(self.prob_draw, 4),
            "prob_away_win": round(self.prob_away_win, 4),
            "most_likely_score": list(self.most_likely_score),
        }


def dixon_coles_correction(
    home_goals: int, away_goals: int,
    lambda_home: float, lambda_away: float, rho: float,
) -> float:
    """Dixon-Coles tau correction for scores (0,0), (1,0), (0,1), (1,1)."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def strengths_to_expected_goals(
    home_strength: float,
    away_strength: float,
    params: DixonColesParams,
    neutral: bool = True,
) -> tuple[float, float]:
    """Convert strength ratings (Elo or player-derived, same scale) to expected goals.

    Log-linear model fitted by Poisson regression:
        log(lambda_home) = intercept + elo_coef * (diff/100) + home_adv * (1 - neutral)
        log(lambda_away) = intercept - elo_coef * (diff/100)
    """
    diff = (home_strength - away_strength) / 100.0
    log_lh = params.intercept + params.elo_coef * diff + (0.0 if neutral else params.home_adv)
    log_la = params.intercept - params.elo_coef * diff
    lambda_home = float(np.clip(np.exp(log_lh) * GOAL_CALIBRATION, 0.1, 6.0))
    lambda_away = float(np.clip(np.exp(log_la) * GOAL_CALIBRATION, 0.1, 6.0))
    return lambda_home, lambda_away


def scoreline_matrix(
    lambda_home: float, lambda_away: float, rho: float
) -> np.ndarray:
    """Probability matrix over scorelines. [i,j] = P(home=i, away=j). Normalized."""
    home_pmf = poisson.pmf(np.arange(MAX_GOALS), lambda_home)
    away_pmf = poisson.pmf(np.arange(MAX_GOALS), lambda_away)
    probs = np.outer(home_pmf, away_pmf)
    for i in (0, 1):
        for j in (0, 1):
            probs[i, j] *= dixon_coles_correction(i, j, lambda_home, lambda_away, rho)
    return probs / probs.sum()


def outcome_probs(probs: np.ndarray) -> tuple[float, float, float]:
    """(P_home_win, P_draw, P_away_win) from a scoreline matrix."""
    p_home = float(np.sum(np.tril(probs, -1)))  # rows (home) > cols (away)
    p_draw = float(np.trace(probs))
    p_away = float(np.sum(np.triu(probs, 1)))
    return p_home, p_draw, p_away


def predict_match(
    home_team: str,
    away_team: str,
    home_strength: float,
    away_strength: float,
    params: DixonColesParams,
    neutral: bool = True,
) -> MatchPrediction:
    """Full match prediction from strength ratings and FITTED parameters."""
    lambda_h, lambda_a = strengths_to_expected_goals(
        home_strength, away_strength, params, neutral
    )
    probs = scoreline_matrix(lambda_h, lambda_a, params.rho)
    p_home, p_draw, p_away = outcome_probs(probs)
    best = np.unravel_index(np.argmax(probs), probs.shape)

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        home_expected_goals=lambda_h,
        away_expected_goals=lambda_a,
        prob_home_win=p_home,
        prob_draw=p_draw,
        prob_away_win=p_away,
        scoreline_probs=probs,
        most_likely_score=(int(best[0]), int(best[1])),
    )
