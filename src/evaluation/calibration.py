"""
Calibration and scoring metrics for evaluating prediction quality.

What this does in simple English:
    After we predict "France has a 65% chance of winning," and the match
    happens, we need to measure how GOOD that prediction was. This module
    computes standard scoring metrics that tell us:

    - Brier score: How far off were our probabilities on average? (lower = better)
    - Log-loss: How badly did we get punished for confident wrong predictions?
    - RPS: A smarter version of Brier that accounts for ordinal outcomes
    - Reliability: When we say "60% chance," does that event happen ~60% of the time?

    We also compare our model's scores against bookmaker-implied probabilities
    to answer the real question: "Are we adding value, or should people just
    look at the betting odds?"
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PredictionRecord:
    """One match's prediction and outcome."""

    match_id: str
    model_name: str
    prob_home: float
    prob_draw: float
    prob_away: float
    actual_result: str  # "H", "D", or "A"


def brier_score(predictions: list[PredictionRecord]) -> float:
    """Compute multiclass Brier score across all predictions.

    BS = (1/N) * sum over matches of (1/3) * sum_k (p_k - o_k)^2

    Perfect = 0, Uniform random = 0.667
    """
    if not predictions:
        return float("nan")

    total = 0.0
    for p in predictions:
        outcome = _result_to_onehot(p.actual_result)
        probs = np.array([p.prob_home, p.prob_draw, p.prob_away])
        total += np.sum((probs - outcome) ** 2) / 3.0

    return total / len(predictions)


def log_loss(predictions: list[PredictionRecord], clip: float = 0.01) -> float:
    """Compute log-loss (cross-entropy).

    LL = -(1/N) * sum of log(p_actual_outcome)

    Perfect = 0, Uniform = log(3) ≈ 1.099
    """
    if not predictions:
        return float("nan")

    total = 0.0
    for p in predictions:
        probs = np.array([p.prob_home, p.prob_draw, p.prob_away])
        probs = np.clip(probs, clip, 1.0 - clip)
        probs /= probs.sum()  # Re-normalize after clipping
        outcome_idx = {"H": 0, "D": 1, "A": 2}[p.actual_result]
        total -= np.log(probs[outcome_idx])

    return total / len(predictions)


def ranked_probability_score(predictions: list[PredictionRecord]) -> float:
    """Compute Ranked Probability Score (RPS).

    Better than Brier for ordinal outcomes — penalizes less when you predict
    "home win" and the result is a draw vs predicting "home win" and the
    result is an away win.

    RPS = (1/2) * sum_k (CDF_predicted_k - CDF_actual_k)^2
    """
    if not predictions:
        return float("nan")

    total = 0.0
    for p in predictions:
        outcome = _result_to_onehot(p.actual_result)
        probs = np.array([p.prob_home, p.prob_draw, p.prob_away])

        cdf_pred = np.cumsum(probs)
        cdf_actual = np.cumsum(outcome)

        # RPS uses cumulative distributions, sum only first 2 (last is always 1.0)
        total += np.sum((cdf_pred[:2] - cdf_actual[:2]) ** 2) / 2.0

    return total / len(predictions)


def accuracy(predictions: list[PredictionRecord]) -> float:
    """Simple accuracy: did the highest-probability outcome occur?"""
    if not predictions:
        return float("nan")

    correct = 0
    for p in predictions:
        probs = {"H": p.prob_home, "D": p.prob_draw, "A": p.prob_away}
        predicted = max(probs, key=probs.get)
        if predicted == p.actual_result:
            correct += 1

    return correct / len(predictions)


def reliability_bins(
    predictions: list[PredictionRecord],
    outcome: str = "H",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability diagram data for a specific outcome.

    Groups predictions into bins by confidence level, then compares
    average predicted probability to actual frequency in each bin.

    Args:
        predictions: list of prediction records
        outcome: which outcome to analyze ("H", "D", or "A")
        n_bins: number of bins

    Returns:
        DataFrame with columns: bin_center, avg_predicted, actual_frequency, count
    """
    data = []
    for p in predictions:
        prob = {"H": p.prob_home, "D": p.prob_draw, "A": p.prob_away}[outcome]
        actual = 1.0 if p.actual_result == outcome else 0.0
        data.append({"prob": prob, "actual": actual})

    df = pd.DataFrame(data)
    df["bin"] = pd.cut(df["prob"], bins=n_bins, labels=False)

    bins = df.groupby("bin").agg(
        avg_predicted=("prob", "mean"),
        actual_frequency=("actual", "mean"),
        count=("actual", "count"),
    ).reset_index()

    bins["bin_center"] = bins["avg_predicted"]
    return bins


def compare_to_market(
    model_predictions: list[PredictionRecord],
    market_predictions: list[PredictionRecord],
) -> pd.DataFrame:
    """Head-to-head comparison between model and bookmaker predictions.

    Returns per-match Brier scores for both, plus cumulative.
    """
    rows = []
    model_cumulative = 0.0
    market_cumulative = 0.0

    for mp, mkt in zip(model_predictions, market_predictions):
        assert mp.match_id == mkt.match_id, "Match IDs must align"

        outcome = _result_to_onehot(mp.actual_result)

        model_bs = np.sum((np.array([mp.prob_home, mp.prob_draw, mp.prob_away]) - outcome) ** 2) / 3
        market_bs = np.sum((np.array([mkt.prob_home, mkt.prob_draw, mkt.prob_away]) - outcome) ** 2) / 3

        model_cumulative += model_bs
        market_cumulative += market_bs
        n = len(rows) + 1

        rows.append({
            "match_id": mp.match_id,
            "model_brier": model_bs,
            "market_brier": market_bs,
            "model_better": model_bs < market_bs,
            "model_cumulative_avg": model_cumulative / n,
            "market_cumulative_avg": market_cumulative / n,
        })

    return pd.DataFrame(rows)


def scorecard(predictions: list[PredictionRecord], model_name: str = "Model") -> dict:
    """Generate a summary scorecard for a set of predictions."""
    return {
        "model": model_name,
        "n_matches": len(predictions),
        "brier_score": round(brier_score(predictions), 4),
        "log_loss": round(log_loss(predictions), 4),
        "rps": round(ranked_probability_score(predictions), 4),
        "accuracy": round(accuracy(predictions), 4),
    }


def _result_to_onehot(result: str) -> np.ndarray:
    """Convert H/D/A to one-hot: [home, draw, away]."""
    mapping = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}
    return np.array(mapping[result], dtype=float)
