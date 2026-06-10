"""
Leak-free tournament backtest harness.

What this does in simple English:
    To honestly test the model on a past World Cup, we must pretend we're
    standing at the tournament's opening day: fit everything (Elo ratings
    AND the Dixon-Coles parameters) only on matches played BEFORE that day,
    then predict each tournament match. During the tournament, Elo updates
    after each completed match — that's fair, because by then the result
    is a known fact.

    The fitted backtest parameters are never written to disk: they must
    not overwrite the live params file used for real 2026 predictions.
"""

from dataclasses import dataclass

import pandas as pd

from src.evaluation.calibration import PredictionRecord, scorecard
from src.models.dixon_coles import DixonColesParams, predict_match
from src.models.elo import EloSystem
from src.models.fit_params import fit_all


@dataclass
class BacktestResult:
    """Everything produced by one tournament backtest."""

    records: list[PredictionRecord]
    params: DixonColesParams
    n_train_matches: int
    scorecard: dict


def backtest_world_cup(
    results: pd.DataFrame,
    wc_year: int,
    tournament: str = "FIFA World Cup",
    model_name: str = "elo_dixon_coles",
) -> BacktestResult:
    """Backtest the baseline on one World Cup with a strict time cutoff.

    Args:
        results: cleaned results (columns: date, home_code, away_code,
                 home_score, away_score, tournament, neutral, result, match_id)
        wc_year: tournament year (e.g. 2018, 2022)

    Returns:
        BacktestResult with one PredictionRecord per tournament match.
    """
    results = results.sort_values("date").reset_index(drop=True)
    is_test = (results["tournament"] == tournament) & (results["date"].dt.year == wc_year)
    test = results[is_test]
    if test.empty:
        raise ValueError(f"No {tournament} {wc_year} matches found in results")

    cutoff = test["date"].min()
    train = results[results["date"] < cutoff]

    params = fit_all(
        train,
        description=f"pre-{wc_year} cutoff {cutoff:%Y-%m-%d} (backtest, not saved)",
        save=False,
    )

    elo = EloSystem().fit_from_results(train)

    records = []
    for _, row in test.iterrows():
        pred = predict_match(
            home_team=row["home_code"],
            away_team=row["away_code"],
            home_strength=elo.get_rating(row["home_code"]),
            away_strength=elo.get_rating(row["away_code"]),
            params=params,
            neutral=bool(row["neutral"]),
        )
        records.append(
            PredictionRecord(
                match_id=row["match_id"],
                model_name=model_name,
                prob_home=pred.prob_home_win,
                prob_draw=pred.prob_draw,
                prob_away=pred.prob_away_win,
                actual_result=row["result"],
            )
        )
        # Online Elo update: the match is now in the past
        elo.update(
            home_team=row["home_code"],
            away_team=row["away_code"],
            home_score=row["home_score"],
            away_score=row["away_score"],
            tournament=row["tournament"],
            neutral=bool(row["neutral"]),
            date=str(row["date"])[:10],
        )

    return BacktestResult(
        records=records,
        params=params,
        n_train_matches=len(train),
        scorecard=scorecard(records, model_name=f"{model_name}_{wc_year}"),
    )
