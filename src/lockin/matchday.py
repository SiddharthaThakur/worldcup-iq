"""
Build model predictions for one 2026 matchday, ready for lock-in.

What this does in simple English:
    Given a date, look up that day's World Cup fixtures, compute each
    model's probabilities (currently the Elo+Dixon-Coles champion), and
    package them with conservative kickoff times for the lock script.

    We don't track exact kickoff times, so every match is treated as
    kicking off at MIDNIGHT UTC of its match date — strictly earlier
    than any real kickoff. Practical consequence: predictions must be
    locked the day before. Refusing a lockable match is acceptable;
    accepting an unlockable one is not.

Usage:
    python -m src.lockin.matchday --matchday 2026-06-11        # dry run
    python -m src.lockin.matchday --matchday 2026-06-11 --lock # lock + commit
"""

import argparse

from src.data.results_loader import load_processed_results
from src.data.wc2026 import load_wc2026
from src.models.dixon_coles import DixonColesParams, predict_match
from src.models.elo import EloSystem

MODEL_NAME = "elo_dixon_coles_v1"


def build_matchday_predictions(matchday: str) -> tuple[list[dict], dict[str, str]]:
    """Predictions + conservative kickoff times for all fixtures on a date.

    Returns:
        (predictions, kickoff_times) — predictions are MatchPrediction
        dicts plus match_id/model_name; kickoff_times maps each match_id
        to midnight UTC of the match date.
    """
    wc = load_wc2026(save=False)
    day = wc.fixtures[wc.fixtures["date"].dt.strftime("%Y-%m-%d") == matchday]
    if day.empty:
        raise ValueError(f"No 2026 WC fixtures on {matchday}")

    params = DixonColesParams.load()
    elo = EloSystem().fit_from_results(load_processed_results())

    predictions, kickoffs = [], {}
    for _, row in day.iterrows():
        pred = predict_match(
            home_team=row["home_code"],
            away_team=row["away_code"],
            home_strength=elo.get_rating(row["home_code"]),
            away_strength=elo.get_rating(row["away_code"]),
            params=params,
            neutral=bool(row["neutral"]),
        )
        d = pred.to_dict()
        d["match_id"] = row["match_id"]
        d["model_name"] = MODEL_NAME
        d["group"] = row["group"]
        d["neutral"] = bool(row["neutral"])
        d["home_elo"] = round(elo.get_rating(row["home_code"]), 1)
        d["away_elo"] = round(elo.get_rating(row["away_code"]), 1)
        predictions.append(d)
        kickoffs[row["match_id"]] = f"{matchday}T00:00:00+00:00"

    return predictions, kickoffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matchday", required=True, help="YYYY-MM-DD")
    parser.add_argument("--lock", action="store_true",
                        help="write the lock file and git-commit it")
    args = parser.parse_args()

    preds, kickoffs = build_matchday_predictions(args.matchday)
    for p in preds:
        print(f"{p['match_id']}  H {p['prob_home_win']:.3f}  D {p['prob_draw']:.3f}  "
              f"A {p['prob_away_win']:.3f}  (xG {p['home_xg']:.2f}-{p['away_xg']:.2f})")

    if args.lock:
        from src.lockin.lock_predictions import lock_predictions
        lock_predictions(preds, args.matchday, kickoff_times=kickoffs)
    else:
        print("\nDry run. Add --lock to write the lock file and git-commit.")
