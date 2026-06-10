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
from src.models.champion_model import champion_ratings_for_wc
from src.models.dixon_coles import DixonColesParams, predict_match
from src.models.elo import EloSystem

# v3 champion: Elo + player-composition blend (D022). The Elo baseline is
# still locked and scored alongside it. June 11-13 locks were Elo-only
# (v1 importance-K, v2 flat-K) and remain valid under their own versions.
MODEL_NAME = "composition_champion_v3"
BASELINE_NAME = "elo_dixon_coles_v2"


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
    champ = champion_ratings_for_wc(elo)  # blended ratings (D022)

    # Two models locked per match: the composition champion (v3) and the
    # Elo baseline (still scored — the comparison never stops).
    models = {
        MODEL_NAME: lambda t: champ.get(t, elo.get_rating(t)),
        BASELINE_NAME: elo.get_rating,
    }

    predictions, kickoffs = [], {}
    for _, row in day.iterrows():
        h, a, neut = row["home_code"], row["away_code"], bool(row["neutral"])
        for model_name, rating_of in models.items():
            pred = predict_match(h, a, rating_of(h), rating_of(a), params, neutral=neut)
            d = pred.to_dict()
            d["match_id"] = row["match_id"]
            d["model_name"] = model_name
            d["group"] = row["group"]
            d["neutral"] = neut
            d["home_rating"] = round(rating_of(h), 1)
            d["away_rating"] = round(rating_of(a), 1)
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
        print(f"{p['match_id']}  [{p['model_name']}]  H {p['prob_home_win']:.3f}  "
              f"D {p['prob_draw']:.3f}  A {p['prob_away_win']:.3f}  "
              f"(xG {p['home_xg']:.2f}-{p['away_xg']:.2f})")

    if args.lock:
        from src.lockin.lock_predictions import lock_predictions
        lock_predictions(preds, args.matchday, kickoff_times=kickoffs)
    else:
        print("\nDry run. Add --lock to write the lock file and git-commit.")
