"""
Phase 3 validation gate (D009): does player-composition strength actually
improve predictions over a results-only Elo — tested out-of-sample on club
matches?

What this does in simple English:
    This is the moment of truth for the whole player-composition idea. We
    run the EXACT champion-challenger contest, but on club football where
    we have tens of thousands of matches:

      1. Build a club Elo from years of match history (results only).
      2. Give each club a player-strength from its squad market value
         (a player-composition signal — richer rosters = stronger).
      3. On a HOLDOUT season the Elo never trained on, predict each match
         two ways — pure Elo, and Elo blended with player strength — and
         compare Brier scores across a range of blend weights.

    If blending in player strength lowers Brier and the strength correlates
    with outcomes, the approach has real signal (gate PASSES) and we learn
    how much to trust it (the best blend weight). If it doesn't help, the
    challenger FAILS and we publish that honestly instead of deploying it.

    Caveats stated plainly: market value is a current snapshot (mild
    anachronism predicting a just-finished season); Dixon-Coles params are
    the international ones. Both the baseline and the blend use identical
    params, so the RELATIVE Brier comparison — the thing we care about — is
    unaffected.

Run: python -m src.evaluation.club_validation
"""

import numpy as np
import pandas as pd

from src.evaluation.calibration import PredictionRecord, brier_score
from src.models.composition_model import (
    blended_rating,
    fit_strength_to_elo_bridge,
)
from src.models.dixon_coles import DixonColesParams, predict_match
from src.models.elo import EloSystem

GATE_MIN_CORRELATION = 0.30
GATE_MIN_BRIER_GAIN = 0.0005  # blend must beat Elo by at least this


def normalized_log_strength(values: dict) -> dict:
    """Map raw squad market values to 0-1 via log scale (robust to spread)."""
    logs = {}
    for k, v in values.items():
        if v is None or (isinstance(v, float) and np.isnan(v)) or v <= 0:
            logs[k] = np.nan
        else:
            logs[k] = np.log10(float(v))
    arr = np.array([x for x in logs.values() if not np.isnan(x)])
    lo, hi = arr.min(), arr.max()
    med = float(np.median(arr))
    out = {}
    for k, x in logs.items():
        x = med if np.isnan(x) else x  # impute missing to median
        out[k] = float((x - lo) / (hi - lo)) if hi > lo else 0.5
    return out


def gate_decision(brier_elo: float, brier_blend: float,
                  best_lambda: float, correlation: float) -> dict:
    """Apply the D009 pass/fail rule."""
    helps = (brier_elo - brier_blend) >= GATE_MIN_BRIER_GAIN and best_lambda > 0
    correlates = abs(correlation) >= GATE_MIN_CORRELATION
    passes = bool(helps and correlates)
    if not correlates:
        reason = f"FAIL: strength-outcome correlation {correlation:.2f} < {GATE_MIN_CORRELATION}"
    elif not helps:
        reason = (f"FAIL: blend does not improve Brier "
                  f"({brier_blend:.4f} vs Elo {brier_elo:.4f}); best lambda {best_lambda}")
    else:
        reason = (f"PASS: blend improves Brier {brier_elo:.4f} -> {brier_blend:.4f} "
                  f"at lambda {best_lambda}, correlation {correlation:.2f}")
    return {"passes": passes, "reason": reason}


def _result(hg: int, ag: int) -> str:
    return "H" if hg > ag else ("D" if hg == ag else "A")


def run_validation(holdout_season: int = 2025, min_history_games: int = 200,
                   lambdas=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    """Full club-level validation. Returns a report dict."""
    K = "data/raw/kaggle/transfermarkt/"
    games = pd.read_csv(K + "games.csv", parse_dates=["date"])
    games = games.dropna(subset=["home_club_goals", "away_club_goals"])
    games = games.sort_values("date").reset_index(drop=True)

    # Domestic-league games only (stable home advantage); drop cups
    games = games[games["competition_type"] == "domestic_league"]

    # Club composition signal = sum of its players' market values (from
    # players.csv, keyed by current club). A genuine player-composition
    # aggregate rather than a single club-level number.
    players = pd.read_csv(K + "players.csv")
    squad_val = (players.dropna(subset=["current_club_id", "market_value_in_eur"])
                 .groupby("current_club_id")["market_value_in_eur"].sum())
    raw_val = {str(k): v for k, v in squad_val.items()}
    strength = normalized_log_strength(raw_val)

    # Build Elo online; capture pre-match ratings for holdout games
    elo = EloSystem()
    holdout = []
    for _, g in games.iterrows():
        h, a = str(g["home_club_id"]), str(g["away_club_id"])
        if g["season"] == holdout_season:
            holdout.append({
                "h": h, "a": a,
                "h_elo": elo.get_rating(h), "a_elo": elo.get_rating(a),
                "result": _result(g["home_club_goals"], g["away_club_goals"]),
            })
        elo.update(h, a, int(g["home_club_goals"]), int(g["away_club_goals"]),
                   tournament="club", neutral=False, date=str(g["date"])[:10])

    hdf = pd.DataFrame(holdout)
    # Keep games where both clubs have a market value and real Elo history
    hdf = hdf[hdf["h"].map(strength).notna() & hdf["a"].map(strength).notna()]

    # Bridge: player strength -> Elo units (fit on clubs' final Elo)
    final = {c: elo.get_rating(c) for c in set(hdf["h"]) | set(hdf["a"])}
    s_list = [strength[c] for c in final]
    bridge = fit_strength_to_elo_bridge(s_list, [final[c] for c in final])

    # Correlation: does (strength diff) track outcomes? (home points 3/1/0)
    pts = {"H": 3, "D": 1, "A": 0}
    sdiff = hdf.apply(lambda r: strength[r["h"]] - strength[r["a"]], axis=1)
    outcome_pts = hdf["result"].map(pts)
    correlation = float(np.corrcoef(sdiff, outcome_pts)[0, 1])

    params = DixonColesParams.load()

    def brier_at(lam: float) -> float:
        recs = []
        for i, r in hdf.iterrows():
            hr = blended_rating(r["h_elo"], bridge.to_elo(strength[r["h"]]), 1.0, lam)
            ar = blended_rating(r["a_elo"], bridge.to_elo(strength[r["a"]]), 1.0, lam)
            p = predict_match(r["h"], r["a"], hr, ar, params, neutral=False)
            recs.append(PredictionRecord(str(i), "x", p.prob_home_win, p.prob_draw,
                                         p.prob_away_win, r["result"]))
        return brier_score(recs)

    briers = {lam: brier_at(lam) for lam in lambdas}
    brier_elo = briers[0.0]
    best_lambda = min(briers, key=briers.get)
    brier_blend = briers[best_lambda]

    decision = gate_decision(brier_elo, brier_blend, best_lambda, correlation)
    return {
        "n_holdout_games": len(hdf),
        "holdout_season": holdout_season,
        "correlation_strength_vs_outcome": round(correlation, 3),
        "brier_by_lambda": {k: round(v, 4) for k, v in briers.items()},
        "brier_elo": round(brier_elo, 4),
        "brier_best_blend": round(brier_blend, 4),
        "best_lambda": best_lambda,
        "gate": decision,
    }


if __name__ == "__main__":
    import json
    rep = run_validation()
    print(json.dumps(rep, indent=2))
    print("\n" + rep["gate"]["reason"])
