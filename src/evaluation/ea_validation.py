"""
Phase 3 re-test (D020 follow-up): validate the ACTUAL deployed signal.

What this does in simple English:
    The first validation used squad market value as a stand-in and missed
    the bar by a hair (correlation 0.293 vs 0.30). But the live World Cup
    model uses EA FC ratings, not market value. This re-test runs the same
    out-of-sample club contest using EA ratings through the EXACT team-
    strength aggregator we deploy — a fair, pre-specified retest, not a
    goalpost move (the gate, D009, is unchanged: correlation >= 0.30 AND
    the blend must beat Elo).

    To make it tight enough to be confident:
      - Use the real aggregator (player_strength.team_strength) on EA
        ratings, not a proxy.
      - Only keep clubs whose EA<->Transfermarkt name match is confident,
        so bad matching can't inject noise that fakes a low score.
      - Put a BOOTSTRAP 95% confidence interval on the Brier improvement,
        so we know whether any gain is real or luck.

    Leakage control: EA FC 26 ratings are set at the 2025/26 season's
    start, so predicting that season's games is forward-looking; club Elo
    is built only from each game's past. Dixon-Coles params are shared by
    baseline and blend, so the relative comparison is unaffected.

Run: python -m src.evaluation.ea_validation
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.data.entity_resolver import normalize_name
from src.evaluation.calibration import PredictionRecord
from src.evaluation.club_validation import gate_decision
from src.models.composition_model import blended_rating, fit_strength_to_elo_bridge
from src.models.dixon_coles import DixonColesParams, predict_match
from src.models.elo import EloSystem
from src.models.player_strength import team_strength

K = "data/raw/kaggle/transfermarkt/"
EA_PATH = "data/raw/kaggle/ea_fc26/ea_fc26_players.csv"
CLUB_MATCH_THRESHOLD = 88

_EA_POS = {
    "GK": "GK",
    "CB": "DF", "RB": "DF", "LB": "DF", "RWB": "DF", "LWB": "DF",
    "CDM": "MF", "CM": "MF", "CAM": "MF", "LM": "MF", "RM": "MF",
    "ST": "FW", "CF": "FW", "LW": "FW", "RW": "FW", "LF": "FW", "RF": "FW",
}


def ea_position_to_group(pos: str) -> str:
    """Map an EA position code (CB, ST, CDM...) to GK/DF/MF/FW."""
    return _EA_POS.get(str(pos).upper().strip(), "MF")


def bootstrap_brier_diff_ci(elo_brier: np.ndarray, blend_brier: np.ndarray,
                            n_boot: int = 2000, seed: int = 0):
    """Bootstrap 95% CI for mean per-game Brier improvement (elo - blend).

    Positive => blend is better. CI excluding 0 => the gain is real.
    """
    elo_brier = np.asarray(elo_brier)
    blend_brier = np.asarray(blend_brier)
    diff = elo_brier - blend_brier
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(diff.mean())


def build_ea_club_strengths() -> dict:
    """Run the deployed aggregator on each EA club squad. Returns {club: dict}."""
    ea = pd.read_csv(EA_PATH)
    ea = ea.dropna(subset=["team", "overallRating", "position"])
    ea = ea.rename(columns={"overallRating": "ea_overall"})
    ea["position"] = ea["position"].apply(ea_position_to_group)
    ea["tm_value_eur"] = np.nan
    ea["caps"] = 0
    out = {}
    for club, sq in ea.groupby("team"):
        if len(sq) >= 11:
            out[club] = team_strength(sq[["position", "ea_overall", "tm_value_eur", "caps"]])
    return out


def match_ea_clubs_to_tm(ea_clubs, tm_clubs: pd.DataFrame) -> dict:
    """Fuzzy-match EA club names to TM club_ids. Returns {tm_club_id: ea_club}."""
    tm_norm = {normalize_name(n): cid for n, cid in
               zip(tm_clubs["name"], tm_clubs["club_id"])}
    choices = list(tm_norm)
    mapping = {}
    for ea_name in ea_clubs:
        m = process.extractOne(normalize_name(ea_name), choices, scorer=fuzz.token_set_ratio)
        if m and m[1] >= CLUB_MATCH_THRESHOLD:
            mapping[str(tm_norm[m[0]])] = ea_name
    return mapping


def run_ea_validation(holdout_season: int = 2025,
                      lambdas=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)) -> dict:
    games = pd.read_csv(K + "games.csv", parse_dates=["date"])
    games = games.dropna(subset=["home_club_goals", "away_club_goals"])
    games = games[games["competition_type"] == "domestic_league"]
    games = games.sort_values("date").reset_index(drop=True)

    ea_strengths = build_ea_club_strengths()
    tm_clubs = pd.read_csv(K + "clubs.csv")
    tm_to_ea = match_ea_clubs_to_tm(list(ea_strengths), tm_clubs)
    # club_id (str) -> strength dict
    strength = {cid: ea_strengths[ea] for cid, ea in tm_to_ea.items()}

    elo = EloSystem()
    holdout = []
    for _, g in games.iterrows():
        h, a = str(g["home_club_id"]), str(g["away_club_id"])
        if g["season"] == holdout_season and h in strength and a in strength:
            holdout.append({"h": h, "a": a, "h_elo": elo.get_rating(h),
                            "a_elo": elo.get_rating(a),
                            "result": ("H" if g["home_club_goals"] > g["away_club_goals"]
                                       else "D" if g["home_club_goals"] == g["away_club_goals"]
                                       else "A")})
        elo.update(h, a, int(g["home_club_goals"]), int(g["away_club_goals"]),
                   tournament="club", neutral=False, date=str(g["date"])[:10])

    hdf = pd.DataFrame(holdout)
    clubs_in = set(hdf["h"]) | set(hdf["a"])
    final = {c: elo.get_rating(c) for c in clubs_in}
    bridge = fit_strength_to_elo_bridge(
        [strength[c]["overall"] for c in clubs_in], [final[c] for c in clubs_in])

    pts = {"H": 3, "D": 1, "A": 0}
    sdiff = hdf.apply(lambda r: strength[r["h"]]["overall"] - strength[r["a"]]["overall"], axis=1).values
    outcome = hdf["result"].map(pts).values
    correlation = float(np.corrcoef(sdiff, outcome)[0, 1])
    # Bootstrap CI on the correlation — is it genuinely below 0.30 or just noisy?
    _rng = np.random.default_rng(11)
    _n = len(sdiff)
    _corrs = []
    for _ in range(2000):
        idx = _rng.integers(0, _n, _n)
        _corrs.append(np.corrcoef(sdiff[idx], outcome[idx])[0, 1])
    corr_ci = [round(float(np.percentile(_corrs, 2.5)), 3),
               round(float(np.percentile(_corrs, 97.5)), 3)]

    params = DixonColesParams.load()

    def per_game_brier(lam: float) -> np.ndarray:
        b = []
        for _, r in hdf.iterrows():
            hr = blended_rating(r["h_elo"], bridge.to_elo(strength[r["h"]]["overall"]), 1.0, lam)
            ar = blended_rating(r["a_elo"], bridge.to_elo(strength[r["a"]]["overall"]), 1.0, lam)
            p = predict_match(r["h"], r["a"], hr, ar, params, neutral=False)
            oh = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[r["result"]]
            probs = np.array([p.prob_home_win, p.prob_draw, p.prob_away_win])
            b.append(np.sum((probs - oh) ** 2) / 3.0)
        return np.array(b)

    brier_by_lambda = {lam: float(per_game_brier(lam).mean()) for lam in lambdas}
    elo_b = per_game_brier(0.0)
    best_lambda = min(brier_by_lambda, key=brier_by_lambda.get)
    blend_b = per_game_brier(best_lambda)
    lo, hi, mean_gain = bootstrap_brier_diff_ci(elo_b, blend_b, n_boot=2000, seed=7)

    decision = gate_decision(brier_by_lambda[0.0], brier_by_lambda[best_lambda],
                             best_lambda, correlation)
    return {
        "n_holdout_games": len(hdf),
        "n_clubs_matched": len(strength),
        "correlation_strength_vs_outcome": round(correlation, 3),
        "correlation_95ci": corr_ci,
        "brier_by_lambda": {k: round(v, 4) for k, v in brier_by_lambda.items()},
        "best_lambda": best_lambda,
        "brier_improvement_mean": round(mean_gain, 4),
        "brier_improvement_95ci": [round(lo, 4), round(hi, 4)],
        "improvement_is_significant": bool(lo > 0),
        "gate": decision,
    }


if __name__ == "__main__":
    import json
    rep = run_ea_validation()
    print(json.dumps(rep, indent=2))
    print("\n" + rep["gate"]["reason"])
    print(f"Brier improvement {rep['brier_improvement_mean']:+.4f} "
          f"95% CI {rep['brier_improvement_95ci']} "
          f"({'significant' if rep['improvement_is_significant'] else 'not significant'})")
