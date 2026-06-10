"""
Generate Phase 3 composition-model predictions for the 2026 WC group stage
and compare them, match by match, against the Elo baseline.

PROVISIONAL: the blend weight `lam` is not yet set by validation (Step 3 /
D009 gate). These predictions illustrate the player model's effect; they do
NOT enter the live scorecard until the gate is passed.

Run: python -m src.models.run_composition
"""

import pandas as pd

from src.data.results_loader import load_processed_results
from src.data.wc2026 import load_wc2026
from src.models.composition_model import (
    fit_strength_to_elo_bridge,
    predict_match_composition,
)
from src.models.dixon_coles import DixonColesParams
from src.models.elo import EloSystem
from src.models.player_strength import team_strength

OUT = "data/predictions/group_stage_composition_vs_baseline.csv"


def build_strengths_and_bridge(elo: EloSystem):
    df = pd.read_parquet("data/processed/unified_players.parquet")
    strengths = {t: team_strength(sq) for t, sq in df.groupby("team")}
    teams = list(strengths)
    bridge = fit_strength_to_elo_bridge(
        [strengths[t]["overall"] for t in teams],
        [elo.get_rating(t) for t in teams])
    return strengths, bridge


def main(lam: float = 0.5):
    results = load_processed_results()
    elo = EloSystem().fit_from_results(results)
    params = DixonColesParams.load()
    wc = load_wc2026(save=False)
    strengths, bridge = build_strengths_and_bridge(elo)

    rows = []
    for _, r in wc.fixtures.iterrows():
        h, a, neut = r["home_code"], r["away_code"], bool(r["neutral"])
        base = predict_match_composition(h, a, elo.get_rating(h), elo.get_rating(a),
                                         strengths[h], strengths[a], bridge, params,
                                         lam=0.0, neutral=neut)
        comp = predict_match_composition(h, a, elo.get_rating(h), elo.get_rating(a),
                                         strengths[h], strengths[a], bridge, params,
                                         lam=lam, neutral=neut)
        rows.append({
            "match": f"{h} v {a}", "group": r["group"],
            "base_home": round(base.prob_home_win, 3), "comp_home": round(comp.prob_home_win, 3),
            "base_draw": round(base.prob_draw, 3), "comp_draw": round(comp.prob_draw, 3),
            "base_away": round(base.prob_away_win, 3), "comp_away": round(comp.prob_away_win, 3),
            "home_shift": round(comp.prob_home_win - base.prob_home_win, 3),
            "conf_home": round(strengths[h]["confidence"], 2),
            "conf_away": round(strengths[a]["confidence"], 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    out = main()
    print(f"bridge + composition predictions for {len(out)} group matches (lam=0.5, PROVISIONAL)")
    big = out.reindex(out.home_shift.abs().sort_values(ascending=False).index).head(10)
    print("\nLargest shifts from baseline (player signal moving the needle):")
    print(big[["match", "base_home", "comp_home", "home_shift", "conf_home", "conf_away"]]
          .to_string(index=False))
    print(f"\nsaved {OUT}")
