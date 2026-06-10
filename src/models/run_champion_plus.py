"""
Champion+ : the composition champion with three free edges layered on —
ensemble Elo, altitude, and rest/travel (D024).

Pipeline per team per match:
    base   = ensemble(our Elo, independent Elo)        # variance reduction
    blend  = base + lam*confidence*(player_elo - base) # composition (D022)
    rating = blend + altitude_adj + rest_travel_adj    # match context

Then the usual Dixon-Coles turns the two ratings into probabilities.

Run: python -m src.models.run_champion_plus
"""

from pathlib import Path

import pandas as pd

from src.data.elo_ensemble import align_and_ensemble, load_independent_ratings
from src.data.results_loader import load_processed_results
from src.data.wc2026 import load_wc2026
from src.models.champion_model import CHAMPION_LAMBDA, load_wc_strengths_and_bridge
from src.models.composition_model import blended_rating
from src.models.dixon_coles import DixonColesParams, predict_match
from src.models.elo import EloSystem
from src.models.match_context import (
    HOST_CITIES,
    altitude_elo_adjustment,
    haversine_km,
    rest_travel_elo_adjustment,
)

OUT = Path("data/predictions/group_stage_champion_plus.csv")


def _schedule_context(fixtures: pd.DataFrame) -> dict:
    """Per (match_id, team): rest_days + travel_km since the team's last game."""
    rows = []
    for _, r in fixtures.iterrows():
        for code in (r["home_code"], r["away_code"]):
            rows.append({"team": code, "date": r["date"], "city": r["city"],
                         "match_id": r["match_id"]})
    sched = pd.DataFrame(rows).sort_values(["team", "date"])
    ctx = {}
    for team, g in sched.groupby("team"):
        prev_date, prev_city = None, None
        for _, m in g.iterrows():
            if prev_date is None:
                rest, travel = 5.0, 0.0   # arrival baseline for first match
            else:
                rest = (m["date"] - prev_date).days
                c0, c1 = HOST_CITIES.get(prev_city), HOST_CITIES.get(m["city"])
                travel = (haversine_km(c0["lat"], c0["lon"], c1["lat"], c1["lon"])
                          if c0 and c1 else 0.0)
            ctx[(m["match_id"], team)] = {"rest": rest, "travel": travel}
            prev_date, prev_city = m["date"], m["city"]
    return ctx


def build_base_ratings(elo: EloSystem) -> dict:
    """Ensembled base rating per WC team (our Elo + independent)."""
    wc = load_wc2026(save=False)
    teams = [t for ts in wc.groups.values() for t in ts]
    ours = {t: elo.get_rating(t) for t in teams}
    return align_and_ensemble(ours, load_independent_ratings())


def main(lam: float = CHAMPION_LAMBDA):
    elo = EloSystem().fit_from_results(load_processed_results())
    params = DixonColesParams.load()
    wc = load_wc2026(save=False)
    strengths, _ = load_wc_strengths_and_bridge(elo)
    base = build_base_ratings(elo)
    # Bridge refit against the ensembled base so player_elo lives on the same scale
    from src.models.composition_model import fit_strength_to_elo_bridge
    bridge = fit_strength_to_elo_bridge(
        [strengths[t]["overall"] for t in base], [base[t] for t in base])
    ctx = _schedule_context(wc.fixtures)

    def rating_for(team, opp, city, match_id):
        blended = blended_rating(base[team], bridge.to_elo(strengths[team]["overall"]),
                                 strengths[team]["confidence"], lam)
        alt = altitude_elo_adjustment(team, city)
        me, op = ctx[(match_id, team)], ctx[(match_id, opp)]
        rt = rest_travel_elo_adjustment(me["rest"], me["travel"], op["rest"], op["travel"])
        return blended + alt + rt, alt, rt

    rows = []
    for _, r in wc.fixtures.iterrows():
        h, a, city, mid, neut = (r["home_code"], r["away_code"], r["city"],
                                 r["match_id"], bool(r["neutral"]))
        hr, h_alt, h_rt = rating_for(h, a, city, mid)
        ar, a_alt, a_rt = rating_for(a, h, city, mid)
        p = predict_match(h, a, hr, ar, params, neutral=neut)
        rows.append({
            "date": r["date"].strftime("%Y-%m-%d"), "group": r["group"],
            "match": f"{h} v {a}", "city": city,
            "p_home": round(p.prob_home_win, 3), "p_draw": round(p.prob_draw, 3),
            "p_away": round(p.prob_away_win, 3),
            "xg": f"{p.home_expected_goals:.2f}-{p.away_expected_goals:.2f}",
            "likely": f"{p.most_likely_score[0]}-{p.most_likely_score[1]}",
            "alt_adj": round(h_alt + a_alt, 0),  # nonzero only at high venues
            "conf": round(min(strengths[h]["confidence"], strengths[a]["confidence"]), 2),
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    df = main()
    print(f"Champion+ predictions for {len(df)} group games saved.")
    alt_games = df[df["alt_adj"] != 0]
    print(f"\nAltitude-affected games ({len(alt_games)}):")
    print(alt_games[["date", "match", "city", "p_home", "p_draw", "p_away", "alt_adj"]]
          .to_string(index=False))
    print(f"\navg draw prob: {df.p_draw.mean():.3f}")
