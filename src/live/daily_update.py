"""
Daily live update: ingest real scores, re-simulate, score, refresh the page.

What this does in simple English (run once a day during the tournament):
    1. Re-download the results file — played games now have real scores.
    2. Re-fit Elo including those results.
    3. Re-run the 20,000-tournament simulation, but with completed group
       games FIXED to their real outcome, so the title odds sharpen.
    4. Save today's odds snapshot and work out the MOVEMENT vs yesterday.
    5. Grade every locked prediction against what actually happened — the
       running accuracy scorecard (our models, and the market if odds exist).
    6. Rebuild and (optionally) push the shareable page.

Run: python -m src.live.daily_update          # update + rebuild page
     python -m src.live.daily_update --push    # also git-commit & push
"""

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.results_loader import build_results_dataset, RAW_PATH
from src.data.wc2026 import load_wc2026
from src.live.scorecard import match_brier, movement_vs_previous, running_brier
from src.models.champion_model import load_wc_strengths_and_bridge
from src.models.composition_model import blended_rating, fit_strength_to_elo_bridge
from src.models.champion_model import CHAMPION_LAMBDA
from src.models.dixon_coles import DixonColesParams, predict_match
from src.models.elo import EloSystem
from src.models.run_champion_plus import build_base_ratings
from src.simulation.tournament import SimulationConfig, run_simulation

PRED = Path("data/predictions")
LOCKED = PRED / "locked_predictions.csv"          # all models, all 72 games
HISTORY = PRED / "champion_history.csv"           # daily odds snapshots
SCORECARD = PRED / "scorecard.json"               # running accuracy
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def refresh_results_file() -> None:
    """Re-download the results CSV so newly-played games carry real scores.

    Without this, the pipeline would keep reading yesterday's cached file
    and never see new results.
    """
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(RESULTS_URL, RAW_PATH)
    except Exception as e:  # offline / source down: fall back to cache
        print(f"WARNING: could not refresh results ({e}); using cached file.")


def generate_locked_predictions() -> pd.DataFrame:
    """Produce the committed pre-tournament prediction file (all 3 models)."""
    elo = EloSystem().fit_from_results(build_results_dataset())
    params = DixonColesParams.load()
    wc = load_wc2026(save=False)
    strengths, _ = load_wc_strengths_and_bridge(elo)
    base = build_base_ratings(elo)
    from src.models.run_champion_plus import _schedule_context
    from src.models.match_context import altitude_elo_adjustment, rest_travel_elo_adjustment
    bridge = fit_strength_to_elo_bridge([strengths[t]["overall"] for t in base],
                                        [base[t] for t in base])
    ctx = _schedule_context(wc.fixtures)

    rows = []
    for _, r in wc.fixtures.iterrows():
        h, a, city, mid, neut = (r["home_code"], r["away_code"], r["city"],
                                 r["match_id"], bool(r["neutral"]))

        def champ_plus_rating(team, opp):
            bl = blended_rating(base[team], bridge.to_elo(strengths[team]["overall"]),
                                strengths[team]["confidence"], CHAMPION_LAMBDA)
            me, op = ctx[(mid, team)], ctx[(mid, opp)]
            return (bl + altitude_elo_adjustment(team, city)
                    + rest_travel_elo_adjustment(me["rest"], me["travel"],
                                                 op["rest"], op["travel"]))

        models = {
            "champion_plus": (champ_plus_rating(h, a), champ_plus_rating(a, h)),
            "baseline_elo": (elo.get_rating(h), elo.get_rating(a)),
        }
        row = {"match_id": mid, "home": h, "away": a}
        for name, (hr, ar) in models.items():
            p = predict_match(h, a, hr, ar, params, neutral=neut)
            row[f"{name}_H"] = round(p.prob_home_win, 4)
            row[f"{name}_D"] = round(p.prob_draw, 4)
            row[f"{name}_A"] = round(p.prob_away_win, 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    PRED.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCKED, index=False)
    return df


def _completed_group_games(results: pd.DataFrame, wc) -> tuple[dict, dict]:
    """Return (completed_for_sim, results_by_match_id) for played group games."""
    played = results[(results["tournament"] == "FIFA World Cup")
                     & (results["date"].dt.year == 2026)
                     & results["home_score"].notna()]
    fixtures = {r["match_id"]: r for _, r in wc.fixtures.iterrows()}
    sim_completed, by_id = {}, {}
    for _, m in played.iterrows():
        h, a = m["home_code"], m["away_code"]
        # only group games are in wc.fixtures (knockouts handled later)
        if m["match_id"] not in fixtures:
            continue
        hg, ag = int(m["home_score"]), int(m["away_score"])
        sim_completed[frozenset({h, a})] = {h: hg, a: ag}
        by_id[m["match_id"]] = "H" if hg > ag else ("D" if hg == ag else "A")
    return sim_completed, by_id


def update(n_sims: int = 20000) -> dict:
    if not LOCKED.exists():
        generate_locked_predictions()
    locked = pd.read_csv(LOCKED)

    # 1-2. refresh results (re-download) + Elo
    refresh_results_file()
    results = build_results_dataset()
    wc = load_wc2026(save=False)
    elo = EloSystem().fit_from_results(results)
    params = DixonColesParams.load()
    completed_sim, actual_by_id = _completed_group_games(results, wc)

    # 3. re-simulate with completed games fixed (champion+ static base ratings)
    strengths, _ = load_wc_strengths_and_bridge(elo)
    base = build_base_ratings(elo)
    bridge = fit_strength_to_elo_bridge([strengths[t]["overall"] for t in base],
                                        [base[t] for t in base])
    champ = {t: blended_rating(base[t], bridge.to_elo(strengths[t]["overall"]),
                               strengths[t]["confidence"], CHAMPION_LAMBDA) for t in base}
    sim = run_simulation(wc.groups, champ, host_teams={"USA", "CAN", "MEX"},
                         params=params, config=SimulationConfig(n_sims=n_sims),
                         completed=completed_sim)
    today_probs = dict(zip(sim["team"], sim["p_champion"]))

    # 4. movement vs previous snapshot
    prev = {}
    if HISTORY.exists():
        h = pd.read_csv(HISTORY)
        last = h[h["date"] == h["date"].max()]
        prev = dict(zip(last["team"], last["p_champion"]))
    movement = movement_vs_previous(today_probs, prev)
    snap = pd.DataFrame({"date": _today(), "team": list(today_probs),
                         "p_champion": list(today_probs.values())})
    if HISTORY.exists() and _today() not in set(pd.read_csv(HISTORY)["date"]):
        snap.to_csv(HISTORY, mode="a", header=False, index=False)
    elif not HISTORY.exists():
        snap.to_csv(HISTORY, index=False)

    # 5. score locked predictions vs actual results
    models = sorted({c.rsplit("_", 1)[0] for c in locked.columns if c.endswith(("_H",))})
    scores = {}
    for mdl in models:
        pairs = []
        for _, r in locked.iterrows():
            if r["match_id"] in actual_by_id:
                probs = {"H": r[f"{mdl}_H"], "D": r[f"{mdl}_D"], "A": r[f"{mdl}_A"]}
                pairs.append((probs, actual_by_id[r["match_id"]]))
        scores[mdl] = {"n": len(pairs), "brier": running_brier(pairs)}

    scorecard = {"updated": _today(), "n_completed": len(actual_by_id), "models": scores}
    SCORECARD.write_text(json.dumps(scorecard, indent=2))

    return {"today_probs": today_probs, "movement": movement,
            "scorecard": scorecard, "n_completed": len(actual_by_id)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="git commit + push the refreshed page")
    ap.add_argument("--sims", type=int, default=20000)
    args = ap.parse_args()

    out = update(n_sims=args.sims)
    print(f"Updated {out['n_completed']} completed games.")
    if out["scorecard"]["models"]:
        for mdl, s in out["scorecard"]["models"].items():
            b = f"{s['brier']:.4f}" if s["brier"] is not None else "—"
            print(f"  {mdl}: Brier {b} over {s['n']} games")

    from src.dashboard.build_react_page import main as build_page
    build_page()
    print("Page rebuilt.")

    if args.push:
        import subprocess
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", f"daily update {out['n_completed']} games"], check=False)
        subprocess.run(["git", "push", "-q", "origin", "main"], check=False)
        print("Pushed.")
