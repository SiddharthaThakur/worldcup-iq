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
ADVANCEMENT = PRED / "advancement.csv"            # per-round probs (for the bracket funnel)
GOALS_TABLE = PRED / "goals_table.csv"            # xG for/against per team (group + knockout rate)
BRACKET = PRED / "bracket.json"                   # R32 slots with top-3 likely occupants
PREDSCORES = PRED / "predicted_scores.csv"        # frozen pre-match predicted scoreline per game
MODEL_PREDS = PRED / "model_predictions.csv"      # all 3 models on a consistent frozen basis
TOURNAMENT_PARAMS = Path("models/dixon_coles_tournament_params.json")
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
TOURNAMENT_OVER_AFTER = "2026-07-26"  # a week past the final (catches late result fixes)


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


def build_model_predictions() -> pd.DataFrame:
    """All three models' predictions on ONE consistent frozen basis.

    Same pre-WC ratings for every model (Elo fit excluding the 2026 WC), so
    the comparison isolates each model's difference:
      - champion_plus      : composition blend, all-internationals goal model
      - champion_tournament : same ratings, goal model fitted on TOURNAMENTS
                              (steeper margins, no calibration fudge)
      - baseline_elo       : pure Elo, all-internationals goal model
    Frozen + reproducible. Stores H/D/A probs and predicted goals per model.
    """
    from src.models.run_champion_plus import _schedule_context
    from src.models.match_context import altitude_elo_adjustment, rest_travel_elo_adjustment

    res = build_results_dataset()
    pre = res[~((res["tournament"] == "FIFA World Cup") & (res["date"].dt.year == 2026))]
    elo = EloSystem().fit_from_results(pre)
    params = DixonColesParams.load()
    tour_params = DixonColesParams.load(TOURNAMENT_PARAMS)
    wc = load_wc2026(save=False)
    strengths, _ = load_wc_strengths_and_bridge(elo)
    base = build_base_ratings(elo)
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

        cr_h, cr_a = champ_plus_rating(h, a), champ_plus_rating(a, h)
        preds = {
            "champion_plus": predict_match(h, a, cr_h, cr_a, params, neutral=neut),
            "champion_tournament": predict_match(h, a, cr_h, cr_a, tour_params,
                                                 neutral=neut, goal_scale=1.0),
            "baseline_elo": predict_match(h, a, elo.get_rating(h), elo.get_rating(a),
                                          params, neutral=neut),
        }
        row = {"match_id": mid, "home": h, "away": a}
        for name, p in preds.items():
            row[f"{name}_H"] = round(p.prob_home_win, 4)
            row[f"{name}_D"] = round(p.prob_draw, 4)
            row[f"{name}_A"] = round(p.prob_away_win, 4)
            row[f"{name}_ph"] = round(p.home_expected_goals)
            row[f"{name}_pa"] = round(p.away_expected_goals)
        rows.append(row)
    df = pd.DataFrame(rows)
    PRED.mkdir(parents=True, exist_ok=True)
    df.to_csv(MODEL_PREDS, index=False)
    return df


def build_predicted_scores() -> pd.DataFrame:
    """Frozen pre-match predicted scoreline (champion+) for all 72 games.

    Elo is fit EXCLUDING the 2026 World Cup, so this is reproducible and
    matches the all-pre-tournament locking philosophy (the same view the
    locked predictions were made from). Generated once, then frozen.
    """
    from src.models.run_champion_plus import _schedule_context
    from src.models.match_context import altitude_elo_adjustment, rest_travel_elo_adjustment

    res = build_results_dataset()
    pre = res[~((res["tournament"] == "FIFA World Cup") & (res["date"].dt.year == 2026))]
    elo = EloSystem().fit_from_results(pre)
    params = DixonColesParams.load()
    wc = load_wc2026(save=False)
    strengths, _ = load_wc_strengths_and_bridge(elo)
    base = build_base_ratings(elo)
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

        p = predict_match(h, a, champ_plus_rating(h, a), champ_plus_rating(a, h),
                          params, neutral=neut)
        rows.append({"match_id": mid, "pred_h": round(p.home_expected_goals),
                     "pred_a": round(p.away_expected_goals)})
    df = pd.DataFrame(rows)
    PRED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDSCORES, index=False)
    return df


def build_goals_table(results: pd.DataFrame, wc, champ_ratings: dict,
                      params) -> pd.DataFrame:
    """Per-team xG: group-stage totals (+ actual once played) and a
    per-knockout-match rate vs a typical qualifier."""
    from collections import defaultdict
    from src.models.dixon_coles import strengths_to_expected_goals

    live = pd.read_csv(PRED / "group_stage_champion_plus.csv")
    team_group = {t: g for g, ts in wc.groups.items() for t in ts}

    gxf, gxa = defaultdict(float), defaultdict(float)
    for _, r in live.iterrows():
        h, a = str(r["match"]).split(" v ")
        xh, xa = map(float, str(r["xg"]).split("-"))
        gxf[h] += xh; gxa[h] += xa
        gxf[a] += xa; gxa[a] += xh

    # actual goals in played group games
    af, aa, played_n = defaultdict(int), defaultdict(int), defaultdict(int)
    grp_played = results[(results["tournament"] == "FIFA World Cup")
                         & (results["date"].dt.year == 2026)
                         & results["home_score"].notna()]
    fixt_ids = set(wc.fixtures["match_id"])
    for _, m in grp_played.iterrows():
        if m["match_id"] not in fixt_ids:
            continue
        h, a = m["home_code"], m["away_code"]
        hg, ag = int(m["home_score"]), int(m["away_score"])
        af[h] += hg; aa[h] += ag; played_n[h] += 1
        af[a] += ag; aa[a] += hg; played_n[a] += 1

    # knockout per-match rate vs a TYPICAL QUALIFIER: the average rating of
    # the ~32 teams that reach the knockouts (stronger than the full field,
    # since the weakest 16 are eliminated).
    top32 = sorted(champ_ratings.values(), reverse=True)[:32]
    avg_opp = sum(top32) / len(top32)

    rows = []
    for team in team_group:
        ko_f, ko_a = strengths_to_expected_goals(
            champ_ratings.get(team, avg_opp), avg_opp, params, neutral=True)
        rows.append({
            "team": team, "group": team_group[team],
            "group_xgf": round(gxf[team], 1), "group_xga": round(gxa[team], 1),
            "group_xgd": round(gxf[team] - gxa[team], 1),
            "group_actual_f": af[team], "group_actual_a": aa[team],
            "group_played": played_n[team],
            "ko_xgf": round(ko_f, 2), "ko_xga": round(ko_a, 2),
        })
    df = pd.DataFrame(rows).sort_values("group_xgd", ascending=False)
    df.to_csv(GOALS_TABLE, index=False)
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


def _completed_knockout_games(results: pd.DataFrame) -> dict[frozenset, str]:
    """Return {frozenset({team_a, team_b}): winner_code} for played knockout games.

    Knockout games can't draw (ET/pens decide), so the team with more goals
    wins. If scores are level the data source records the penalty winner as
    having an extra goal.
    """
    from src.data.team_aliases import resolve_team_code
    from src.data.wc2026 import OFFICIAL_GROUPS

    team_to_group = {t: g for g, teams in OFFICIAL_GROUPS.items() for t in teams}
    played = results[(results["tournament"] == "FIFA World Cup")
                     & (results["date"].dt.year == 2026)
                     & results["home_score"].notna()]
    ko_results = {}
    for _, m in played.iterrows():
        h = resolve_team_code(m["home_team"])
        a = resolve_team_code(m["away_team"])
        if team_to_group.get(h) == team_to_group.get(a):
            continue  # group game, not knockout
        hg, ag = int(m["home_score"]), int(m["away_score"])
        winner = h if hg > ag else a
        ko_results[frozenset({h, a})] = winner
    return ko_results


def _apply_knockout_results(bracket: list[dict], ko_results: dict[frozenset, str]):
    """Update bracket matches with actual knockout results.

    For completed matches, sets pAdv to {winner: 1.0} and candidate lists
    to just the winner at p=1.0.
    """
    for rnd in bracket:
        for m in rnd["matches"]:
            teams_a = {c["team"] for c in m.get("a", [])}
            teams_b = {c["team"] for c in m.get("b", [])}
            all_teams = teams_a | teams_b
            for matchup, winner in ko_results.items():
                if matchup <= all_teams and len(matchup & teams_a) == 1 and len(matchup & teams_b) == 1:
                    m["pAdv"] = {winner: 1.0}
                    m["a"] = [{"team": winner, "p": 1.0}] if winner in teams_a else [{"team": list(matchup & teams_a)[0], "p": 0.0}]
                    m["b"] = [{"team": winner, "p": 1.0}] if winner in teams_b else [{"team": list(matchup & teams_b)[0], "p": 0.0}]
                    # Simplify: only keep the winner
                    m["a"] = [c for c in m["a"] if c["team"] == winner]
                    m["b"] = [c for c in m["b"] if c["team"] == winner]
                    break


def update(n_sims: int = 20000) -> dict:
    if _today() > TOURNAMENT_OVER_AFTER:
        print(f"The 2026 World Cup is over (after {TOURNAMENT_OVER_AFTER}); "
              "daily updates have stopped. Final state is preserved.")
        return {"stopped": True, "n_completed": None}
    if not LOCKED.exists():
        generate_locked_predictions()
    if not PREDSCORES.exists():
        build_predicted_scores()
    if not MODEL_PREDS.exists():
        build_model_predictions()
    locked = pd.read_csv(MODEL_PREDS)  # 3-model consistent comparison basis

    # 1-2. refresh results (re-download) + Elo
    refresh_results_file()
    results = build_results_dataset()

    # Regenerate LIVE match predictions for upcoming games — these update
    # daily as teams play (a team's later games shift once its rating moves).
    # The pre-tournament LOCKED predictions stay frozen for honest scoring.
    from src.models.run_champion_plus import main as regen_live_predictions
    regen_live_predictions()
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
    # Full per-round probabilities for the bracket funnel (reflects games so far)
    sim[["team", "p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]].to_csv(
        ADVANCEMENT, index=False)
    build_goals_table(results, wc, champ, params)

    # Bracket: project the Round of 32 (top-3 per slot); later rounds are
    # placeholders that fill in as feeder games are decided.
    from src.simulation.bracket import build_progressive_bracket, simulate_positions
    positions = simulate_positions(wc.groups, champ, {"USA", "CAN", "MEX"}, params,
                                   n_sims=min(n_sims, 20000), completed=completed_sim)
    bracket = build_progressive_bracket(positions, wc.groups)
    from src.simulation.bracket import annotate_knockout_probs
    ko_results = _completed_knockout_games(results)
    annotate_knockout_probs(bracket, champ, {"USA", "CAN", "MEX"}, params,
                            ko_results=ko_results)
    BRACKET.write_text(json.dumps(bracket))

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

    # Merge in the frozen offline Bayesian predictions (precomputed; the daily
    # run needs no PyMC). Drop tournament-fit from the scorecard.
    bayes_path = PRED / "bayesian_predictions.csv"
    if bayes_path.exists():
        locked = locked.merge(pd.read_csv(bayes_path), on="match_id", how="left")

    # 5. score locked predictions vs actual results
    models = [m for m in ("champion_plus", "champion_bayesian", "baseline_elo")
              if f"{m}_H" in locked.columns]
    scores = {}
    for mdl in models:
        pairs = []
        for _, r in locked.iterrows():
            if r["match_id"] in actual_by_id and pd.notna(r.get(f"{mdl}_H")):
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
    if out.get("stopped"):
        raise SystemExit(0)

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
