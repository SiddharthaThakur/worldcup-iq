"""
Align bookmaker odds with our match results for honest scoring.

What this does in simple English:
    Our results data and the odds workbook sometimes disagree about which
    team is listed as 'home' (FIFA's official designations vs the
    bookmaker's listing). To compare model vs market on the same matches,
    we join on the date + the PAIR of teams, and if the orientation is
    flipped we swap the market's home/away probabilities and mirror the
    result. Unmatched matches are reported, never silently dropped.
"""

import pandas as pd


def align_market_to_results(results: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Align odds rows to results rows on (date, unordered team pair).

    Args:
        results: must have match_id, date, home_code, away_code
        odds: must have date (YYYY-MM-DD str), home_code, away_code,
              prob_home, prob_draw, prob_away, result_90

    Returns:
        DataFrame in the RESULTS orientation with columns: match_id,
        prob_home, prob_draw, prob_away, result_90. The list of result
        match_ids that found no odds is stored in .attrs["unmatched_match_ids"].
    """
    odds_by_key = {}
    for _, r in odds.iterrows():
        key = (str(r["date"])[:10], frozenset((r["home_code"], r["away_code"])))
        odds_by_key[key] = r

    rows, unmatched = [], []
    for _, m in results.iterrows():
        key = (str(m["date"])[:10], frozenset((m["home_code"], m["away_code"])))
        r = odds_by_key.get(key)
        if r is None:
            unmatched.append(m["match_id"])
            continue
        same_orientation = r["home_code"] == m["home_code"]
        if same_orientation:
            ph, pd_, pa = r["prob_home"], r["prob_draw"], r["prob_away"]
            result_90 = r["result_90"]
        else:
            ph, pd_, pa = r["prob_away"], r["prob_draw"], r["prob_home"]
            result_90 = {"H": "A", "A": "H", "D": "D"}.get(r["result_90"])
        rows.append({
            "match_id": m["match_id"],
            "prob_home": ph, "prob_draw": pd_, "prob_away": pa,
            "result_90": result_90,
        })

    aligned = pd.DataFrame(rows)
    aligned.attrs["unmatched_match_ids"] = unmatched
    return aligned
