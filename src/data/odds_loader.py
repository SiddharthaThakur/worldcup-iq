"""
Bookmaker odds ingestion: overround removal and Closing Line Value.

What this does in simple English:
    Bookmaker odds contain a built-in profit margin (the "overround" or
    "vig") — the implied probabilities sum to ~105-110%, not 100%. To use
    odds as a fair benchmark, we strip that margin out.

    We use the basic proportional method by default and also implement the
    power method, which the literature suggests better handles the
    favorite-longshot bias (bookmakers shade longshots more than favorites).
    We report both — whether they disagree materially is itself worth knowing.

    Closing Line Value (CLV): the sharpest single test of whether a model
    has real edge. If the market's CLOSING odds consistently move TOWARD
    our earlier prediction, we knew something before the market did.

Data sources:
    - Historical (2018/2022 WC): football-data.co.uk CSVs
    - Live (2026): The Odds API free tier, or manual entry CSV
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ODDS_DIR = Path("data/processed")


@dataclass
class MarketProbs:
    """De-vigged market probabilities for one match."""

    match_id: str
    prob_home: float
    prob_draw: float
    prob_away: float
    overround: float          # the margin that was removed
    method: str               # "proportional" or "power"
    snapshot: str             # "opening" or "closing"


def devig_proportional(odds_home: float, odds_draw: float,
                       odds_away: float) -> tuple[float, float, float, float]:
    """Remove the overround by simple normalization.

    Returns (p_home, p_draw, p_away, overround).
    """
    raw = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    total = raw.sum()
    p = raw / total
    return float(p[0]), float(p[1]), float(p[2]), float(total - 1.0)


def devig_power(odds_home: float, odds_draw: float, odds_away: float,
                tol: float = 1e-10) -> tuple[float, float, float, float]:
    """Remove the overround by the power method: find k such that
    sum((1/odds)^k) = 1. Better handles favorite-longshot bias than
    proportional normalization.
    """
    raw = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    overround = float(raw.sum() - 1.0)

    lo, hi = 0.5, 3.0
    for _ in range(200):
        k = (lo + hi) / 2
        s = float(np.sum(raw ** k))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = k
        else:
            hi = k
    p = raw ** k
    p /= p.sum()  # numerical cleanup
    return float(p[0]), float(p[1]), float(p[2]), overround


def market_probs_from_odds(
    match_id: str, odds_home: float, odds_draw: float, odds_away: float,
    method: str = "power", snapshot: str = "closing",
) -> MarketProbs:
    """Convert decimal odds to de-vigged probabilities."""
    fn = devig_power if method == "power" else devig_proportional
    ph, pd_, pa, ov = fn(odds_home, odds_draw, odds_away)
    return MarketProbs(match_id, ph, pd_, pa, ov, method, snapshot)


def closing_line_value(
    model_prob: float, opening_market_prob: float, closing_market_prob: float,
) -> float:
    """CLV for one outcome of one match.

    Positive when the market moved TOWARD our prediction between open
    and close — i.e., the model 'knew' before the market priced it in.
    """
    return (closing_market_prob - opening_market_prob) * np.sign(
        model_prob - opening_market_prob
    )


def load_historical_wc_odds(path: Path) -> pd.DataFrame:
    """Load football-data.co.uk style CSV of World Cup odds.

    Expected columns (their convention): Date, HomeTeam, AwayTeam,
    B365H, B365D, B365A (Bet365 home/draw/away decimal odds), plus
    optionally PSH/PSD/PSA (Pinnacle, sharpest book — prefer if present).
    """
    df = pd.read_csv(path)
    use_pinnacle = all(c in df.columns for c in ("PSH", "PSD", "PSA"))
    h, d, a = ("PSH", "PSD", "PSA") if use_pinnacle else ("B365H", "B365D", "B365A")

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get(h)) or pd.isna(r.get(d)) or pd.isna(r.get(a)):
            continue
        mp = market_probs_from_odds(
            match_id=f"{r['Date']}_{r['HomeTeam']}_{r['AwayTeam']}",
            odds_home=float(r[h]), odds_draw=float(r[d]), odds_away=float(r[a]),
        )
        rows.append({
            "match_id": mp.match_id, "home_team": r["HomeTeam"],
            "away_team": r["AwayTeam"],
            "prob_home": mp.prob_home, "prob_draw": mp.prob_draw,
            "prob_away": mp.prob_away, "overround": mp.overround,
            "book": "pinnacle" if use_pinnacle else "bet365",
        })
    return pd.DataFrame(rows)


def load_wc_odds_workbook(path: Path, sheet: str, method: str = "power") -> pd.DataFrame:
    """Load one sheet of the football-data.co.uk World Cup workbook
    (WorldCup2026.xlsx contains sheets for 2014/2018/2022 + 2026 qualifiers).

    Uses the cross-bookmaker AVERAGE odds (H-Avg/D-Avg/A-Avg) as the
    market consensus. Also extracts the 90-MINUTE result (HGFT/AGFT):
    1X2 odds settle on the 90-minute score, so a knockout match decided
    in extra time counts as a draw for market-comparison purposes.

    Returns columns: match_id, home_code, away_code, date, prob_home,
    prob_draw, prob_away, overround, result_90.
    """
    from src.data.team_aliases import resolve_team_code

    df = pd.read_excel(path, sheet_name=sheet)
    # 2026 qualifiers sheet uses H_Avg / HG; tournament sheets use H-Avg / HGFT
    df.columns = [c.replace("_", "-") for c in df.columns]
    hg_col = "HGFT" if "HGFT" in df.columns else "HG"
    ag_col = "AGFT" if "AGFT" in df.columns else "AG"

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("H-Avg")) or pd.isna(r.get("D-Avg")) or pd.isna(r.get("A-Avg")):
            continue
        home_code = resolve_team_code(str(r["Home"]).strip())
        away_code = resolve_team_code(str(r["Away"]).strip())
        date = pd.Timestamp(r["Date"]).strftime("%Y-%m-%d")
        mp = market_probs_from_odds(
            match_id=f"{date}_{home_code}_{away_code}",
            odds_home=float(r["H-Avg"]), odds_draw=float(r["D-Avg"]),
            odds_away=float(r["A-Avg"]), method=method,
        )
        result_90 = None
        if not pd.isna(r.get(hg_col)) and not pd.isna(r.get(ag_col)):
            hg, ag = int(r[hg_col]), int(r[ag_col])
            result_90 = "H" if hg > ag else ("D" if hg == ag else "A")
        rows.append({
            "match_id": mp.match_id, "home_code": home_code, "away_code": away_code,
            "date": date, "prob_home": mp.prob_home, "prob_draw": mp.prob_draw,
            "prob_away": mp.prob_away, "overround": mp.overround,
            "result_90": result_90,
        })
    return pd.DataFrame(rows)


def load_manual_odds(path: Path = ODDS_DIR / "manual_odds_2026.csv") -> pd.DataFrame:
    """Load manually entered 2026 odds.

    CSV format: match_id, snapshot, odds_home, odds_draw, odds_away
    (snapshot = 'opening' or 'closing'). Manual entry takes ~10 minutes
    per match day and removes all API dependency risk.
    """
    if not path.exists():
        return pd.DataFrame(
            columns=["match_id", "snapshot", "prob_home", "prob_draw",
                     "prob_away", "overround"])
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        mp = market_probs_from_odds(
            r["match_id"], r["odds_home"], r["odds_draw"], r["odds_away"],
            snapshot=r.get("snapshot", "closing"),
        )
        rows.append({
            "match_id": mp.match_id, "snapshot": mp.snapshot,
            "prob_home": mp.prob_home, "prob_draw": mp.prob_draw,
            "prob_away": mp.prob_away, "overround": mp.overround,
        })
    return pd.DataFrame(rows)
