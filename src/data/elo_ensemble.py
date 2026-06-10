"""
Ensemble our Elo with an independent rating system (variance reduction).

What this does in simple English:
    We built our own Elo from match results. An independent project
    (eloratings.net-style, via Kaggle) built theirs with a differently
    tuned formula. Neither is "right", but averaging two reasonable
    rating systems cancels some of each one's idiosyncratic errors — the
    cheapest reliable trick in forecasting.

    The two systems live on different scales (theirs is more spread out),
    so we first rescale the independent ratings to OUR mean and spread
    across the shared teams, then average. Teams the independent system
    doesn't cover keep our rating unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.team_aliases import resolve_team_code

# Distilled 48-team file (committed, so the cloud daily-update needs no
# Kaggle access). Falls back to the full raw file if the distilled one
# is absent (local dev right after a fresh Kaggle pull).
INDEP_PATH = Path("data/processed/independent_elo_2026.csv")
_RAW_INDEP_PATH = Path("data/raw/kaggle/elo2026/elo_ratings_wc2026.csv")


def align_and_ensemble(ours: dict, indep: dict, weight: float = 0.5) -> dict:
    """Rescale `indep` to ours' mean/std over shared teams, then average.

    Args:
        ours: {team_code: our_elo}
        indep: {team_code: independent_rating} (any scale)
        weight: blend weight on the rescaled independent rating
    """
    shared = [t for t in ours if t in indep]
    if len(shared) < 3:
        return dict(ours)

    o = np.array([ours[t] for t in shared], dtype=float)
    i = np.array([indep[t] for t in shared], dtype=float)
    o_mu, o_sd = o.mean(), o.std()
    i_mu, i_sd = i.mean(), i.std()
    i_sd = i_sd if i_sd > 1e-9 else 1.0

    out = {}
    for t in ours:
        if t in indep:
            rescaled = o_mu + (indep[t] - i_mu) / i_sd * o_sd
            out[t] = float((1 - weight) * ours[t] + weight * rescaled)
        else:
            out[t] = float(ours[t])
    return out


def load_independent_ratings(path: Path = INDEP_PATH) -> dict:
    """Latest independent ratings per team, mapped to FIFA codes."""
    if path.exists():  # distilled {team, rating} file
        df = pd.read_csv(path)
        return {r["team"]: float(r["rating"]) for _, r in df.iterrows()}
    # Fallback: parse the full raw Kaggle file
    df = pd.read_csv(_RAW_INDEP_PATH)
    latest = df[df["year"] == df["year"].max()].drop_duplicates("country")
    out = {}
    for _, r in latest.iterrows():
        code = resolve_team_code(str(r["country"]))
        if len(code) == 3 and code.isupper():
            out[code] = float(r["rating"])
    return out
