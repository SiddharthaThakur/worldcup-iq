"""
Phase 3 (Challenger 1): turn a squad of players into team strength.

What this does in simple English:
    The Elo baseline rates a team from its match history. This challenger
    rates a team from WHO IS IN IT. Each player gets a 0-1 quality score
    from the best signal we have — their EA FC rating if we know it, else
    their market value, else just their international caps as a coarse
    floor. We pick the team's likely best XI, then combine those players
    into attack, defense, and overall strength.

    Crucially, each team strength comes with a CONFIDENCE number: how much
    of the XI rests on real quality signal versus the caps-only floor. A
    team of Big-5 stars scores high confidence; a team of unrated
    domestic-league players scores low. Downstream, low confidence means
    "defer to Elo" — we never let a strength we can't see drive a
    prediction (see D018/D019).

    EA rating is the backbone because it is the one number that is both
    near-universal AND already calibrated across leagues by scouts —
    sidestepping the impossible task of putting xG and market value on a
    single scale. xG/creation refine this later (composition + context).
"""

import numpy as np
import pandas as pd

# Position groups from squad position codes (GK/DF/MF/FW from Wikipedia)
_POS_GROUP = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}

# 4-4-2 best-XI shape: how many of each group to start
FORMATION = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}

# How much each position group contributes to attack vs defense
_ATTACK_W = {"FWD": 0.50, "MID": 0.35, "DEF": 0.15, "GK": 0.0}
_DEFENSE_W = {"GK": 0.30, "DEF": 0.50, "MID": 0.20, "FWD": 0.0}


def pos_group(position: str) -> str:
    """Map a squad position code to GK/DEF/MID/FWD."""
    p = str(position).upper().strip()[:2]
    if p in _POS_GROUP:
        return _POS_GROUP[p]
    return {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}.get(p[:1], "MID")


def player_quality_score(player: dict) -> tuple[float, str]:
    """0-1 quality for one player, plus the basis used.

    Ladder (highest-resolution signal first):
      EA overall rating -> market value -> caps floor -> unknown floor.
    EA and value are scaled to land on a comparable range (a solid pro
    ~0.6, an elite player ~0.9+). Caps alone cap out at 0.5 — experience
    is not the same as quality.
    """
    ea = player.get("ea_overall")
    val = player.get("tm_value_eur")
    caps = player.get("caps") or 0

    if ea is not None and pd.notna(ea):
        return float(np.clip((float(ea) - 50.0) / 40.0, 0.0, 1.0)), "ea_or_value"
    if val is not None and pd.notna(val) and val > 0:
        # €100k -> 0, €200M -> 1 on a log scale
        score = (np.log10(float(val)) - 5.0) / (np.log10(2e8) - 5.0)
        return float(np.clip(score, 0.0, 1.0)), "ea_or_value"
    if caps and caps > 0:
        return float(np.clip(caps / 120.0, 0.0, 0.5)), "caps"
    return 0.30, "floor"


def _scored_squad(squad: pd.DataFrame) -> pd.DataFrame:
    df = squad.copy()
    qb = df.apply(lambda r: player_quality_score(r.to_dict()), axis=1)
    df["quality"] = [q for q, _ in qb]
    df["basis"] = [b for _, b in qb]
    df["pos_group"] = df["position"].apply(pos_group)
    return df


def best_xi(squad: pd.DataFrame) -> pd.DataFrame:
    """Estimate the strongest XI: top players per group by quality.

    We don't know the real lineup pre-match, so the team is represented
    by its likely best XI in a 4-4-2 shape. Short groups are filled with
    whoever is left (by quality) so we always return 11.
    """
    df = _scored_squad(squad).sort_values("quality", ascending=False)
    picked = []
    for group, n in FORMATION.items():
        picked.append(df[df["pos_group"] == group].head(n))
    xi = pd.concat(picked)
    if len(xi) < 11:  # backfill from best remaining
        rest = df[~df.index.isin(xi.index)].head(11 - len(xi))
        xi = pd.concat([xi, rest])
    return xi.head(11).reset_index(drop=True)


def team_strength(squad: pd.DataFrame) -> dict:
    """Compose a squad into overall / attack / defense strength + confidence.

    Returns dict with keys: overall, attack, defense, confidence,
    n_signal (XI players with real signal), group_means.
    """
    xi = best_xi(squad)
    group_means = {
        g: float(xi.loc[xi["pos_group"] == g, "quality"].mean())
        for g in ("GK", "DEF", "MID", "FWD")
        if (xi["pos_group"] == g).any()
    }

    def weighted(weights):
        num = sum(w * group_means[g] for g, w in weights.items() if g in group_means)
        den = sum(w for g, w in weights.items() if g in group_means)
        return num / den if den else float("nan")

    overall = float(xi["quality"].mean())
    # Confidence: share of the XI whose quality rests on real signal
    n_signal = int((xi["basis"] == "ea_or_value").sum())
    confidence = n_signal / len(xi)

    return {
        "overall": overall,
        "attack": weighted(_ATTACK_W),
        "defense": weighted(_DEFENSE_W),
        "confidence": confidence,
        "n_signal": n_signal,
        "group_means": group_means,
    }
