"""
Phase 4 dataset: real club lineups -> per-player feature tensors.

What this does in simple English:
    The attention model needs to read an actual starting XI as 11 player
    feature vectors. We take Big-5 club matches with known lineups
    (Transfermarkt), link each starter to their EA FC attributes
    (pace/shooting/passing/dribbling/defending/physical + overall), and
    assemble one tensor per team per match. Players we can't link are
    filled in with their position group's average — so a match is still
    usable if a couple of fringe players are missing.

    Output feeds two models trained head-to-head: attention (reads the XI
    as a set, models interactions) vs aggregation (mean-pools the same
    features). The comparison answers H3 — do player interactions help?

Run: python -m src.features.lineup_dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.entity_resolver import normalize_name

K = "data/raw/kaggle/transfermarkt/"
EA_PATH = "data/raw/kaggle/ea_fc26/ea_fc26_players.csv"
OUT = Path("data/processed/lineup_dataset.npz")

EA_FEATURES = ["pac", "sho", "pas", "dri", "def", "phy", "overallRating"]
N_FEATURES = len(EA_FEATURES)
BIG5 = {"GB1", "ES1", "IT1", "FR1", "L1"}
POS_GROUP_IDX = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

_TM_POS = {
    "Goalkeeper": "GK",
    "Centre-Back": "DEF", "Right-Back": "DEF", "Left-Back": "DEF",
    "Defender": "DEF", "Sweeper": "DEF",
    "Defensive Midfield": "MID", "Central Midfield": "MID",
    "Attacking Midfield": "MID", "Right Midfield": "MID", "Left Midfield": "MID",
    "midfield": "MID", "Midfield": "MID",
    "Centre-Forward": "FWD", "Left Winger": "FWD", "Right Winger": "FWD",
    "Second Striker": "FWD", "Attack": "FWD",
}


def tm_position_to_group(pos: str) -> str:
    return _TM_POS.get(str(pos).strip(), "MID")


def impute_missing_players(feats: list, groups: list, n_features: int) -> np.ndarray:
    """Replace None feature vectors with their position-group mean.

    Falls back to the overall mean when a group has no known player.
    """
    known = [(g, f) for g, f in zip(groups, feats) if f is not None]
    overall_mean = (np.mean([f for _, f in known], axis=0)
                    if known else np.zeros(n_features))
    group_mean = {}
    for g in set(groups):
        gf = [f for gg, f in known if gg == g]
        group_mean[g] = np.mean(gf, axis=0) if gf else overall_mean

    out = np.zeros((len(feats), n_features))
    for i, (g, f) in enumerate(zip(groups, feats)):
        out[i] = f if f is not None else group_mean[g]
    return out


def _build_ea_lookup(ea: pd.DataFrame) -> dict:
    """(norm_name, birth_year) -> feature vector."""
    ea = ea.copy()
    ea["norm"] = (ea.firstName.fillna("") + " " + ea.lastName.fillna("")).apply(normalize_name)
    ea["by"] = pd.to_datetime(ea.birthdate, errors="coerce").dt.year
    lookup = {}
    for _, r in ea.iterrows():
        if pd.isna(r["by"]):
            continue
        vec = r[EA_FEATURES].to_numpy(dtype=float)
        lookup[(r["norm"], int(r["by"]))] = vec
    return lookup


def _link_players(player_ids, tm_players: pd.DataFrame, ea_lookup: dict) -> dict:
    """player_id -> feature vector (or None). Exact name+birthyear (±1)."""
    sub = tm_players[tm_players.player_id.isin(player_ids)].copy()
    sub["norm"] = sub.name.fillna("").apply(normalize_name)
    sub["by"] = pd.to_datetime(sub.date_of_birth, errors="coerce").dt.year
    out = {}
    for _, r in sub.iterrows():
        vec = None
        if pd.notna(r["by"]):
            for dy in (0, 1, -1):
                vec = ea_lookup.get((r["norm"], int(r["by"]) + dy))
                if vec is not None:
                    break
        out[r["player_id"]] = vec
    return out


def build_dataset(min_year: int = 2022, min_linked: int = 8, save: bool = True) -> dict:
    """Assemble per-match lineup tensors for Big-5 games since min_year."""
    games = pd.read_csv(K + "games.csv", parse_dates=["date"])
    games = games[(games.date.dt.year >= min_year)
                  & (games.competition_id.isin(BIG5))
                  & games.home_club_goals.notna()]
    game_ids = set(games.game_id)

    lu = pd.read_csv(K + "game_lineups.csv", low_memory=False)
    lu = lu[(lu.game_id.isin(game_ids)) & (lu.type == "starting_lineup")]

    ea_lookup = _build_ea_lookup(pd.read_csv(EA_PATH))
    tm_players = pd.read_csv(K + "players.csv")
    feat_by_player = _link_players(lu.player_id.unique(), tm_players, ea_lookup)

    # Index lineups by (game, club)
    lu["group"] = lu.position.apply(tm_position_to_group)
    matches = {gid: g for gid, g in games.set_index("game_id").iterrows()}

    Xh, Ph, Xa, Pa, GH, GA, dates = [], [], [], [], [], [], []
    for gid, gl in lu.groupby("game_id"):
        if gid not in matches:
            continue
        m = matches[gid]
        sides = {}
        ok = True
        for club_id, cl in gl.groupby("club_id"):
            cl = cl.head(11)
            if len(cl) < 11:
                ok = False
                break
            feats = [feat_by_player.get(pid) for pid in cl.player_id]
            if sum(f is not None for f in feats) < min_linked:
                ok = False
                break
            groups = list(cl.group)
            X = impute_missing_players(feats, groups, N_FEATURES)
            P = np.array([POS_GROUP_IDX[g] for g in groups])
            sides[club_id] = (X, P)
        if not ok or len(sides) != 2:
            continue
        if m.home_club_id not in sides or m.away_club_id not in sides:
            continue
        Xh.append(sides[m.home_club_id][0]); Ph.append(sides[m.home_club_id][1])
        Xa.append(sides[m.away_club_id][0]); Pa.append(sides[m.away_club_id][1])
        GH.append(int(m.home_club_goals)); GA.append(int(m.away_club_goals))
        dates.append(str(m.date)[:10])

    data = {
        "Xh": np.array(Xh, dtype=np.float32), "Ph": np.array(Ph),
        "Xa": np.array(Xa, dtype=np.float32), "Pa": np.array(Pa),
        "gh": np.array(GH), "ga": np.array(GA), "dates": np.array(dates),
    }
    if save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT, **data)
    return data


if __name__ == "__main__":
    d = build_dataset()
    n = len(d["Xh"])
    print(f"{n} matches assembled, features {d['Xh'].shape}")
    ds = sorted(d["dates"].tolist())
    print(f"date range {ds[0]} -> {ds[-1]}")
    print(f"avg goals home {d['gh'].mean():.2f} away {d['ga'].mean():.2f}")
