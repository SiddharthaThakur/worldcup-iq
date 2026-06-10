"""
Understat player xG/xA stats for the Big-5 European leagues.

What this does in simple English:
    Understat publishes "expected goals" (xG) — a measure of the quality
    of chances a player gets, which predicts future scoring better than
    actual goals do. Their site loads data from a JSON endpoint; we call
    it politely (one request per league with delays), cache the raw
    responses, and parse them into a clean table with per-90-minute rates.

    Players with under 270 minutes (3 full matches) get NaN per-90 rates:
    a rate computed from 50 minutes of play is noise wearing a number's
    clothes, and downstream models must see "unknown", not noise.

Usage:
    python -m src.data.understat_loader        # download all 5 leagues + parse
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
RAW_DIR = Path("data/raw/understat")
OUT_PARQUET = Path("data/processed/understat_players.parquet")
MIN_MINUTES_FOR_RATES = 270

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"),
    "X-Requested-With": "XMLHttpRequest",
}

_NUMERIC = {
    "time": "minutes", "games": "games", "goals": "goals", "assists": "assists",
    "shots": "shots", "key_passes": "key_passes", "npg": "npg",
    "xG": "xg", "xA": "xa", "npxG": "npxg",
    "xGChain": "xg_chain", "xGBuildup": "xg_buildup",
}

_PER90 = ["goals", "assists", "shots", "key_passes", "xg", "xa", "npxg"]


def fetch_league(league: str, season: int = 2025) -> dict:
    """Fetch one league's player stats from Understat's JSON endpoint."""
    resp = requests.post(
        "https://understat.com/main/getPlayersStats/",
        data={"league": league, "season": str(season)},
        headers={**_HEADERS, "Referer": f"https://understat.com/league/{league}/{season}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def download_all(season: int = 2025, delay_s: float = 2.0) -> None:
    """Download all 5 leagues with polite delays; skip files already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for league in LEAGUES:
        out = RAW_DIR / f"{league}_{season}.json"
        if out.exists():
            print(f"{league}: cached")
            continue
        data = fetch_league(league, season)
        n = len(data.get("players", []))
        if n < 100:
            raise RuntimeError(f"{league}: only {n} players returned — endpoint changed?")
        out.write_text(json.dumps(data))
        print(f"{league}: {n} players downloaded")
        time.sleep(delay_s)


def load_understat_players(season: int = 2025, save: bool = True) -> pd.DataFrame:
    """Parse cached league JSONs into one tidy DataFrame with per-90 rates."""
    frames = []
    for league in LEAGUES:
        path = RAW_DIR / f"{league}_{season}.json"
        if not path.exists():
            continue
        players = json.loads(path.read_text())["players"]
        df = pd.DataFrame(players)
        df["league"] = league
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No cached Understat files in {RAW_DIR}. "
                                "Run `python -m src.data.understat_loader` first.")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={**_NUMERIC, "team_title": "club"})
    for col in _NUMERIC.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["season"] = season

    enough = df["minutes"] >= MIN_MINUTES_FOR_RATES
    for col in _PER90:
        df[f"{col}90"] = np.where(enough, df[col] / df["minutes"] * 90, np.nan)

    keep = (["id", "player_name", "club", "league", "season", "position",
             "games", "minutes"] + list(_NUMERIC.values())[2:]
            + [f"{c}90" for c in _PER90])
    keep = [c for c in dict.fromkeys(keep)]  # dedupe, keep order
    df = df[keep]

    if save:
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PARQUET, index=False)
    return df


if __name__ == "__main__":
    download_all()
    df = load_understat_players()
    print(f"\n{len(df)} players across {df['league'].nunique()} leagues")
    print(df.groupby('league').size().to_string())
    top = df.nlargest(5, 'xg90')[['player_name', 'club', 'league', 'minutes', 'xg90']]
    print("\nTop xG/90 (min 270 min):")
    print(top.to_string(index=False))
