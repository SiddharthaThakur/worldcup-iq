"""
FBRef Big-5 leagues player stats, loaded from a Wayback Machine snapshot.

What this does in simple English:
    FBRef blocks automated requests, but the Internet Archive keeps
    public snapshots of its pages — reading a library copy, not
    sneaking past the bouncer. The snapshot we have is from
    2026-03-09 (~75% of the 2025/26 season). Partial-season per-90
    rates are fine; totals would be misleading vs full-season sources,
    so every row carries an `as_of` date and downstream code must
    treat these as rates, not totals.

    Bonus: FBRef publishes birth year AND nationality — the proper
    deterministic key for entity resolution (D006), which Understat
    lacks.

    FBRef quirk: data tables are wrapped in HTML comments (an old
    anti-scraping trick); we unwrap before parsing.
"""

import re
from io import StringIO
from pathlib import Path

import pandas as pd

HTML_PATH = Path("data/raw/fbref_big5_standard_20260309.html")
OUT_PARQUET = Path("data/processed/fbref_big5.parquet")

_RENAME = {
    "Player": "player", "Nation": "nation", "Pos": "position", "Squad": "squad",
    "Comp": "league", "Born": "born", "MP": "matches", "Min": "minutes",
    "Gls": "goals", "Ast": "assists", "G-PK": "np_goals",
    "CrdY": "yellow", "CrdR": "red",
}


def _as_of_from_path(path: Path) -> str:
    m = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)
    return "-".join(m.groups()) if m else "unknown"


def load_fbref_big5(path: Path = HTML_PATH, save: bool = True) -> pd.DataFrame:
    """Parse the standard-stats table into a tidy DataFrame."""
    html = path.read_text()
    html = html.replace("<!--", "").replace("-->", "")  # unwrap commented tables
    df = pd.read_html(StringIO(html), attrs={"id": "stats_standard"})[0]

    # Flatten the two-level header. The bottom level alone collides
    # ("Gls" exists in both Performance and Per 90 Minutes groups), so
    # per-90 columns get a suffix.
    df.columns = [
        f"{c[1]}_per90" if str(c[0]).startswith("Per 90") else c[1]
        for c in df.columns
    ]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[df["Player"] != "Player"]  # drop repeated mid-table headers
    df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})
    df = df[[c for c in _RENAME.values() if c in df.columns]].copy()

    # "ma MAR" -> "MAR"; some rows have only the uppercase code
    df["nation"] = df["nation"].astype(str).str.split().str[-1].str.upper()
    # "Premier League" stays; "eng Premier League" -> "Premier League"
    df["league"] = df["league"].astype(str).str.replace(r"^[a-z]+ ", "", regex=True)

    for col in ("born", "matches", "minutes", "goals", "assists", "np_goals",
                "yellow", "red"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["born", "minutes"])
    df["born"] = df["born"].astype(int)
    df["as_of"] = _as_of_from_path(path)
    df = df.reset_index(drop=True)

    if save:
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PARQUET, index=False)
    return df


if __name__ == "__main__":
    df = load_fbref_big5()
    print(f"{len(df)} players, as of {df['as_of'].iloc[0]}")
    print(df.groupby("league").size().to_string())
