"""
Parse 2026 World Cup squads from the Wikipedia squads page.

What this does in simple English:
    Each of the 48 teams has a 26-player table on Wikipedia (number,
    position, name, birth date, caps, goals, club). We walk the page in
    document order: every team header (h3) is followed by its squad
    table. Team names are resolved to FIFA codes; player birth dates are
    normalized to YYYY-MM-DD (needed later for entity resolution, where
    name + birth year is the matching key).

Output: data/processed/wc2026_squads.json — {team_code: [player, ...]}
"""

import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.data.team_aliases import resolve_team_code

HTML_PATH = Path("data/raw/wc2026_squads.html")
SQUADS_JSON = Path("data/processed/wc2026_squads.json")

# Sections after the squads that also use h3 headers (stats appendices)
_SQUAD_COLUMNS = {"No.", "Pos.", "Player", "Caps", "Goals", "Club"}


def _clean_name(raw: str) -> str:
    """Strip footnote markers and captain annotations from a player name."""
    name = re.sub(r"\[.*?\]", "", raw)          # [a], [1] footnotes
    name = re.sub(r"\(.*?captain.*?\)", "", name, flags=re.I)
    name = name.replace("(c)", "").replace("(C)", "")
    return name.strip()


def _parse_birth_date(raw: str) -> str | None:
    """'May 17, 2000 (aged 26)' → '2000-05-17'."""
    m = re.match(r"([A-Za-z]+ \d{1,2}, \d{4})", str(raw).strip())
    if not m:
        return None
    return pd.Timestamp(m.group(1)).strftime("%Y-%m-%d")


def parse_squads(html_path: Path = HTML_PATH, save: bool = True) -> dict[str, list[dict]]:
    """Parse all 48 squads. Returns {fifa_code: [player dicts]}."""
    soup = BeautifulSoup(html_path.read_text(), "lxml")

    squads: dict[str, list[dict]] = {}
    for h3 in soup.find_all("h3"):
        team_name = (h3.get("id") or h3.get_text()).replace("_", " ").strip()
        code = resolve_team_code(team_name)
        if len(code) != 3 or not code.isupper():
            continue  # not a team header (stats appendix sections)

        table = h3.find_next("table", class_="wikitable")
        if table is None:
            continue
        df = pd.read_html(StringIO(str(table)))[0]
        if not _SQUAD_COLUMNS <= set(map(str, df.columns)):
            continue

        players = []
        for _, r in df.iterrows():
            name = _clean_name(str(r["Player"]))
            if not name or name.lower() == "nan":
                continue
            players.append({
                "name": name,
                "number": int(r["No."]) if pd.notna(r["No."]) else None,
                "position": re.sub(r"[^A-Z]", "", str(r["Pos."]))[-2:],
                "birth_date": _parse_birth_date(r["Date of birth (age)"]),
                "caps": int(r["Caps"]) if pd.notna(r["Caps"]) else 0,
                "goals": int(r["Goals"]) if pd.notna(r["Goals"]) else 0,
                "club": str(r["Club"]).strip(),
            })
        if players:
            squads[code] = players

    if save:
        SQUADS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(SQUADS_JSON, "w") as f:
            json.dump(squads, f, indent=2, ensure_ascii=False)
    return squads
