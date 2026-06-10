"""
2026 World Cup fixtures and groups.

What this does in simple English:
    The results dataset already lists the 72 group-stage fixtures (with
    empty scores — they haven't been played). Since group-stage teams
    only ever play inside their own group, we can recover the 12 groups
    directly from the fixture list: each group is an isolated cluster of
    4 teams in the who-plays-whom graph. No scraping needed.

    Group letters: the three host groups are fixed by FIFA (Mexico = A,
    Canada = B, USA = D). The remaining letters are assigned to the other
    clusters in chronological order of their first match. This labeling
    is an approximation — the simulator only uses group composition and
    the structural bracket (see D010), so letters don't affect champion
    probabilities.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.results_loader import RAW_PATH
from src.data.team_aliases import resolve_team_code

FIXTURES_JSON = Path("data/processed/wc2026_fixtures.json")
GROUPS_JSON = Path("data/processed/wc2026_groups.json")

HOST_GROUPS = {"MEX": "A", "CAN": "B", "USA": "D"}


@dataclass
class WC2026:
    """Group-stage fixtures (DataFrame) and groups (letter → 4 team codes)."""

    fixtures: pd.DataFrame
    groups: dict[str, list[str]]


def _connected_components(edges: list[tuple[str, str]]) -> list[set[str]]:
    """Union-find over team codes."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comps: dict[str, set[str]] = {}
    for team in parent:
        comps.setdefault(find(team), set()).add(team)
    return list(comps.values())


def load_wc2026(raw_path: Path = RAW_PATH, save: bool = True) -> WC2026:
    """Extract 2026 group-stage fixtures and derive groups."""
    raw = pd.read_csv(raw_path, parse_dates=["date"])
    fx = raw[(raw["tournament"] == "FIFA World Cup") & (raw["date"].dt.year == 2026)].copy()

    fx["home_code"] = fx["home_team"].apply(resolve_team_code)
    fx["away_code"] = fx["away_team"].apply(resolve_team_code)
    fx["match_id"] = (
        fx["date"].dt.strftime("%Y-%m-%d") + "_" + fx["home_code"] + "_" + fx["away_code"]
    )

    components = _connected_components(list(zip(fx["home_code"], fx["away_code"])))

    # Host groups get their FIFA letters; the rest go chronologically
    first_match = {}
    for comp in components:
        key = frozenset(comp)
        mask = fx["home_code"].isin(comp)
        first_match[key] = (fx.loc[mask, "date"].min(), tuple(sorted(comp)))

    groups: dict[str, list[str]] = {}
    unassigned = []
    for comp in components:
        host = next((t for t in comp if t in HOST_GROUPS), None)
        if host:
            groups[HOST_GROUPS[host]] = sorted(comp)
        else:
            unassigned.append(comp)

    free_letters = [ch for ch in "ABCDEFGHIJKL" if ch not in groups]
    unassigned.sort(key=lambda c: first_match[frozenset(c)])
    for letter, comp in zip(free_letters, unassigned):
        groups[letter] = sorted(comp)

    team_group = {t: g for g, teams in groups.items() for t in teams}
    fx["group"] = fx["home_code"].map(team_group)
    fx = fx[["match_id", "date", "home_code", "away_code", "home_team", "away_team",
             "city", "country", "neutral", "group"]].sort_values("date").reset_index(drop=True)

    if save:
        FIXTURES_JSON.parent.mkdir(parents=True, exist_ok=True)
        fx.assign(date=fx["date"].dt.strftime("%Y-%m-%d")).to_json(
            FIXTURES_JSON, orient="records", indent=2)
        with open(GROUPS_JSON, "w") as f:
            json.dump(dict(sorted(groups.items())), f, indent=2)

    return WC2026(fixtures=fx, groups=groups)
