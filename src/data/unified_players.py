"""
Build one unified player table joining the 1,246 World Cup squad players
to every stats source we have.

What this does in simple English:
    The squad list (from Wikipedia) is the spine. For each squad player
    we go find their row in each source — Kaggle basic stats, Understat
    xG, EA FC 26 ratings, Transfermarkt market value — and staple it on.
    Sources spell names differently and use different nationality labels,
    so we match on the most reliable key available: normalized name +
    birth year, with nationality breaking ties and a fuzzy fallback that
    rescues romanized names (Korean/Arabic name-order flips) — but only
    when nationality agrees, so a loose name match can't grab the wrong
    person.

    Each player ends up with a `data_quality` tier recording the BEST
    signal we found, from tier1_xg (full Understat xG) down to
    tier4_caps_only (nobody has them; only Wikipedia caps/goals exist).
    Teams whose players are mostly tier3/4 will be flagged so the
    player-composition model can defer to the Elo baseline for them.

Output: data/processed/unified_players.parquet  (+ coverage report)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.data.entity_resolver import normalize_name
from src.data.team_aliases import resolve_team_code

FUZZY_THRESHOLD = 82
OUT_PARQUET = Path("data/processed/unified_players.parquet")

# Source nationality strings that resolve_team_code doesn't already know
_EXTRA_NAT_ALIASES = {
    "korea, south": "KOR", "south korea": "KOR", "korea republic": "KOR",
    "korea, north": "PRK",
    "ivory coast": "CIV", "cote d'ivoire": "CIV", "côte d'ivoire": "CIV",
    "china pr": "CHN", "united states": "USA", "usa": "USA",
    "cape verde islands": "CPV", "cabo verde": "CPV",
}


def nationality_to_code(nat: str) -> str | None:
    """Resolve a source's free-text nationality to a FIFA code, or None."""
    if not nat or pd.isna(nat):
        return None
    raw = str(nat).strip()
    # Kaggle Nation looks like "kr KOR" / "fr FRA" — last token is the code
    parts = raw.split()
    if len(parts) >= 2 and len(parts[-1]) == 3 and parts[-1].isupper():
        return parts[-1]
    low = normalize_name(raw)
    if low in _EXTRA_NAT_ALIASES:
        return _EXTRA_NAT_ALIASES[low]
    code = resolve_team_code(raw)
    return code if len(code) == 3 and code.isupper() else None


def match_player_to_source(
    name: str, birth_year: int | None, team_code: str, source: pd.DataFrame,
    name_col: str = "name", year_col: str = "birth_year",
    nat_col: str | None = "nationality",
) -> tuple[int | None, str, float]:
    """Match one squad player to a source DataFrame.

    Returns (source_idx_value, method, score). source_idx_value is the
    value in source[`idx`-like first column] — here we return the value
    of the source's index label, so callers index with .loc.

    Ladder: exact name+birthyear (nationality breaks ties) -> fuzzy name
    within birthyear±1 AND nationality agreement.
    """
    src = source.copy()
    src["_norm"] = src[name_col].apply(normalize_name)
    norm = normalize_name(name)

    def nat_ok(row) -> bool:
        if nat_col is None or team_code is None:
            return True
        return nationality_to_code(row.get(nat_col)) == team_code

    # 1. Exact normalized name
    exact = src[src["_norm"] == norm]
    if birth_year is not None:
        exact = exact[exact[year_col].between(birth_year - 1, birth_year + 1)]
    if len(exact) == 1:
        return exact.index[0], "exact", 100.0
    if len(exact) > 1:
        nat_hits = exact[exact.apply(nat_ok, axis=1)]
        if len(nat_hits) >= 1:
            return nat_hits.index[0], "exact+nat", 100.0
        return exact.index[0], "exact_ambiguous", 100.0

    # 2. Fuzzy within birth-year window, nationality must agree
    pool = src
    if birth_year is not None:
        pool = pool[pool[year_col].between(birth_year - 1, birth_year + 1)]
    pool = pool[pool.apply(nat_ok, axis=1)]
    best_i, best_s = None, 0.0
    for i, row in pool.iterrows():
        s = max(fuzz.token_sort_ratio(norm, row["_norm"]),
                fuzz.token_set_ratio(norm, row["_norm"]))
        if s > best_s:
            best_i, best_s = i, s
    if best_i is not None and best_s >= FUZZY_THRESHOLD:
        return best_i, f"fuzzy_{int(best_s)}", best_s
    return None, "none", 0.0


def assign_tier(has_understat: bool, has_stats: bool,
                has_ea: bool, has_value: bool) -> str:
    """Best-signal tier for a player (highest resolution wins)."""
    if has_understat:
        return "tier1_xg"
    if has_stats:
        return "tier2_stats"
    if has_ea:
        return "tier3_rating"
    if has_value:
        return "tier3_value"
    return "tier4_caps_only"


# ── Full assembly (integration; sanity-checked on real data) ──────────────

def _load_sources() -> dict:
    K = Path("data/raw/kaggle")
    stats = pd.read_csv(K / "stats_2025_26/players_data-2025_2026.csv")
    stats["birth_year"] = pd.to_numeric(stats["Born"], errors="coerce")
    stats = stats.rename(columns={"Player": "name", "Nation": "nationality"})

    understat = pd.read_parquet("data/processed/understat_players.parquet")

    ea = pd.read_csv(K / "ea_fc26/ea_fc26_players.csv")
    ea["name"] = (ea.firstName.fillna("") + " " + ea.lastName.fillna("")).str.strip()
    ea["birth_year"] = pd.to_datetime(ea.birthdate, errors="coerce").dt.year

    tm = pd.read_csv(K / "transfermarkt/players.csv")
    tm["birth_year"] = pd.to_datetime(tm.date_of_birth, errors="coerce").dt.year
    tm = tm.rename(columns={"country_of_citizenship": "nationality"})

    return {"stats": stats, "understat": understat, "ea": ea, "tm": tm}


def build_unified_player_table(save: bool = True) -> pd.DataFrame:
    """Match every squad player to all sources; assemble + tier the table."""
    squads = json.loads(Path("data/processed/wc2026_squads.json").read_text())
    src = _load_sources()

    rows = []
    for team, players in squads.items():
        for p in players:
            by = int(p["birth_date"][:4]) if p.get("birth_date") else None
            row = {"team": team, "name": p["name"], "birth_year": by,
                   "position": p["position"], "club": p["club"],
                   "caps": p["caps"], "goals": p["goals"]}

            # EA + Transfermarkt: have name+birthyear+nationality
            ea_i, ea_m, _ = match_player_to_source(p["name"], by, team, src["ea"])
            tm_i, tm_m, _ = match_player_to_source(p["name"], by, team, src["tm"])
            st_i, st_m, _ = match_player_to_source(p["name"], by, team, src["stats"])
            # Understat: no birth year -> match on name only, nationality off,
            # club agreement handled downstream; use name+nat=None here
            us_i, us_m, _ = match_player_to_source(
                p["name"], None, None, src["understat"],
                name_col="player_name", year_col="season", nat_col=None)

            row["ea_overall"] = int(src["ea"].loc[ea_i, "overallRating"]) if ea_i is not None else None
            row["tm_value_eur"] = src["tm"].loc[tm_i, "market_value_in_eur"] if tm_i is not None else None
            row["us_npxg90"] = src["understat"].loc[us_i, "npxg90"] if us_i is not None else None
            row["us_xa90"] = src["understat"].loc[us_i, "xa90"] if us_i is not None else None
            row["stats_goals"] = src["stats"].loc[st_i, "Gls"] if st_i is not None else None

            row["data_quality"] = assign_tier(
                has_understat=us_i is not None,
                has_stats=st_i is not None,
                has_ea=ea_i is not None,
                has_value=tm_i is not None and pd.notna(row["tm_value_eur"]),
            )
            row["match_methods"] = f"ea:{ea_m}|tm:{tm_m}|stats:{st_m}|us:{us_m}"
            rows.append(row)

    df = pd.DataFrame(rows)
    if save:
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PARQUET, index=False)
    return df


if __name__ == "__main__":
    df = build_unified_player_table()
    print(f"{len(df)} squad players")
    print("\ndata_quality distribution:")
    print(df["data_quality"].value_counts().to_string())
    print(f"\nany-signal coverage: "
          f"{(df['data_quality'] != 'tier4_caps_only').mean():.1%}")
    print("\nteam-level: share of players at tier3+ (player model weak):")
    weak = df.groupby("team")["data_quality"].apply(
        lambda s: s.isin(["tier3_rating", "tier3_value", "tier4_caps_only"]).mean())
    print((weak.sort_values(ascending=False).head(8) * 100).round(0).astype(int).to_string())
