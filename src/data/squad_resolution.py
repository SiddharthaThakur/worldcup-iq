"""
Match the 1,246 World Cup squad players to Understat players.

What this does in simple English:
    The squads (from Wikipedia) and the stats (from Understat) spell
    names differently: "Kylian Mbappé" vs "Kylian Mbappe", hyphens vs
    spaces, Turkish dotless-i, etc. Understat doesn't publish birth
    dates, so we can't use the usual name+birthyear key. Instead, the
    CLUB is the validator: Wikipedia tells us each squad player's club,
    Understat tells us each stats row's club. A name match is trusted
    when the clubs agree.

    Matching ladder (strictest first):
      1. exact normalized name + club agreement        -> "exact+club"
      2. exact normalized name, globally unique on both
         sides, multi-word name                        -> "exact_unique"
      3. fuzzy name (>= 85) among same-club players    -> "fuzzy_club"
    Anything else stays unmatched (data_quality="none") rather than
    guessing — a wrong player mapping silently poisons the model.

Output: data/processed/squad_understat_map.csv
"""

from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from src.data.entity_resolver import normalize_name

OUT_CSV = Path("data/processed/squad_understat_map.csv")

CLUB_AGREE_THRESHOLD = 75   # token_set_ratio on normalized club names
FUZZY_NAME_THRESHOLD = 85   # token_sort_ratio for same-club fuzzy pass


def _norm_club(club: str) -> str:
    """Normalize club names: accents, case, punctuation, filler words."""
    n = normalize_name(str(club))
    for token in ("fc", "cf", "afc", "ac", "as", "ss", "sc", "club", "de"):
        n = f" {n} ".replace(f" {token} ", " ").strip()
    return n.replace("-", " ")


def clubs_agree(club_a: str, club_b: str) -> bool:
    return fuzz.token_set_ratio(_norm_club(club_a), _norm_club(club_b)) >= CLUB_AGREE_THRESHOLD


def resolve_squads_to_understat(
    squads: pd.DataFrame, understat: pd.DataFrame, save: bool = False
) -> pd.DataFrame:
    """Resolve each squad player to an Understat id where possible.

    Args:
        squads: columns team, name, club, position, birth_date
        understat: columns id, player_name, club, league

    Returns:
        squads plus: understat_id, understat_name, understat_club,
        match_method, match_score, data_quality.
    """
    us = understat.copy()
    us["_norm"] = us["player_name"].apply(normalize_name)
    us_by_norm: dict[str, list] = {}
    for _, r in us.iterrows():
        us_by_norm.setdefault(r["_norm"], []).append(r)

    squad_name_counts = squads["name"].apply(normalize_name).value_counts()

    rows = []
    for _, p in squads.iterrows():
        norm = normalize_name(p["name"])
        match, method, score = None, None, None

        candidates = us_by_norm.get(norm, [])

        # 1. Exact name, club agrees (handles duplicate names cleanly)
        club_ok = [c for c in candidates if clubs_agree(p["club"], c["club"])]
        if len(club_ok) == 1:
            match, method, score = club_ok[0], "exact+club", 100
        # 2. Exact name, unique on both sides, multi-word (no club info match,
        #    e.g. player transferred mid-season) — single-token names like
        #    "Danilo" are too ambiguous to accept without club agreement
        elif (len(candidates) == 1 and not club_ok
              and squad_name_counts[norm] == 1 and len(norm.split()) >= 2):
            match, method, score = candidates[0], "exact_unique", 100
        else:
            # 3. Fuzzy among same-club Understat players. token_set_ratio
            # is subset-aware: Wikipedia's full legal names ("Achraf
            # Hakimi Mouh") match Understat's short forms ("Achraf
            # Hakimi") at 100. Safe here ONLY because the club must
            # already agree — never use set-ratio for a global pass.
            best, best_score = None, 0
            for _, c in us.iterrows():
                if not clubs_agree(p["club"], c["club"]):
                    continue
                s = max(fuzz.token_sort_ratio(norm, c["_norm"]),
                        fuzz.token_set_ratio(norm, c["_norm"]))
                if s > best_score:
                    best, best_score = c, s
            if best is not None and best_score >= FUZZY_NAME_THRESHOLD:
                match, method, score = best, "fuzzy_club", best_score

        row = p.to_dict()
        if match is not None:
            row.update(understat_id=match["id"], understat_name=match["player_name"],
                       understat_club=match["club"], match_method=method,
                       match_score=score, data_quality="tier1_xg")
        else:
            row.update(understat_id=None, understat_name=None, understat_club=None,
                       match_method=None, match_score=None, data_quality="none")
        rows.append(row)

    out = pd.DataFrame(rows)
    if save:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
    return out


def squads_json_to_df(squads_dict: dict) -> pd.DataFrame:
    """Flatten the wc2026_squads.json structure into a DataFrame."""
    rows = []
    for team, players in squads_dict.items():
        for p in players:
            rows.append({"team": team, "name": p["name"], "club": p["club"],
                         "position": p["position"], "birth_date": p["birth_date"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json

    squads = squads_json_to_df(json.loads(Path("data/processed/wc2026_squads.json").read_text()))
    understat = pd.read_parquet("data/processed/understat_players.parquet")
    out = resolve_squads_to_understat(squads, understat, save=True)

    matched = out["understat_id"].notna()
    print(f"matched {matched.sum()} / {len(out)} squad players "
          f"({matched.mean():.0%})")
    print("\nby method:")
    print(out[matched]["match_method"].value_counts().to_string())
    print("\nper-team coverage (lowest 8):")
    cov = out.groupby("team")["understat_id"].apply(lambda s: s.notna().mean())
    print((cov.sort_values().head(8) * 100).round(0).astype(int).to_string())
    print("\nper-team coverage (highest 5):")
    print((cov.sort_values().tail(5) * 100).round(0).astype(int).to_string())
