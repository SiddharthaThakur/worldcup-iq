"""
Entity resolution: match player names across FBRef, Understat, StatsBomb.

What this does in simple English:
    The same player appears with different name spellings in different
    databases. This module figures out that "Kylian Mbappé Lottin" (FBRef)
    and "Kylian Mbappe" (Understat) are the same person, and gives them
    one canonical ID so we can join their data together.
"""

import unicodedata
import uuid
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

PLAYER_IDS_PATH = Path("data/processed/player_ids.csv")
MANUAL_MAPPINGS_PATH = Path("data/processed/manual_mappings.csv")
FUZZY_REVIEW_PATH = Path("data/processed/fuzzy_matches_review.csv")


def normalize_name(name: str) -> str:
    """Strip accents, lowercase, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = stripped.lower().strip()
    return " ".join(cleaned.split())


def deterministic_match(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    name_col: str = "name",
    year_col: str = "birth_year",
    nationality_col: str = "nationality",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pass 1: exact match on normalized name + birth year.

    Returns:
        matched: rows from source_df that found a canonical match
        unmatched: rows from source_df with no match (feed into fuzzy pass)
    """
    source = source_df.copy()
    canonical = canonical_df.copy()

    source["_norm_name"] = source[name_col].apply(normalize_name)
    canonical["_norm_name"] = canonical["canonical_name"].apply(normalize_name)

    merged = source.merge(
        canonical[["canonical_id", "_norm_name", year_col]],
        on=["_norm_name", year_col],
        how="left",
    )
    matched = merged[merged["canonical_id"].notna()].drop(columns=["_norm_name"])
    unmatched = merged[merged["canonical_id"].isna()].drop(columns=["_norm_name", "canonical_id"])

    return matched, unmatched


def fuzzy_match(
    unmatched_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    name_col: str = "name",
    nationality_col: str = "nationality",
    auto_threshold: int = 88,
    review_threshold: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pass 2: fuzzy match using rapidfuzz token_sort_ratio.

    Returns:
        auto_matched: high-confidence fuzzy matches (score >= auto_threshold)
        review: medium-confidence matches for human review (review_threshold <= score < auto_threshold)
        still_unmatched: no match found
    """
    auto_matches = []
    review_matches = []
    still_unmatched = []

    canonical_names = canonical_df[["canonical_id", "canonical_name", nationality_col]].to_dict("records")

    for _, row in unmatched_df.iterrows():
        norm_source = normalize_name(row[name_col])
        best_score = 0
        best_match = None

        for canon in canonical_names:
            # Only compare within same nationality
            if row.get(nationality_col) and canon.get(nationality_col):
                if row[nationality_col] != canon[nationality_col]:
                    continue

            norm_canon = normalize_name(canon["canonical_name"])
            score = fuzz.token_sort_ratio(norm_source, norm_canon)

            if score > best_score:
                best_score = score
                best_match = canon

        if best_score >= auto_threshold:
            auto_matches.append({**row.to_dict(), "canonical_id": best_match["canonical_id"],
                                "match_score": best_score})
        elif best_score >= review_threshold:
            review_matches.append({**row.to_dict(), "candidate_id": best_match["canonical_id"],
                                  "candidate_name": best_match["canonical_name"],
                                  "match_score": best_score})
        else:
            still_unmatched.append(row.to_dict())

    return (
        pd.DataFrame(auto_matches) if auto_matches else pd.DataFrame(),
        pd.DataFrame(review_matches) if review_matches else pd.DataFrame(),
        pd.DataFrame(still_unmatched) if still_unmatched else pd.DataFrame(),
    )


def create_canonical_entry(name: str, birth_year: int, nationality: str, position: str) -> dict:
    """Create a new canonical player entry."""
    return {
        "canonical_id": str(uuid.uuid4())[:8],
        "canonical_name": name,
        "birth_year": birth_year,
        "nationality": nationality,
        "position": position,
    }


def load_player_ids() -> pd.DataFrame:
    """Load the canonical player ID table."""
    if PLAYER_IDS_PATH.exists():
        return pd.read_csv(PLAYER_IDS_PATH)
    return pd.DataFrame(columns=[
        "canonical_id", "canonical_name", "birth_year", "nationality", "position",
        "fbref_id", "understat_id", "statsbomb_id",
    ])


def save_player_ids(df: pd.DataFrame) -> None:
    """Save the canonical player ID table."""
    PLAYER_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PLAYER_IDS_PATH, index=False)


def coverage_report(player_ids: pd.DataFrame) -> dict:
    """Report entity resolution coverage statistics."""
    total = len(player_ids)
    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "has_fbref": int(player_ids["fbref_id"].notna().sum()),
        "has_understat": int(player_ids["understat_id"].notna().sum()),
        "has_statsbomb": int(player_ids["statsbomb_id"].notna().sum()),
        "full_coverage": int(
            (player_ids["fbref_id"].notna() & player_ids["understat_id"].notna()).sum()
        ),
        "no_stats": int(
            (player_ids["fbref_id"].isna() & player_ids["understat_id"].isna()).sum()
        ),
    }
