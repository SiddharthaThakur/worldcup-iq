"""
Load and clean international football match results from the Kaggle dataset.

Source: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
Contains 49,000+ international match results from 1872 to 2025.

What this does in simple English:
    Downloads a big CSV of every international football match ever played,
    cleans it up (standardizes team names, parses dates), and filters it
    to the relevant time window for our models (post-2010 by default).
"""

from pathlib import Path
import pandas as pd
from src.data.team_aliases import resolve_team_code

RAW_PATH = Path("data/raw/international_results.csv")
PROCESSED_PATH = Path("data/processed/international_results.parquet")


def load_raw_results(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV. Expects columns: date, home_team, away_team,
    home_score, away_score, tournament, city, country, neutral."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean_results(df: pd.DataFrame, min_year: int = 2010) -> pd.DataFrame:
    """Clean and filter match results.

    Steps:
        1. Map team names to FIFA 3-letter codes
        2. Filter to matches after min_year
        3. Add result column (H/D/A)
        4. Add match_id column
        5. Drop rows with missing scores
    """
    df = df.copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["date"].dt.year >= min_year]

    df["home_code"] = df["home_team"].apply(resolve_team_code)
    df["away_code"] = df["away_team"].apply(resolve_team_code)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Result from home team perspective
    df["result"] = "D"
    df.loc[df["home_score"] > df["away_score"], "result"] = "H"
    df.loc[df["home_score"] < df["away_score"], "result"] = "A"

    # Unique match ID
    df["match_id"] = (
        df["date"].dt.strftime("%Y-%m-%d")
        + "_"
        + df["home_code"]
        + "_"
        + df["away_code"]
    )

    return df.reset_index(drop=True)


def load_processed_results() -> pd.DataFrame:
    """Load the cleaned parquet if it exists."""
    if PROCESSED_PATH.exists():
        return pd.read_parquet(PROCESSED_PATH)
    raise FileNotFoundError(f"{PROCESSED_PATH} not found. Run the data pipeline first.")


def build_results_dataset(raw_path: Path = RAW_PATH, min_year: int = 2010) -> pd.DataFrame:
    """Full pipeline: load raw → clean → save parquet → return."""
    df = load_raw_results(raw_path)
    df = clean_results(df, min_year=min_year)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df
