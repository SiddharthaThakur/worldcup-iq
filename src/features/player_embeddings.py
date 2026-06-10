"""
Build player feature vectors from club season stats.

What this does in simple English:
    Each player gets a list of numbers that describe how they play — how
    many goals they score per game, how many tackles they make, how creative
    they are with passes, etc. All stats are "per 90 minutes" so we can
    fairly compare a player who played 3000 minutes to one who played 1500.

    We normalize these numbers within each position group (defenders compared
    to defenders, forwards to forwards) so a defender with 2 goals per 90
    is recognized as exceptional even though a forward might have 8.

    These player vectors are the building blocks for the team composition
    models — we combine 11 players' vectors into a team representation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PLAYER_STATS_PATH = Path("data/processed/player_stats.parquet")

# Feature groups — what stats we use for each aspect of player quality
OFFENSIVE_FEATURES = [
    "goals_per90", "assists_per90", "xg_per90", "xa_per90",
    "shots_per90", "shots_on_target_per90",
]

CREATIVE_FEATURES = [
    "key_passes_per90", "progressive_passes_per90",
    "progressive_carries_per90", "xg_chain_per90", "xg_buildup_per90",
]

DEFENSIVE_FEATURES = [
    "tackles_per90", "interceptions_per90", "blocks_per90",
    "clearances_per90", "pressures_per90",
]

PHYSICAL_FEATURES = [
    "minutes_played",  # raw total, not per-90 (proxy for importance/fitness)
]

ALL_FEATURES = OFFENSIVE_FEATURES + CREATIVE_FEATURES + DEFENSIVE_FEATURES + PHYSICAL_FEATURES

POSITION_GROUPS = {
    "GK": ["GK"],
    "DEF": ["DF", "CB", "LB", "RB", "WB"],
    "MID": ["MF", "CM", "DM", "AM", "LM", "RM"],
    "FWD": ["FW", "CF", "LW", "RW", "SS"],
}


def map_to_position_group(position: str) -> str:
    """Map a specific position to a position group (GK/DEF/MID/FWD)."""
    pos_upper = position.upper().strip()
    for group, positions in POSITION_GROUPS.items():
        if pos_upper in positions:
            return group
    # Default: try to infer from first character
    if pos_upper.startswith("G"):
        return "GK"
    elif pos_upper.startswith("D"):
        return "DEF"
    elif pos_upper.startswith("M"):
        return "MID"
    elif pos_upper.startswith("F") or pos_upper.startswith("A"):
        return "FWD"
    return "MID"  # fallback


def build_player_embeddings(
    player_stats: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Build normalized player feature vectors.

    Args:
        player_stats: DataFrame with columns including canonical_id, position,
                      and the stat columns listed in ALL_FEATURES
        features: which features to include (default: ALL_FEATURES)

    Returns:
        DataFrame with canonical_id, position_group, data_quality, and
        normalized feature columns (prefixed with "emb_")
    """
    if features is None:
        features = ALL_FEATURES

    df = player_stats.copy()
    df["position_group"] = df["position"].apply(map_to_position_group)

    # Identify which features are available
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        # Fill missing columns with NaN (will be imputed)
        for col in missing:
            df[col] = np.nan

    # Assign data quality based on feature coverage
    df["data_quality"] = "full"
    n_missing = df[features].isna().sum(axis=1)
    df.loc[n_missing > len(features) * 0.3, "data_quality"] = "partial"
    df.loc[n_missing > len(features) * 0.7, "data_quality"] = "minimal"

    # Impute missing values with position-group median
    for group in df["position_group"].unique():
        mask = df["position_group"] == group
        group_df = df.loc[mask, features]
        medians = group_df.median()
        df.loc[mask, features] = group_df.fillna(medians)

    # Global fallback for any remaining NaN
    df[features] = df[features].fillna(0.0)

    # Normalize within position group (z-score)
    for group in df["position_group"].unique():
        mask = df["position_group"] == group
        if mask.sum() < 3:
            continue
        scaler = StandardScaler()
        df.loc[mask, features] = scaler.fit_transform(df.loc[mask, features])

    # Rename to embedding columns
    rename_map = {f: f"emb_{f}" for f in features}
    result = df[["canonical_id", "position_group", "data_quality"] + features].rename(columns=rename_map)

    return result


def compose_team_vector(
    player_ids: list[str],
    embeddings: pd.DataFrame,
) -> np.ndarray:
    """Compose 11 player embeddings into one team vector.

    Strategy: average embeddings within each position group (GK/DEF/MID/FWD),
    then concatenate. This gives a fixed-size team vector regardless of formation.

    Args:
        player_ids: list of 11 canonical player IDs
        embeddings: DataFrame from build_player_embeddings

    Returns:
        1D numpy array of shape (4 * n_features,)
    """
    team = embeddings[embeddings["canonical_id"].isin(player_ids)]
    emb_cols = [c for c in team.columns if c.startswith("emb_")]

    group_vectors = []
    for group in ["GK", "DEF", "MID", "FWD"]:
        group_players = team[team["position_group"] == group]
        if group_players.empty:
            group_vectors.append(np.zeros(len(emb_cols)))
        else:
            group_vectors.append(group_players[emb_cols].mean().values)

    return np.concatenate(group_vectors)
