"""
Tests for the Phase 4 lineup-feature dataset builder (pure helpers).
"""

import numpy as np

from src.features.lineup_dataset import (
    POS_GROUP_IDX,
    impute_missing_players,
    tm_position_to_group,
)


def test_position_mapping_covers_known_labels():
    assert tm_position_to_group("Goalkeeper") == "GK"
    assert tm_position_to_group("Centre-Back") == "DEF"
    assert tm_position_to_group("Left-Back") == "DEF"
    assert tm_position_to_group("Defensive Midfield") == "MID"
    assert tm_position_to_group("Attacking Midfield") == "MID"
    assert tm_position_to_group("Centre-Forward") == "FWD"
    assert tm_position_to_group("Left Winger") == "FWD"


def test_position_index_complete():
    assert set(POS_GROUP_IDX) == {"GK", "DEF", "MID", "FWD"}


def test_impute_fills_missing_with_group_mean():
    # 3 players: two DEF with features, one DEF missing -> imputed to mean
    feats = [np.array([80.0, 70.0]), np.array([60.0, 50.0]), None]
    groups = ["DEF", "DEF", "DEF"]
    out = impute_missing_players(feats, groups, n_features=2)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[2], [70.0, 60.0])  # mean of the two known


def test_impute_uses_global_mean_when_group_empty():
    feats = [np.array([80.0, 70.0]), None]
    groups = ["DEF", "FWD"]  # FWD has no known player
    out = impute_missing_players(feats, groups, n_features=2)
    # FWD missing -> falls back to overall mean (the one known player)
    np.testing.assert_allclose(out[1], [80.0, 70.0])
