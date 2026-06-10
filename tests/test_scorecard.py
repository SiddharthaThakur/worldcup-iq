"""
Tests for the live self-scoring scorecard: matching completed results to
predictions, computing running Brier, and tracking championship movement.
"""

from src.live.scorecard import match_brier, movement_vs_previous, running_brier


def test_match_brier_perfect_prediction():
    # Predicted home win with certainty, home won -> Brier 0
    assert match_brier({"H": 1.0, "D": 0.0, "A": 0.0}, "H") == 0.0


def test_match_brier_worst_prediction():
    # Predicted away with certainty, home won -> Brier 2/3 (max for one match)
    b = match_brier({"H": 0.0, "D": 0.0, "A": 1.0}, "H")
    assert abs(b - (2.0 / 3.0)) < 1e-9


def test_running_brier_averages():
    preds = [
        ({"H": 0.6, "D": 0.25, "A": 0.15}, "H"),
        ({"H": 0.2, "D": 0.3, "A": 0.5}, "A"),
    ]
    rb = running_brier(preds)
    expected = (match_brier(*preds[0]) + match_brier(*preds[1])) / 2
    assert abs(rb - expected) < 1e-9


def test_running_brier_empty_is_none():
    assert running_brier([]) is None


def test_movement_computes_deltas():
    today = {"ESP": 0.14, "ARG": 0.11, "BRA": 0.09}
    prev = {"ESP": 0.13, "ARG": 0.12, "BRA": 0.09}
    mv = movement_vs_previous(today, prev)
    assert abs(mv["ESP"] - 0.01) < 1e-9     # up
    assert abs(mv["ARG"] - (-0.01)) < 1e-9  # down
    assert abs(mv["BRA"]) < 1e-9            # flat


def test_movement_new_team_has_no_delta():
    mv = movement_vs_previous({"ESP": 0.14}, {})
    assert mv["ESP"] is None  # no prior snapshot
