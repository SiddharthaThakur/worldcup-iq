"""
Tests for the EA-based Phase 3 re-test pieces: EA position mapping and
the bootstrap confidence interval on the Brier difference.
"""

import numpy as np

from src.evaluation.ea_validation import bootstrap_brier_diff_ci, ea_position_to_group


def test_ea_position_mapping():
    assert ea_position_to_group("GK") == "GK"
    assert ea_position_to_group("CB") == "DF"
    assert ea_position_to_group("RWB") == "DF"
    assert ea_position_to_group("CDM") == "MF"
    assert ea_position_to_group("CAM") == "MF"
    assert ea_position_to_group("ST") == "FW"
    assert ea_position_to_group("LW") == "FW"


def test_bootstrap_ci_detects_real_improvement():
    # Blend per-game Brier is consistently ~0.01 lower than Elo's
    rng = np.random.default_rng(0)
    elo_b = rng.uniform(0.15, 0.30, size=4000)
    blend_b = elo_b - 0.01
    lo, hi, mean = bootstrap_brier_diff_ci(elo_b, blend_b, n_boot=500, seed=1)
    # Improvement = elo - blend ≈ +0.01, CI should exclude 0
    assert mean > 0
    assert lo > 0


def test_bootstrap_ci_includes_zero_when_no_difference():
    rng = np.random.default_rng(2)
    elo_b = rng.uniform(0.15, 0.30, size=4000)
    noise = rng.normal(0, 0.05, size=4000)
    noise -= noise.mean()  # exactly zero-mean difference -> no real gain
    blend_b = elo_b + noise
    lo, hi, mean = bootstrap_brier_diff_ci(elo_b, blend_b, n_boot=500, seed=3)
    assert lo < 0 < hi
