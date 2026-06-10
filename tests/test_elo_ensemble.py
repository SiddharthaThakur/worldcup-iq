"""
Tests for ensembling our Elo with an independent rating system.
The independent ratings are rescaled to our distribution before averaging.
"""

from src.data.elo_ensemble import align_and_ensemble


def test_ensemble_averages_after_scale_alignment():
    ours = {"A": 2000, "B": 1500, "C": 1000}
    # Independent system, different scale (mean/spread) but same ordering
    indep = {"A": 100, "B": 50, "C": 0}
    out = align_and_ensemble(ours, indep)
    # After rescaling indep to ours' mean/std, A>B>C preserved, and B (the
    # middle, matching both means) stays ~1500
    assert out["A"] > out["B"] > out["C"]
    assert abs(out["B"] - 1500) < 1.0


def test_ensemble_reduces_to_ours_when_indep_missing():
    ours = {"A": 2000, "B": 1500, "C": 1000, "D": 1750}
    # Different shape (A and B close together) so rescale != ours
    indep = {"A": 100, "B": 95, "C": 0}  # D missing in indep
    out = align_and_ensemble(ours, indep)
    assert out["D"] == 1750          # no indep rating -> our value unchanged
    assert abs(out["A"] - 2000) > 1e-6  # A blended with rescaled indep


def test_identical_systems_unchanged():
    ours = {"A": 1800, "B": 1400, "C": 1600}
    out = align_and_ensemble(ours, dict(ours))
    for k in ours:
        assert abs(out[k] - ours[k]) < 1e-6
