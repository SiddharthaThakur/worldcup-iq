"""
Tests for the club-level validation gate harness (pure logic pieces).
"""

from src.evaluation.club_validation import gate_decision, normalized_log_strength


def test_normalized_log_strength_monotonic_and_bounded():
    vals = {"a": 1_000_000, "b": 100_000_000, "c": 10_000_000}
    norm = normalized_log_strength(vals)
    assert 0.0 <= min(norm.values()) and max(norm.values()) <= 1.0
    assert norm["b"] > norm["c"] > norm["a"]


def test_normalized_log_strength_handles_missing_and_zero():
    vals = {"a": 0, "b": None, "c": 5_000_000}
    norm = normalized_log_strength(vals)
    assert all(0.0 <= v <= 1.0 for v in norm.values())


def test_gate_passes_when_blend_improves_and_correlates():
    d = gate_decision(brier_elo=0.220, brier_blend=0.210,
                      best_lambda=0.4, correlation=0.45)
    assert d["passes"] is True
    assert "improves" in d["reason"].lower() or "pass" in d["reason"].lower()


def test_gate_fails_when_blend_does_not_help():
    d = gate_decision(brier_elo=0.220, brier_blend=0.221,
                      best_lambda=0.0, correlation=0.45)
    assert d["passes"] is False


def test_gate_fails_on_weak_correlation():
    # Correlation below 0.3 fails regardless of a tiny Brier gain
    d = gate_decision(brier_elo=0.220, brier_blend=0.219,
                      best_lambda=0.2, correlation=0.12)
    assert d["passes"] is False
