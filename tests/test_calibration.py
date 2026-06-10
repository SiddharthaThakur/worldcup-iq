"""Tests for calibration and evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.calibration import (
    PredictionRecord,
    brier_score,
    log_loss,
    ranked_probability_score,
    accuracy,
    scorecard,
)


def _make_pred(prob_h, prob_d, prob_a, result, match_id="test"):
    return PredictionRecord(
        match_id=match_id, model_name="test",
        prob_home=prob_h, prob_draw=prob_d, prob_away=prob_a,
        actual_result=result,
    )


class TestBrierScore:
    def test_perfect_prediction(self):
        preds = [_make_pred(1.0, 0.0, 0.0, "H")]
        assert brier_score(preds) == pytest.approx(0.0)

    def test_worst_prediction(self):
        preds = [_make_pred(0.0, 0.0, 1.0, "H")]
        bs = brier_score(preds)
        assert bs > 0.5

    def test_uniform_prediction(self):
        preds = [_make_pred(1/3, 1/3, 1/3, "H")]
        bs = brier_score(preds)
        assert bs == pytest.approx(2/9, abs=0.01)

    def test_empty_predictions(self):
        assert np.isnan(brier_score([]))


class TestLogLoss:
    def test_perfect_prediction(self):
        preds = [_make_pred(0.99, 0.005, 0.005, "H")]
        ll = log_loss(preds)
        assert ll < 0.1

    def test_uniform_prediction(self):
        preds = [_make_pred(1/3, 1/3, 1/3, "H")]
        ll = log_loss(preds)
        assert ll == pytest.approx(np.log(3), abs=0.05)

    def test_clipping_prevents_infinity(self):
        """Even P=0 for the actual outcome shouldn't cause infinity."""
        preds = [_make_pred(0.0, 0.0, 1.0, "H")]
        ll = log_loss(preds, clip=0.01)
        assert np.isfinite(ll)


class TestRPS:
    def test_perfect_prediction(self):
        preds = [_make_pred(1.0, 0.0, 0.0, "H")]
        assert ranked_probability_score(preds) == pytest.approx(0.0)

    def test_adjacent_error_less_than_distant(self):
        """Predicting home win when draw occurs should be less penalized
        than predicting home win when away win occurs."""
        preds_draw = [_make_pred(0.8, 0.1, 0.1, "D")]
        preds_away = [_make_pred(0.8, 0.1, 0.1, "A")]
        assert ranked_probability_score(preds_draw) < ranked_probability_score(preds_away)


class TestAccuracy:
    def test_correct_prediction(self):
        preds = [_make_pred(0.6, 0.2, 0.2, "H")]
        assert accuracy(preds) == 1.0

    def test_incorrect_prediction(self):
        preds = [_make_pred(0.6, 0.2, 0.2, "A")]
        assert accuracy(preds) == 0.0


class TestScorecard:
    def test_scorecard_structure(self):
        preds = [
            _make_pred(0.6, 0.25, 0.15, "H", "m1"),
            _make_pred(0.3, 0.4, 0.3, "D", "m2"),
            _make_pred(0.2, 0.3, 0.5, "A", "m3"),
        ]
        card = scorecard(preds, "TestModel")
        assert card["model"] == "TestModel"
        assert card["n_matches"] == 3
        assert "brier_score" in card
        assert "log_loss" in card
        assert "rps" in card
        assert "accuracy" in card
