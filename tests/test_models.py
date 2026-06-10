"""Tests for the core prediction models (Elo system).

Dixon-Coles tests moved to test_v2_patches.py after the v2 rewrite —
the model now requires FITTED parameters and the old hand-constant
API (elo_to_expected_goals) was deliberately removed.
"""

import numpy as np
import pytest

from src.models.elo import EloSystem, DEFAULT_RATING


class TestEloSystem:
    """Tests for the Elo rating system."""

    def test_initial_rating(self):
        elo = EloSystem()
        assert elo.get_rating("ARG") == DEFAULT_RATING
        assert elo.get_rating("NONEXISTENT") == DEFAULT_RATING

    def test_update_home_win(self):
        elo = EloSystem()
        elo.update("ARG", "BRA", 2, 0, tournament="Friendly", neutral=True)
        assert elo.get_rating("ARG") > DEFAULT_RATING
        assert elo.get_rating("BRA") < DEFAULT_RATING

    def test_update_draw(self):
        elo = EloSystem()
        elo.ratings["ARG"] = 1700
        elo.ratings["BRA"] = 1500
        elo.update("ARG", "BRA", 1, 1, tournament="Friendly", neutral=True)
        assert elo.get_rating("ARG") < 1700
        assert elo.get_rating("BRA") > 1500

    def test_update_preserves_total(self):
        """Elo changes should be zero-sum."""
        elo = EloSystem()
        elo.ratings["ARG"] = 1600
        elo.ratings["BRA"] = 1400
        total_before = elo.get_rating("ARG") + elo.get_rating("BRA")
        elo.update("ARG", "BRA", 3, 1, tournament="Friendly", neutral=True)
        total_after = elo.get_rating("ARG") + elo.get_rating("BRA")
        assert abs(total_before - total_after) < 1e-10

    def test_k_factor_world_cup_higher(self):
        elo = EloSystem()
        assert elo.get_k_factor("FIFA World Cup") > elo.get_k_factor("Friendly")

    def test_expected_score_equal_teams(self):
        elo = EloSystem()
        assert abs(elo.expected_score(1500, 1500) - 0.5) < 1e-10

    def test_expected_score_stronger_team(self):
        elo = EloSystem()
        expected = elo.expected_score(1700, 1500)
        assert 0.5 < expected < 1.0
