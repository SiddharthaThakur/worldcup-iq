"""
Tests for the match-context adjustment layer: altitude + rest/travel.
Adjustments are in Elo points, applied to each team for a specific match.
"""

from src.models.match_context import (
    HOST_CITIES,
    altitude_elo_adjustment,
    haversine_km,
    rest_travel_elo_adjustment,
)


def test_all_16_host_cities_have_elevation_and_coords():
    assert len(HOST_CITIES) == 16
    for city, info in HOST_CITIES.items():
        assert "elev_m" in info and "lat" in info and "lon" in info


def test_altitude_penalizes_lowland_team_at_azteca():
    # Lowland team (Germany) visiting Mexico City (2240m) gets a negative adj
    adj = altitude_elo_adjustment("GER", "Mexico City")
    assert adj < 0


def test_altitude_no_penalty_for_adapted_nation():
    # Mexico is altitude-adapted -> no penalty at home altitude
    assert altitude_elo_adjustment("MEX", "Mexico City") == 0.0
    # Ecuador (Andean) also adapted
    assert altitude_elo_adjustment("ECU", "Mexico City") == 0.0


def test_altitude_zero_at_sea_level_venue():
    assert altitude_elo_adjustment("GER", "Miami Gardens") == 0.0
    assert altitude_elo_adjustment("BRA", "East Rutherford") == 0.0


def test_haversine_known_distance():
    # NYC (East Rutherford) to LA (Inglewood) ~ 3900 km
    d = haversine_km(40.81, -74.07, 33.95, -118.34)
    assert 3800 < d < 4050


def test_rest_advantage_for_more_rested_team():
    # Team A had 4 rest days, B had 2 -> A gets positive adjustment
    adj_a = rest_travel_elo_adjustment(rest_days=4, travel_km=0,
                                       opp_rest_days=2, opp_travel_km=0)
    assert adj_a > 0


def test_travel_penalty():
    # Long travel, equal rest -> negative adjustment
    adj = rest_travel_elo_adjustment(rest_days=3, travel_km=4000,
                                     opp_rest_days=3, opp_travel_km=0)
    assert adj < 0


def test_no_differential_means_zero():
    adj = rest_travel_elo_adjustment(rest_days=3, travel_km=1000,
                                     opp_rest_days=3, opp_travel_km=1000)
    assert abs(adj) < 1e-9
