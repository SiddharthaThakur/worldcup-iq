"""
Match-context adjustments: altitude and rest/travel (Elo points).

What this does in simple English:
    Two real, free edges that team ratings alone miss — especially at a
    World Cup spread across a continent:

    1. ALTITUDE — Mexico City's Estadio Azteca sits at 2,240m. Teams not
       used to thin air tire faster and their lungs do less work; it's a
       documented, physiological home-edge (McSharry, BMJ 2007). We
       penalize teams that are NOT altitude-adapted when they play at a
       high venue, scaled by how high it is. Adapted nations (Mexico, the
       Andean sides) pay no penalty.

    2. REST & TRAVEL — this tournament zig-zags from Vancouver to Mexico
       City to Miami. A team with more rest days and less travel since its
       last match has a small edge. We compute the DIFFERENCE between the
       two teams and nudge accordingly.

    HONESTY NOTE: unlike the model's core (everything fitted, D007), these
    coefficients are literature-informed, NOT fitted from our data — there
    is no rest/travel/altitude signal in the results file to fit on. They
    are deliberately MODEST and capped, so they sharpen specific matches
    without being able to do much harm. Flagged as such (D024).
"""

import math

# 16 host cities: elevation (m) + coordinates. Only Mexican venues are high.
HOST_CITIES = {
    "Toronto":         {"elev_m": 76,   "lat": 43.64, "lon": -79.39},
    "Vancouver":       {"elev_m": 4,    "lat": 49.28, "lon": -123.12},
    "Mexico City":     {"elev_m": 2240, "lat": 19.30, "lon": -99.15},
    "Zapopan":         {"elev_m": 1560, "lat": 20.68, "lon": -103.46},
    "Guadalupe":       {"elev_m": 540,  "lat": 25.67, "lon": -100.24},
    "Inglewood":       {"elev_m": 30,   "lat": 33.95, "lon": -118.34},
    "Santa Clara":     {"elev_m": 7,    "lat": 37.40, "lon": -121.97},
    "East Rutherford": {"elev_m": 3,    "lat": 40.81, "lon": -74.07},
    "Foxborough":      {"elev_m": 88,   "lat": 42.09, "lon": -71.26},
    "Arlington":       {"elev_m": 184,  "lat": 32.75, "lon": -97.09},
    "Houston":         {"elev_m": 15,   "lat": 29.68, "lon": -95.41},
    "Philadelphia":    {"elev_m": 12,   "lat": 39.90, "lon": -75.17},
    "Seattle":         {"elev_m": 56,   "lat": 47.59, "lon": -122.33},
    "Atlanta":         {"elev_m": 320,  "lat": 33.76, "lon": -84.40},
    "Miami Gardens":   {"elev_m": 3,    "lat": 25.96, "lon": -80.24},
    "Kansas City":     {"elev_m": 270,  "lat": 39.05, "lon": -94.48},
}

# Nations whose players live/play at altitude — no penalty at high venues.
HIGH_ALTITUDE_NATIONS = {"MEX", "BOL", "ECU", "COL", "PER"}

ALTITUDE_THRESHOLD_M = 1500
ALTITUDE_ELO_PER_KM = 80.0      # penalty per km above threshold (non-adapted)
REST_ELO_PER_DAY = 8.0          # advantage per extra rest day
REST_DAY_CAP = 3.0
TRAVEL_ELO_PER_1000KM = 6.0     # penalty per 1000 km travelled
TRAVEL_KM_CAP = 5000.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def altitude_elo_adjustment(team: str, venue_city: str) -> float:
    """Elo adjustment for `team` playing at `venue_city` (<= 0).

    Non-adapted teams are penalized at high venues; adapted nations and
    sea-level venues yield 0.
    """
    city = HOST_CITIES.get(venue_city)
    if city is None or city["elev_m"] <= ALTITUDE_THRESHOLD_M:
        return 0.0
    if team in HIGH_ALTITUDE_NATIONS:
        return 0.0
    km_above = (city["elev_m"] - ALTITUDE_THRESHOLD_M) / 1000.0
    return -ALTITUDE_ELO_PER_KM * km_above


def rest_travel_elo_adjustment(rest_days: float, travel_km: float,
                               opp_rest_days: float, opp_travel_km: float) -> float:
    """Elo adjustment from rest/travel DIFFERENTIAL vs the opponent."""
    rest_diff = max(-REST_DAY_CAP, min(REST_DAY_CAP, rest_days - opp_rest_days))
    travel_diff = (min(travel_km, TRAVEL_KM_CAP) - min(opp_travel_km, TRAVEL_KM_CAP)) / 1000.0
    return REST_ELO_PER_DAY * rest_diff - TRAVEL_ELO_PER_1000KM * travel_diff
