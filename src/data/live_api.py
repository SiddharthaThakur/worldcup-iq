"""
Client for the free World Cup 2026 REST API.

What this does in simple English:
    During the actual 2026 World Cup (June 11 – July 19), this module
    polls a free API to get live scores, match results, and group
    standings. We use this to update our predictions in real-time and
    track our calibration scorecard.

Source: https://github.com/rezarahiminia/worldcup2026
"""

from pathlib import Path

import requests

# The free API base URL — may need updating when the tournament starts
API_BASE = "https://worldcupjson.net"  # placeholder — verify before tournament

FIXTURES_CACHE = Path("data/raw/wc2026_fixtures.json")
RESULTS_CACHE = Path("data/processed/wc2026_results.json")


def get_matches() -> list[dict]:
    """Fetch all 104 World Cup matches (scheduled + completed)."""
    # TODO: implement when API is live
    # resp = requests.get(f"{API_BASE}/matches", timeout=10)
    # resp.raise_for_status()
    # return resp.json()
    raise NotImplementedError("Phase 5: implement when tournament starts June 11")


def get_today_matches() -> list[dict]:
    """Fetch today's matches only."""
    raise NotImplementedError("Phase 5: implement when tournament starts June 11")


def get_group_standings() -> list[dict]:
    """Fetch current group stage standings."""
    raise NotImplementedError("Phase 5: implement when tournament starts June 11")


def get_match_result(match_id: str) -> dict:
    """Fetch result for a specific completed match."""
    raise NotImplementedError("Phase 5: implement when tournament starts June 11")
