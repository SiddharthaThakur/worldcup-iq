"""
Live self-scoring scorecard helpers + championship movement.

What this does in simple English:
    Once a match is played, we grade every model's pre-match prediction
    with the Brier score (squared error of the probabilities; 0 = perfect,
    0.667 = a useless coin-flip). Averaged over all played games, this is
    the running scorecard — the honest answer to "is this thing any good?"

    We also track how each team's title chance MOVES day to day, so the
    page can show ▲/▼ instead of just refreshing numbers.
"""

_ONEHOT = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}


def match_brier(probs: dict, actual: str) -> float:
    """Brier score for one match. probs has keys H/D/A; actual in H/D/A."""
    o = _ONEHOT[actual]
    p = (probs["H"], probs["D"], probs["A"])
    return sum((pi - oi) ** 2 for pi, oi in zip(p, o)) / 3.0


def running_brier(predictions: list[tuple[dict, str]]) -> float | None:
    """Mean Brier over (probs, actual) pairs. None if empty."""
    if not predictions:
        return None
    return sum(match_brier(p, a) for p, a in predictions) / len(predictions)


def movement_vs_previous(today: dict, previous: dict) -> dict:
    """Per-team change in championship probability since the last snapshot.

    Returns {team: delta} where delta is None if the team had no prior value.
    """
    out = {}
    for team, val in today.items():
        out[team] = (val - previous[team]) if team in previous else None
    return out
