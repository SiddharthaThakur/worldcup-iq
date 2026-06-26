"""
Monte Carlo simulator for the 2026 World Cup's new 48-team format.

What this does in simple English:
    To answer "who wins the World Cup?", we simulate the entire tournament
    thousands of times. Each simulation plays out all 104 matches by sampling
    scorelines from our model's probability distributions, applies FIFA's
    actual advancement rules, and records who lifts the trophy. Run it 10,000
    times and you get champion probabilities, round-by-round advancement odds
    for every team, and full bracket distributions.

    The 2026 format is NEW and fiddly — nobody's old 32-team code works:
    - 12 groups of 4 (groups A through L)
    - Top 2 from each group advance (24 teams)
    - The 8 BEST third-place teams also advance (ranked by points, then
      goal difference, then goals scored) → 32 teams total
    - Round of 32 → Round of 16 → QF → SF → Final
    - Knockout draws go to extra time, then penalties

Knockout advancement model:
    Dixon-Coles gives P(win), P(draw), P(loss) for 90 minutes. For knockouts,
    the draw probability must be resolved. We model extra time as a shortened
    match (same strength ratio, ~1/3 the goals) and penalties as a coin flip
    with a small edge to the stronger team. This decomposition is standard
    in the literature and matters: naively splitting draws 50/50 overrates
    underdogs in knockouts.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.dixon_coles import (
    DixonColesParams,
    outcome_probs,
    scoreline_matrix,
    strengths_to_expected_goals,
)
from src.simulation.fifa_annex_c import lookup_assignment

N_GROUPS = 12
GROUP_NAMES = "ABCDEFGHIJKL"
THIRD_PLACE_QUALIFIERS = 8


@dataclass
class GroupResult:
    """Final standings of one group after simulation."""

    group: str
    standings: list[dict]  # sorted: [{team, points, gd, gf}, ...]


@dataclass
class SimulationConfig:
    n_sims: int = 10_000
    extra_time_goal_factor: float = 0.33  # ET expected goals ≈ 1/3 of 90-min rate
    penalty_strength_edge: float = 0.03   # stronger team's edge per 100 Elo, capped
    penalty_edge_cap: float = 0.10        # max deviation from 50/50
    seed: int | None = 42


def simulate_group_match(
    strength_a: float, strength_b: float,
    params: DixonColesParams, rng: np.random.Generator,
    neutral: bool = True,
) -> tuple[int, int]:
    """Sample one scoreline from the model's distribution."""
    lam_a, lam_b = strengths_to_expected_goals(strength_a, strength_b, params, neutral)
    probs = scoreline_matrix(lam_a, lam_b, params.rho)
    flat_idx = rng.choice(probs.size, p=probs.ravel())
    home_goals, away_goals = np.unravel_index(flat_idx, probs.shape)
    return int(home_goals), int(away_goals)


def simulate_match_oriented(
    team_a: str, team_b: str,
    strengths: dict[str, float],
    host_teams: set[str],
    params: DixonColesParams,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample a scoreline with home advantage applied to the HOST team.

    The fitted home_adv parameter boosts the first (home) argument of the
    goal model — so when the host is team_b, we simulate with the host
    first and flip the scoreline back. Without this, home advantage would
    go to whichever team happened to be listed first.
    """
    a_host, b_host = team_a in host_teams, team_b in host_teams
    if b_host and not a_host:
        gb, ga = simulate_group_match(
            strengths[team_b], strengths[team_a], params, rng, neutral=False)
        return ga, gb
    neutral = not a_host  # both hosts can't meet in a group; treat as a-home
    return simulate_group_match(
        strengths[team_a], strengths[team_b], params, rng, neutral=neutral)


def resolve_knockout(
    team_a: str, team_b: str,
    strengths: dict[str, float],
    host_teams: set[str],
    params: DixonColesParams, rng: np.random.Generator,
    config: SimulationConfig,
) -> bool:
    """Simulate a knockout match. Returns True if team A advances.

    90 minutes → if draw, extra time (reduced goal rate) → if still
    level, penalties (near-coin-flip with small strength edge).
    """
    strength_a, strength_b = strengths[team_a], strengths[team_b]
    hg, ag = simulate_match_oriented(team_a, team_b, strengths, host_teams, params, rng)
    if hg != ag:
        return hg > ag
    neutral = not (team_a in host_teams or team_b in host_teams)

    # Extra time: same strength ratio, reduced goal expectation.
    # Orient home advantage to the host, mirroring simulate_match_oriented.
    if team_b in host_teams and team_a not in host_teams:
        lam_b, lam_a = strengths_to_expected_goals(strength_b, strength_a, params, False)
    else:
        lam_a, lam_b = strengths_to_expected_goals(strength_a, strength_b, params, neutral)
    et_a = rng.poisson(lam_a * config.extra_time_goal_factor)
    et_b = rng.poisson(lam_b * config.extra_time_goal_factor)
    if et_a != et_b:
        return et_a > et_b

    # Penalties: 50/50 plus a small, capped edge for the stronger team
    edge = np.clip(
        (strength_a - strength_b) / 100.0 * config.penalty_strength_edge,
        -config.penalty_edge_cap, config.penalty_edge_cap,
    )
    return bool(rng.random() < 0.5 + edge)


def _resolve_h2h_tie(
    tied: list[dict], h2h: dict, rng: np.random.Generator,
) -> list[dict]:
    """Break a tie using head-to-head results among the tied teams.

    FIFA 2026 tiebreaker after equal points (changed from prior WCs):
      1. H2H points among tied teams
      2. H2H goal difference
      3. H2H goals scored
      4. Overall goal difference (all group matches)
      5. Overall goals scored
      6. Random (proxy for fair play / FIFA ranking / lots)
    """
    tied_teams = {t["team"] for t in tied}
    stats = {t["team"]: {"pts": 0, "gd": 0, "gf": 0} for t in tied}
    for t1 in tied_teams:
        for t2 in tied_teams:
            if t1 == t2:
                continue
            ga, gb = h2h[(t1, t2)]
            stats[t1]["gf"] += ga
            stats[t1]["gd"] += ga - gb
            if ga > gb:
                stats[t1]["pts"] += 3
            elif ga == gb:
                stats[t1]["pts"] += 1
    return sorted(
        tied,
        key=lambda t: (stats[t["team"]]["pts"], stats[t["team"]]["gd"],
                        stats[t["team"]]["gf"],
                        t["gd"], t["gf"],
                        rng.random()),
        reverse=True,
    )


def play_group(
    teams: list[str],
    strengths: dict[str, float],
    host_teams: set[str],
    params: DixonColesParams,
    rng: np.random.Generator,
    completed: dict | None = None,
) -> GroupResult:
    """Simulate all 6 matches of a 4-team group and return final standings.

    `completed` maps frozenset({team_a, team_b}) -> {team_a: goals, team_b:
    goals} for matches already PLAYED. Those use their real score; only the
    remaining matches are simulated. This is how live odds sharpen toward
    reality as the group stage unfolds.

    Tiebreakers (FIFA 2026 — H2H before overall GD, changed from prior WCs):
    points → H2H points → H2H GD → H2H GF → overall GD → overall GF → random.
    """
    completed = completed or {}
    table = {t: {"team": t, "points": 0, "gd": 0, "gf": 0} for t in teams}
    h2h: dict[tuple[str, str], tuple[int, int]] = {}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            real = completed.get(frozenset({a, b}))
            if real is not None:
                ga, gb = real[a], real[b]
            else:
                ga, gb = simulate_match_oriented(a, b, strengths, host_teams, params, rng)
            h2h[(a, b)] = (ga, gb)
            h2h[(b, a)] = (gb, ga)
            table[a]["gf"] += ga; table[a]["gd"] += ga - gb
            table[b]["gf"] += gb; table[b]["gd"] += gb - ga
            if ga > gb:
                table[a]["points"] += 3
            elif gb > ga:
                table[b]["points"] += 3
            else:
                table[a]["points"] += 1; table[b]["points"] += 1

    def points_key(r):
        return r["points"]

    team_list = sorted(table.values(), key=points_key, reverse=True)

    result = []
    i = 0
    while i < len(team_list):
        j = i + 1
        while j < len(team_list) and points_key(team_list[j]) == points_key(team_list[i]):
            j += 1
        tied = team_list[i:j]
        if len(tied) > 1:
            tied = _resolve_h2h_tie(tied, h2h, rng)
        result.extend(tied)
        i = j

    return GroupResult(group="", standings=result)


def rank_third_place(group_results: list[GroupResult],
                     rng: np.random.Generator) -> list[str]:
    """Rank the 12 third-place teams; best 8 advance."""
    thirds = [gr.standings[2] for gr in group_results]
    thirds_sorted = sorted(
        thirds, key=lambda r: (r["points"], r["gd"], r["gf"], rng.random()),
        reverse=True,
    )
    return [r["team"] for r in thirds_sorted[:THIRD_PLACE_QUALIFIERS]]


_R32_MATCHES = [
    (73, "R", "A", "R", "B"),
    (74, "W", "E", "T", "74"),
    (75, "W", "F", "R", "C"),
    (76, "W", "C", "R", "F"),
    (77, "W", "I", "T", "77"),
    (78, "R", "E", "R", "I"),
    (79, "W", "A", "T", "79"),
    (80, "W", "L", "T", "80"),
    (81, "W", "D", "T", "81"),
    (82, "W", "G", "T", "82"),
    (83, "R", "K", "R", "L"),
    (84, "W", "H", "R", "J"),
    (85, "W", "B", "T", "85"),
    (86, "W", "J", "R", "H"),
    (87, "W", "K", "T", "87"),
    (88, "R", "D", "R", "G"),
]
_R16_PAIRS = [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80),
              (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)]
_QF_PAIRS = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
_SF_PAIRS = [(101, 97, 98), (102, 99, 100)]
_FINAL_PAIR = (104, 101, 102)


def build_round_of_32(group_results: list[GroupResult],
                      rng: np.random.Generator) -> dict[int, tuple[str, str]]:
    """Build R32 ties using the actual FIFA bracket and Annex C table.

    Returns {match_number: (team_a, team_b)}.
    """
    by_group = {gr.group: gr for gr in group_results}
    team_group = {}
    for gr in group_results:
        for r in gr.standings:
            team_group[r["team"]] = gr.group

    thirds = [gr.standings[2] for gr in group_results]
    best = sorted(thirds, key=lambda r: (r["points"], r["gd"], r["gf"], rng.random()),
                  reverse=True)[:THIRD_PLACE_QUALIFIERS]
    qual_groups = {team_group[r["team"]] for r in best}
    third_by_group = {team_group[r["team"]]: r["team"] for r in best}

    assignment = lookup_assignment(qual_groups)

    ties = {}
    for match_no, kind_a, ref_a, kind_b, ref_b in _R32_MATCHES:
        a = _resolve_slot(kind_a, ref_a, by_group, third_by_group, assignment, match_no)
        b = _resolve_slot(kind_b, ref_b, by_group, third_by_group, assignment, match_no)
        ties[match_no] = (a, b)
    return ties


def _resolve_slot(kind, ref, by_group, third_by_group, assignment, match_no):
    if kind == "W":
        return by_group[ref].standings[0]["team"]
    if kind == "R":
        return by_group[ref].standings[1]["team"]
    g = assignment[match_no]
    return third_by_group[g]


def simulate_tournament_once(
    groups: dict[str, list[str]],
    strengths: dict[str, float],
    host_teams: set[str],
    params: DixonColesParams,
    rng: np.random.Generator,
    config: SimulationConfig,
    completed: dict | None = None,
) -> dict[str, str]:
    """One full tournament simulation. Returns {team: furthest_stage_reached}."""
    reached: dict[str, str] = {t: "group" for ts in groups.values() for t in ts}

    group_results = []
    for gname, teams in groups.items():
        gr = play_group(teams, strengths, host_teams, params, rng, completed=completed)
        gr.group = gname
        group_results.append(gr)

    r32 = build_round_of_32(group_results, rng)

    match_winner: dict[int, str] = {}

    for match_no, (a, b) in r32.items():
        reached[a] = "r32"
        reached[b] = "r32"
        a_wins = resolve_knockout(a, b, strengths, host_teams, params, rng, config)
        match_winner[match_no] = a if a_wins else b

    for stage, pairs in [("r16", _R16_PAIRS), ("qf", _QF_PAIRS),
                         ("sf", _SF_PAIRS), ("final", [_FINAL_PAIR])]:
        for match_no, feed_a, feed_b in pairs:
            a, b = match_winner[feed_a], match_winner[feed_b]
            reached[a] = stage
            reached[b] = stage
            a_wins = resolve_knockout(a, b, strengths, host_teams, params, rng, config)
            match_winner[match_no] = a if a_wins else b

    reached[match_winner[_FINAL_PAIR[0]]] = "champion"
    return reached


def run_simulation(
    groups: dict[str, list[str]],
    strengths: dict[str, float],
    host_teams: set[str] | None = None,
    params: DixonColesParams | None = None,
    config: SimulationConfig | None = None,
    completed: dict | None = None,
) -> pd.DataFrame:
    """Run the full Monte Carlo and return per-team advancement probabilities.

    `completed` (optional) fixes already-played group games so the odds
    reflect real results so far. Returns a DataFrame with columns:
        team, p_r32, p_r16, p_qf, p_sf, p_final, p_champion
    sorted by champion probability.
    """
    config = config or SimulationConfig()
    params = params or DixonColesParams.load()
    host_teams = host_teams or {"USA", "CAN", "MEX"}
    rng = np.random.default_rng(config.seed)

    stage_order = ["group", "r32", "r16", "qf", "sf", "final", "champion"]
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(stage_order)))

    for _ in range(config.n_sims):
        reached = simulate_tournament_once(groups, strengths, host_teams, params, rng,
                                           config, completed=completed)
        for team, stage in reached.items():
            idx = stage_order.index(stage)
            counts[team][: idx + 1] += 1  # reaching SF implies reaching QF, etc.

    rows = []
    for team, c in counts.items():
        probs = c / config.n_sims
        rows.append({
            "team": team,
            "p_r32": probs[1], "p_r16": probs[2], "p_qf": probs[3],
            "p_sf": probs[4], "p_final": probs[5], "p_champion": probs[6],
        })
    return (pd.DataFrame(rows)
            .sort_values("p_champion", ascending=False)
            .reset_index(drop=True))
