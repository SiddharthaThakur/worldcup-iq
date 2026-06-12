"""
Full knockout-bracket projection: who is likely to meet whom, every round.

What this does in simple English:
    We simulate the whole knockout bracket thousands of times using FIFA's
    real 2026 structure (the fixed Round-of-32 matchups, and how winners
    feed into the Round of 16, quarters, semis, final). For every slot in
    every round we record which team landed there, then report the most
    likely occupants. The result is a bracket you can read left-to-right:
    "these teams are most likely to meet here, then the winner most likely
    meets one of these over here", and so on to the final.

    As real games are played the candidates narrow — first in the groups,
    then round by round — so the projected matchups sharpen automatically.

    Honest caveat (D010): the third-place teams are assigned to their R32
    slots by a valid matching that respects FIFA's eligibility groups, not
    FIFA's exact combination table. Deep-round projections are inherently
    uncertain (a team must win several games to get there).
"""

from collections import defaultdict

import numpy as np

from src.models.dixon_coles import DixonColesParams
from src.simulation.bracket import R32_BRACKET
from src.simulation.tournament import SimulationConfig, play_group, resolve_knockout

# Which R32 group-position slots are third-place slots, with eligible groups
THIRD_SLOTS = [(m, frozenset(s[1])) for m, a, b in R32_BRACKET
               for s in (a, b) if s[0] == "T"]

# How winners feed forward (sourced from FIFA's published 2026 bracket)
R16_FEEDERS = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
               93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF_FEEDERS = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF_FEEDERS = {101: (97, 98), 102: (99, 100)}
FINAL_FEEDERS = {104: (101, 102)}
LATER_ROUNDS = [("Round of 16", R16_FEEDERS), ("Quarter-finals", QF_FEEDERS),
                ("Semi-finals", SF_FEEDERS), ("Final", FINAL_FEEDERS)]


def assign_thirds(qualifying_groups) -> dict:
    """Match the 8 qualifying third-place groups to the 8 third-place slots.

    Bipartite matching that respects each slot's eligible groups. Returns
    {match_no: group_letter}. A valid assignment always exists for any real
    qualifying combination.
    """
    groups = list(qualifying_groups)
    grp_to_slot = {}  # group -> slot index

    def augment(si, seen):
        for grp in groups:
            if grp in THIRD_SLOTS[si][1] and grp not in seen:
                seen.add(grp)
                cur = grp_to_slot.get(grp)
                if cur is None or augment(cur, seen):
                    grp_to_slot[grp] = si
                    return True
        return False

    for si in range(len(THIRD_SLOTS)):
        augment(si, set())
    return {THIRD_SLOTS[si][0]: grp for grp, si in grp_to_slot.items()}


def _leaf_order():
    """Match order per round so the tree reads top-to-bottom (feeders aligned)."""
    order = {"Final": [104]}
    for name, feeders in [("Semi-finals", FINAL_FEEDERS), ("Quarter-finals", SF_FEEDERS),
                          ("Round of 16", QF_FEEDERS), ("Round of 32", R16_FEEDERS)]:
        parent = {"Semi-finals": [104], "Quarter-finals": [101, 102],
                  "Round of 16": [97, 98, 99, 100],
                  "Round of 32": [89, 90, 93, 94, 91, 92, 95, 96]}[name]
        seq = []
        for p in parent:
            seq.extend(feeders[p])
        order[name] = seq
    return order


def simulate_bracket_tree(groups, strengths, host_teams, params: DixonColesParams,
                          n_sims: int = 10000, completed: dict | None = None,
                          seed: int = 2) -> list[dict]:
    """Return the bracket as rounds → matches → two slots with top-3 teams."""
    rng = np.random.default_rng(seed)
    config = SimulationConfig()
    count = defaultdict(lambda: defaultdict(int))  # (match, 'a'/'b') -> {team: n}

    for _ in range(n_sims):
        standings = {g: play_group(t, strengths, host_teams, params, rng,
                                   completed=completed).standings
                     for g, t in groups.items()}
        winners = {g: standings[g][0]["team"] for g in groups}
        runners = {g: standings[g][1]["team"] for g in groups}
        thirds = [(standings[g][2], g) for g in groups]
        best = sorted(thirds, key=lambda x: (x[0]["points"], x[0]["gd"], x[0]["gf"],
                                             rng.random()), reverse=True)[:8]
        third_team = {g: row["team"] for row, g in best}
        slot_grp = assign_thirds(set(third_team))

        def resolve(slot, match_no):
            if slot[0] == "W":
                return winners[slot[1]]
            if slot[0] == "R":
                return runners[slot[1]]
            return third_team[slot_grp[match_no]]

        winner = {}
        for m, sa, sb in R32_BRACKET:
            ta, tb = resolve(sa, m), resolve(sb, m)
            count[(m, "a")][ta] += 1
            count[(m, "b")][tb] += 1
            winner[m] = ta if resolve_knockout(ta, tb, strengths, host_teams,
                                               params, rng, config) else tb
        for _name, feeders in LATER_ROUNDS:
            for m, (fa, fb) in feeders.items():
                ta, tb = winner[fa], winner[fb]
                count[(m, "a")][ta] += 1
                count[(m, "b")][tb] += 1
                winner[m] = ta if resolve_knockout(ta, tb, strengths, host_teams,
                                                   params, rng, config) else tb

    def top3(match_no, side):
        items = sorted(count[(match_no, side)].items(), key=lambda x: -x[1])[:3]
        return [{"team": t, "p": n / n_sims} for t, n in items if n / n_sims > 0.005]

    order = _leaf_order()
    rounds = []
    round_defs = [("Round of 32", [m for m, _, _ in R32_BRACKET])] + \
                 [(name, list(f)) for name, f in LATER_ROUNDS]
    for name, _ in round_defs:
        matches = [{"match": m, "a": top3(m, "a"), "b": top3(m, "b")}
                   for m in order[name]]
        rounds.append({"round": name, "matches": matches})
    return rounds
