"""
Round-of-32 bracket projection: who is likely to fill each slot.

What this does in simple English:
    The 2026 Round of 32 has 16 fixed matchups defined by group position
    (e.g. Match 74 = Winner of Group E vs one of the best third-placed
    teams from A/B/C/D/F). We don't yet know WHICH team fills each slot,
    so we simulate the groups thousands of times and, for every slot, list
    the teams most likely to land there.

    As real group games are played, those probabilities concentrate — a
    slot that today shows three candidates will narrow to two, then to the
    one team that actually qualifies, and the matchups become concrete.

    The exact match structure (winners & runners-up) is FIFA's published
    2026 bracket. The third-place SLOTS show the most likely qualifiers
    from the eligible groups; FIFA's exact third-place-to-slot assignment
    depends on a combination table (a documented approximation, D010), so a
    third-place team may appear as a candidate in more than one slot — it
    can of course only end up in one.
"""

from collections import defaultdict

import numpy as np

from src.models.dixon_coles import DixonColesParams
from src.simulation.fifa_annex_c import lookup_assignment, THIRD_SLOTS
from src.simulation.tournament import play_group, resolve_knockout, SimulationConfig

# Each entry: (match_no, slot_a, slot_b). Slots:
#   ("W", "E")  winner of group E
#   ("R", "B")  runner-up of group B
#   ("T", ["A","B","C","D","F","H"])  best third place from one of these groups
R32_BRACKET = [
    (73, ("R", "A"), ("R", "B")),
    (74, ("W", "E"), ("T", ["A", "B", "C", "D", "F"])),
    (75, ("W", "F"), ("R", "C")),
    (76, ("W", "C"), ("R", "F")),
    (77, ("W", "I"), ("T", ["C", "D", "F", "G", "H"])),
    (78, ("R", "E"), ("R", "I")),
    (79, ("W", "A"), ("T", ["C", "E", "F", "H", "I"])),
    (80, ("W", "L"), ("T", ["E", "H", "I", "J", "K"])),
    (81, ("W", "D"), ("T", ["B", "E", "F", "I", "J"])),
    (82, ("W", "G"), ("T", ["A", "E", "H", "I", "J"])),
    (83, ("R", "K"), ("R", "L")),
    (84, ("W", "H"), ("R", "J")),
    (85, ("W", "B"), ("T", ["E", "F", "G", "I", "J"])),
    (86, ("W", "J"), ("R", "H")),
    (87, ("W", "K"), ("T", ["D", "E", "I", "J", "L"])),
    (88, ("R", "D"), ("R", "G")),
]


def simulate_positions(groups, strengths, host_teams, params: DixonColesParams,
                       n_sims: int = 20000, completed: dict | None = None,
                       seed: int = 1) -> dict:
    """Per-team probabilities of finishing 1st, 2nd, and per-slot 3rd.

    Returns {team: {"p1": .., "p2": .., "p3q": .., "p3_slot": {match: p}, "group": g}}.
    p3_slot gives the probability of filling each specific R32 slot.
    """
    rng = np.random.default_rng(seed)
    first = defaultdict(int)
    second = defaultdict(int)
    third_q = defaultdict(int)
    third_slot = defaultdict(lambda: defaultdict(int))
    team_group = {t: g for g, ts in groups.items() for t in ts}

    for _ in range(n_sims):
        thirds = []
        third_by_group = {}
        for g, teams in groups.items():
            gr = play_group(teams, strengths, host_teams, params, rng, completed=completed)
            first[gr.standings[0]["team"]] += 1
            second[gr.standings[1]["team"]] += 1
            thirds.append(gr.standings[2])
            third_by_group[g] = gr.standings[2]["team"]
        best = sorted(thirds, key=lambda r: (r["points"], r["gd"], r["gf"], rng.random()),
                      reverse=True)[:8]
        qual_groups = {team_group[r["team"]] for r in best}
        for r in best:
            third_q[r["team"]] += 1
        assignment = lookup_assignment(qual_groups)
        if assignment:
            for match_no, g in assignment.items():
                third_slot[third_by_group[g]][match_no] += 1

    out = {}
    for t, g in team_group.items():
        slots = {m: c / n_sims for m, c in third_slot[t].items()} if t in third_slot else {}
        out[t] = {"group": g, "p1": first[t] / n_sims, "p2": second[t] / n_sims,
                  "p3q": third_q[t] / n_sims, "p3_slot": slots}
    return out


def _slot_candidates(slot, match_no, positions, groups, top_n=3) -> list[dict]:
    """Top-N teams for one slot, with their probability of filling it."""
    kind = slot[0]
    if kind in ("W", "R"):
        g = slot[1]
        key = "p1" if kind == "W" else "p2"
        teams = groups[g]
        ranked = sorted(teams, key=lambda t: positions[t][key], reverse=True)
        return [{"team": t, "p": positions[t][key]} for t in ranked[:top_n]
                if positions[t][key] > 0.005]
    pool = [t for g in slot[1] for t in groups[g]]
    ranked = sorted(pool, key=lambda t: positions[t]["p3_slot"].get(match_no, 0),
                    reverse=True)
    return [{"team": t, "p": positions[t]["p3_slot"].get(match_no, 0)}
            for t in ranked[:top_n]
            if positions[t]["p3_slot"].get(match_no, 0) > 0.005]


def _slot_label(slot) -> str:
    if slot[0] == "W":
        return f"Winner {slot[1]}"
    if slot[0] == "R":
        return f"Runner-up {slot[1]}"
    return "3rd: " + "/".join(slot[1])


def project_bracket(positions, groups) -> list[dict]:
    """Build the R32 bracket with top-3 candidates per slot."""
    out = []
    for match_no, slot_a, slot_b in R32_BRACKET:
        out.append({
            "match": match_no,
            "a_label": _slot_label(slot_a), "b_label": _slot_label(slot_b),
            "a": _slot_candidates(slot_a, match_no, positions, groups),
            "b": _slot_candidates(slot_b, match_no, positions, groups),
        })
    return out


# How winners feed forward (FIFA's published 2026 bracket). Used to lay out
# the later rounds as placeholders that fill in once their feeder games are
# played — we project only the Round of 32 up front (user's choice).
R16_FEEDERS = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
               93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF_FEEDERS = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF_FEEDERS = {101: (97, 98), 102: (99, 100)}
FINAL_FEEDERS = {104: (101, 102)}


def _round_order():
    """Match order per round so columns read top-to-bottom (feeders aligned)."""
    order = {"Final": [104]}
    order["Semi-finals"] = list(FINAL_FEEDERS[104])
    order["Quarter-finals"] = [m for p in order["Semi-finals"] for m in SF_FEEDERS[p]]
    order["Round of 16"] = [m for p in order["Quarter-finals"] for m in QF_FEEDERS[p]]
    order["Round of 32"] = [m for p in order["Round of 16"] for m in R16_FEEDERS[p]]
    return order


def build_progressive_bracket(positions, groups) -> list[dict]:
    """Round-based bracket: R32 projected now; later rounds are TBD
    placeholders (they fill in as feeder games are decided)."""
    order = _round_order()
    r32_by_match = {m["match"]: m for m in project_bracket(positions, groups)}
    rounds = [{"round": "Round of 32",
               "matches": [r32_by_match[m] for m in order["Round of 32"]]}]
    feeders = {"Round of 16": R16_FEEDERS, "Quarter-finals": QF_FEEDERS,
               "Semi-finals": SF_FEEDERS, "Final": FINAL_FEEDERS}
    for name in ("Round of 16", "Quarter-finals", "Semi-finals", "Final"):
        rounds.append({
            "round": name,
            "matches": [{"match": m, "a": [], "b": [],
                         "from": list(feeders[name][m])} for m in order[name]],
        })
    return rounds


def _knockout_advance_prob(team_a, team_b, strengths, host_teams, params,
                           n=5000, seed=42):
    """P(team_a advances) over a knockout tie, including ET + penalties."""
    config = SimulationConfig()
    rng = np.random.default_rng(seed)
    wins = sum(resolve_knockout(team_a, team_b, strengths, host_teams,
                                params, rng, config) for _ in range(n))
    return round(wins / n, 3)


def _is_confirmed(match):
    """A slot is confirmed when it has exactly one team at >=99.5%."""
    return (len(match.get("a", [])) == 1 and match["a"][0]["p"] >= 0.995 and
            len(match.get("b", [])) == 1 and match["b"][0]["p"] >= 0.995)


def annotate_knockout_probs(bracket, strengths, host_teams, params):
    """Add advancement probabilities to confirmed matchups and propagate
    to later rounds. Modifies the bracket list in place."""
    by_match = {}
    for rnd in bracket:
        for m in rnd["matches"]:
            by_match[m["match"]] = m

    # R32: compute advancement probs for confirmed ties
    r32 = bracket[0]["matches"]
    for m in r32:
        if _is_confirmed(m):
            ta, tb = m["a"][0]["team"], m["b"][0]["team"]
            pa = _knockout_advance_prob(ta, tb, strengths, host_teams, params)
            m["pAdv"] = {ta: pa, tb: round(1 - pa, 3)}

    # Later rounds: populate candidates from feeder matches
    for rnd in bracket[1:]:
        for m in rnd["matches"]:
            if "from" not in m:
                continue
            fa, fb = m["from"]
            ma, mb = by_match.get(fa, {}), by_match.get(fb, {})
            m["a"] = _feeder_candidates(ma)
            m["b"] = _feeder_candidates(mb)


def _feeder_candidates(feeder):
    """Build candidate list for a slot fed by the winner of a prior match."""
    if not feeder:
        return []
    if "pAdv" in feeder:
        return [{"team": t, "p": p} for t, p in
                sorted(feeder["pAdv"].items(), key=lambda x: -x[1])]
    if _is_confirmed(feeder):
        return [{"team": feeder["a"][0]["team"], "p": 0.5},
                {"team": feeder["b"][0]["team"], "p": 0.5}]
    return []
