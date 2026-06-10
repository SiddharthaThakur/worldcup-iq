"""
Lineup sensitivity analysis: how much does each player matter?

What this does in simple English:
    For a given match (e.g., France vs Morocco), we start with the
    expected starting 11 for each team and compute a win probability.
    Then we remove one player at a time (simulating injury or benching),
    replace them with the next-best player in that position from the
    26-man squad, and re-compute. The difference tells you how much
    that player matters to the prediction.

    "If Mbappé is out, France's win probability drops from 64% to 51%"
    — that's the kind of output this module produces.

    This is genuinely novel for World Cup prediction. No existing project
    does this because they all operate at the team level.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class SensitivityResult:
    """Impact of removing one player from the lineup."""

    player_id: str
    player_name: str
    position: str
    team: str
    replacement_id: str
    replacement_name: str
    base_win_prob: float
    adjusted_win_prob: float
    delta_win_prob: float  # negative = player's absence hurts the team
    data_quality: str  # "full", "partial", "minimal"


def compute_lineup_sensitivity(
    team_code: str,
    opponent_code: str,
    starting_xi: list[str],  # canonical player IDs
    full_squad: list[str],   # all 26 canonical player IDs
    player_stats: pd.DataFrame,
    predict_fn: callable,    # function(team_players, opponent_players) -> win_prob
) -> list[SensitivityResult]:
    """Compute sensitivity for every starting player.

    For each player in the starting 11:
        1. Remove them
        2. Find the best replacement from the bench in the same position
        3. Re-predict
        4. Compute delta

    Args:
        team_code: FIFA code of the team being analyzed
        opponent_code: FIFA code of the opponent
        starting_xi: list of 11 canonical player IDs in the starting lineup
        full_squad: list of all 26 canonical player IDs in the squad
        player_stats: DataFrame with player embeddings/stats, indexed by canonical_id
        predict_fn: callable that takes (team_player_ids, opponent_player_ids) → win probability

    Returns:
        List of SensitivityResult, sorted by delta (most impactful first)
    """
    bench = [p for p in full_squad if p not in starting_xi]
    # TODO: implement opponent starting XI selection (use predicted best 11)
    opponent_xi = starting_xi  # placeholder

    base_prob = predict_fn(starting_xi, opponent_xi)

    results = []
    for player_id in starting_xi:
        player_row = player_stats.loc[player_stats["canonical_id"] == player_id]
        if player_row.empty:
            continue

        player_name = player_row.iloc[0].get("canonical_name", player_id)
        position = player_row.iloc[0].get("position", "UNK")
        quality = player_row.iloc[0].get("data_quality", "minimal")

        # Find best replacement in same position group
        replacement_id = _find_replacement(player_id, position, bench, player_stats)
        if replacement_id is None:
            continue

        repl_row = player_stats.loc[player_stats["canonical_id"] == replacement_id]
        repl_name = repl_row.iloc[0].get("canonical_name", replacement_id) if not repl_row.empty else replacement_id

        # Build modified lineup
        modified_xi = [replacement_id if p == player_id else p for p in starting_xi]
        adjusted_prob = predict_fn(modified_xi, opponent_xi)

        results.append(SensitivityResult(
            player_id=player_id,
            player_name=player_name,
            position=position,
            team=team_code,
            replacement_id=replacement_id,
            replacement_name=repl_name,
            base_win_prob=base_prob,
            adjusted_win_prob=adjusted_prob,
            delta_win_prob=adjusted_prob - base_prob,
            data_quality=quality,
        ))

    results.sort(key=lambda r: r.delta_win_prob)
    return results


def _find_replacement(
    removed_id: str,
    position: str,
    bench: list[str],
    player_stats: pd.DataFrame,
) -> str | None:
    """Find the best bench player in the same position group.

    'Best' = highest overall rating (mean of normalized stats).
    """
    candidates = player_stats[
        (player_stats["canonical_id"].isin(bench))
        & (player_stats["position"] == position)
    ]
    if candidates.empty:
        # Fall back: any bench player (wrong position but someone has to play)
        candidates = player_stats[player_stats["canonical_id"].isin(bench)]
    if candidates.empty:
        return None

    # TODO: use a proper player quality metric here
    # For now, pick the one with most minutes played
    if "minutes_per_90" in candidates.columns:
        best = candidates.sort_values("minutes_per_90", ascending=False).iloc[0]
    else:
        best = candidates.iloc[0]

    return best["canonical_id"]


def format_sensitivity_report(results: list[SensitivityResult]) -> str:
    """Format sensitivity results as a readable report."""
    lines = [f"Lineup Sensitivity for {results[0].team}" if results else "No results"]
    lines.append("=" * 50)
    lines.append(f"{'Player':<25} {'Pos':<5} {'Base':>6} {'w/o':>6} {'Δ':>7}")
    lines.append("-" * 50)
    for r in results:
        delta_str = f"{r.delta_win_prob:+.1%}"
        lines.append(
            f"{r.player_name:<25} {r.position:<5} "
            f"{r.base_win_prob:>5.1%} {r.adjusted_win_prob:>5.1%} {delta_str:>7}"
        )
        lines.append(f"  → replaced by {r.replacement_name} [{r.data_quality}]")
    return "\n".join(lines)
