---
name: tournament-simulation
description: Monte Carlo simulation of the 2026 World Cup's new 48-team format — group stage with third-place advancement rules, knockout resolution with extra time and penalties, and champion probability estimation.
---

# Tournament Simulation Skill

## What This Skill Covers

Simulating the full 2026 World Cup thousands of times to produce champion probabilities and round-by-round advancement odds. This is the demo centerpiece — "who wins the World Cup?" is the question everyone asks.

## The 2026 Format (NEW — old 32-team code does not work)

- **48 teams, 12 groups of 4** (groups A–L), 72 group matches
- **Top 2 per group advance** (24 teams)
- **8 best third-place teams also advance** → 32 teams
  - Third-place ranking: points → goal difference → goals scored → (officially: fair play, drawing of lots; we use random tiebreak)
- **Round of 32 → R16 → QF → SF → Final** (104 matches total)
- Hosts USA, Canada, Mexico get home advantage in their own stadiums only

## Knockout Resolution Model

Dixon-Coles gives 90-minute W/D/L. Draws must be resolved:

1. **Extra time**: modeled as a mini-match — same strength ratio, expected goals scaled by ~0.33 (ET is 1/3 the length of regulation; literature supports roughly proportional scoring rates)
2. **Penalties**: near coin-flip with a small capped edge for the stronger team (edge = 3% per 100 Elo, capped at ±10%)

WHY THIS MATTERS: naively splitting draws 50/50 systematically overrates underdogs in knockouts. The decomposition above keeps the favorite's edge alive through ET while acknowledging penalties are mostly luck.

## Known Limitations (documented, not hidden)

1. **Bracket assignment is structural, not exact.** FIFA's published bracket maps specific third-place group combinations to specific slots via a lookup table. Current implementation preserves the structure (winners face thirds, same-group rematches avoided) but randomizes assignment. Effect on champion probabilities: within simulation noise. Exact bracket is a tracked refinement (DECISIONS.md D010).
2. **Group tiebreakers skip head-to-head and fair play.** Points → GD → GF → random. Misorders rare three-way ties.
3. **Static strengths within a simulation.** No within-tournament form updates, no fatigue, no injuries mid-sim. Lineup changes enter via the player model's strength inputs between lock-ins, not inside the Monte Carlo.

## Outputs

`run_simulation()` returns per-team: P(reach R32), P(R16), P(QF), P(SF), P(final), P(champion). 10K simulations runs in well under a minute. Seeded RNG for reproducibility — the same seed and inputs must give the same probabilities (this is tested).

## Usage Pattern

```python
from src.simulation.tournament import run_simulation, SimulationConfig
from src.models.dixon_coles import DixonColesParams

probs = run_simulation(
    groups={"A": ["MEX", "...", "...", "..."], ...},   # 12 groups of 4
    strengths={"MEX": 1712.0, ...},                     # Elo or player-derived
    host_teams={"USA", "CAN", "MEX"},
    params=DixonColesParams.load(),                     # FITTED params only
    config=SimulationConfig(n_sims=10_000, seed=42),
)
```

Run it once with Elo strengths and once with player-derived strengths — diverging champion probabilities between the two models is itself dashboard content.
