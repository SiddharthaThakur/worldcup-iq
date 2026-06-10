"""
Run the 2026 World Cup Monte Carlo.

Champion model (D022): Elo blended with player composition. Also runs the
Elo baseline for comparison (the comparison never stops).

Run: python -m src.simulation.run_2026
Output: data/predictions/champion_probabilities.csv (+ console table)
"""

from pathlib import Path

from src.data.results_loader import load_processed_results
from src.data.wc2026 import load_wc2026
from src.models.champion_model import champion_ratings_for_wc
from src.models.dixon_coles import DixonColesParams
from src.models.elo import EloSystem
from src.simulation.tournament import SimulationConfig, run_simulation

OUT = Path("data/predictions/champion_probabilities.csv")


def main(n_sims: int = 10_000, model: str = "champion"):
    """model='champion' (blend, D022) or 'baseline' (pure Elo)."""
    results = load_processed_results()
    wc = load_wc2026(save=False)
    params = DixonColesParams.load()
    elo = EloSystem().fit_from_results(results)

    if model == "champion":
        strengths = champion_ratings_for_wc(elo)
    else:
        strengths = {t: elo.get_rating(t) for ts in wc.groups.values() for t in ts}

    df = run_simulation(
        groups=wc.groups,
        strengths=strengths,
        host_teams={"USA", "CAN", "MEX"},
        params=params,
        config=SimulationConfig(n_sims=n_sims),
    )
    df["model"] = model
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df, strengths


if __name__ == "__main__":
    df, strengths = main()
    print("Top 15 by champion probability (10,000 sims):")
    top = df.head(15).copy()
    top["elo"] = top["team"].map(strengths).round(0)
    print(top.to_string(index=False,
                        float_format=lambda v: f"{v:.3f}" if v < 1 else f"{v:.0f}"))
