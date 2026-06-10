# ⚽ WorldCupIQ

**Player-composition World Cup predictions with honest evaluation.**

Most World Cup prediction models do the same thing: take a team's Elo rating, feed it into XGBoost, and say "Brazil wins." None of them check whether their model was actually any good afterward.

WorldCupIQ is different:

1. **Predicts using actual players, not just team names.** If Mbappé is injured, France's prediction changes instantly.
2. **Runs multiple models and honestly compares them.** A simple Elo baseline runs alongside fancier player-composition models. If the simple model wins, we say so.
3. **Tracks its own accuracy in real-time.** Every prediction is compared to bookmaker odds with a running Brier score scorecard.

## The Approach

### Champion-Challenger Framework

| Model | What it does | Role |
|-------|-------------|------|
| **Elo + Dixon-Coles** | Team-level ratings → bivariate Poisson score prediction | Champion (baseline to beat) |
| **Aggregated Player** | Average player stats by position → team strength → Dixon-Coles | Challenger 1 |
| **Attention Composition** | Transformer over 11 player embeddings → team representation | Challenger 2 (stretch) |

All three produce full probability distributions (not just "Team A wins"), tracked against bookmaker-implied probabilities.

### Lineup Sensitivity (Novel)

For every match: "With the expected starting XI, France wins at 64%. Without Mbappé, it drops to 51%." No existing WC predictor does this.

### Honest Evaluation

- **Brier score** — Are the probabilities calibrated?
- **Ranked Probability Score** — Better metric for ordinal outcomes
- **Closing Line Value** — Does the model beat the bookmakers?
- **Reliability diagrams** — Visual calibration check
- **Post-tournament scorecard** — Full public post-mortem

## Data Sources (All Free)

- International match results: 49K+ matches ([Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017))
- Club match lineups: [schochastics/football-data](https://github.com/schochastics/football-data)
- Player stats: FBRef basic stats + [Understat](https://understat.com/) xG
- World Cup events: [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- Live 2026 scores: [worldcup2026 API](https://github.com/rezarahiminia/worldcup2026)
- Bookmaker odds: [football-data.co.uk](https://www.football-data.co.uk/)

## Quick Start

```bash
git clone https://github.com/your-repo/worldcup-iq.git
cd worldcup-iq
pip install -e ".[dev]"

# Run tests
pytest

# Run the dashboard
streamlit run src/dashboard/app.py
```

## Project Status

See [docs/ROADMAP.md](docs/ROADMAP.md) for the phased implementation plan.

The 2026 FIFA World Cup runs **June 11 – July 19**. 48 teams, 104 matches.

## License

MIT

## Acknowledgments

- [StatsBomb](https://statsbomb.com/) for open event data
- [Sports Reference / FBRef](https://fbref.com/) for football statistics
- The football analytics community for setting high standards
