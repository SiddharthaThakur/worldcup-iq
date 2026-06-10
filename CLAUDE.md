# WorldCupIQ — Player-Composition World Cup Predictions with Honest Evaluation

## What This Project Does (Plain English)

Every World Cup, hundreds of people build prediction models. They all do the same thing: take a team's historical win rate and FIFA ranking, feed it into a machine learning model, and say "Brazil wins." None of them check whether their model was actually any good afterward.

This project is different in three ways:

1. **It predicts using actual players, not just team names.** Instead of "Brazil vs Germany," it asks "these 11 Brazilians vs these 11 Germans." If Mbappé gets injured, France's prediction changes instantly. If a coach picks a defensive lineup, the model accounts for that.

2. **It runs multiple models and honestly compares them.** A simple Elo baseline (the "champion") runs alongside fancier player-composition models (the "challengers"). If the simple model wins, we say so. No hiding behind complexity.

3. **It tracks its own accuracy in real-time.** Every prediction is compared to bookmaker odds. A running Brier score tells you whether the model is adding value or just making noise. After the tournament, a full post-mortem scorecard gets published — wins, losses, and all.

The tournament runs June 11 – July 19, 2026. 48 teams, 104 matches, 16 stadiums across the US, Canada, and Mexico.

---

## Tech Stack

- **Language:** Python 3.11+
- **Core ML:** scikit-learn, scipy (Dixon-Coles optimization), PyTorch (attention composition)
- **Data:** pandas, statsbombpy, requests, beautifulsoup4
- **Evaluation:** numpy, scipy (calibration tests)
- **Dashboard:** Streamlit, Plotly
- **Entity Resolution:** rapidfuzz (fuzzy name matching)
- **Infrastructure:** pyproject.toml, pytest

---

## Session Start Protocol (`/project:session-start`)

1. Read `docs/PROGRESS.md` — understand what was built last session
2. Read `docs/DECISIONS.md` — understand architectural choices and empirical findings
3. Discover files on disk — `find src/ -name "*.py" | head -30` and check what exists
4. Run tests — `python -m pytest tests/ -x --tb=short` (if tests exist)
5. Check git log — `git log --oneline -10`
6. Identify the current phase from `docs/ROADMAP.md`
7. State: what was done last, what's next, any blockers

## Session End Protocol (`/project:session-end`)

1. Update `docs/PROGRESS.md` — add 2-4 line entry at TOP (newest first) with date
2. Update `docs/DECISIONS.md` — append any new architectural decisions or empirical findings
3. Run tests — `python -m pytest tests/ -x --tb=short`
4. Git commit — `git add -A && git commit -m "<concise summary>"`

## Slash Commands

- `/project:session-start` — Run the session start protocol
- `/project:session-end` — Run the session end protocol
- `/project:roadmap` — Display current phase, completed phases, and next steps
- `/project:implement-phase` — Implement the next phase from the roadmap
- `/project:verify` — Run all tests, check calibration metrics, validate data integrity
- `/project:backtest` — Run backtest on 2018/2022 WC data and report metrics
- `/project:predict <team_a> <team_b>` — Generate match prediction with all models
- `/project:sensitivity <team>` — Show lineup sensitivity analysis for a team
- `/project:scorecard` — Display running calibration scorecard during tournament

---

## Project Structure

```
worldcup-iq/
├── CLAUDE.md                           # This file
├── .claude/skills/                     # Skill files for Claude Code
│   ├── data-pipeline/SKILL.md
│   ├── prediction-models/SKILL.md
│   ├── calibration-evaluation/SKILL.md
│   ├── dashboard/SKILL.md
│   └── entity-resolution/SKILL.md
├── docs/
│   ├── PROGRESS.md                     # Build log (newest first)
│   ├── DECISIONS.md                    # Architectural decisions
│   └── ROADMAP.md                      # Phased implementation plan
├── src/
│   ├── data/                           # Data ingestion and loading
│   ├── features/                       # Feature engineering
│   ├── models/                         # Prediction models
│   ├── evaluation/                     # Calibration and scoring
│   ├── sensitivity/                    # Lineup sensitivity analysis
│   └── dashboard/                      # Streamlit app
├── data/
│   ├── raw/                            # Downloaded datasets (gitignored)
│   ├── processed/                      # Cleaned data (gitignored)
│   └── predictions/                    # Model outputs
├── models/                             # Saved model artifacts (gitignored)
├── tests/                              # pytest test suite
└── notebooks/                          # Exploration notebooks
```

---

## Data Sources

| Source | What | How to Get |
|--------|------|------------|
| Kaggle: international-football-results | 49K+ match results 1872–2025 | `kaggle datasets download martj42/international-football-results-from-1872-to-2017` |
| GitHub: schochastics/football-data | 1.2M club matches + lineups/formations | `git clone` the repo |
| Kaggle: Top 5 League Player Stats | Per-season player stats 2017-2025 | Kaggle download |
| Kaggle: Understat data | xG, xA per player Big 5 leagues | Kaggle download |
| StatsBomb Open Data | Event data for 2018+2022 WC | `pip install statsbombpy` |
| Kaggle: 2026 WC Elo Ratings | Pre-computed Elo for 48 teams | Kaggle download |
| GitHub: worldcup2026 API | Live 2026 WC scores | REST API (free, no key) |
| football-data.co.uk | Historical bookmaker odds | CSV download |

---

## Key Design Principles

1. **Epistemic honesty** — Every prediction comes with calibrated uncertainty. The system flags when data is thin, refuses to over-claim, and tracks its own accuracy.
2. **Champion-challenger** — The simple Elo baseline is the champion. Player-composition models must EARN their place by demonstrating measurable improvement.
3. **Player-level, not team-level** — Teams are compositions of players. Injuries, lineup changes, and tactical selections all affect predictions.
4. **Calibration over accuracy** — Getting probabilities right matters more than the headline "% correct." A model that says "60% ± 5%" and is calibrated is better than one that says "90%" and is wrong.
5. **Market as ground truth** — Bookmaker closing odds are the benchmark. Beating the market is the real test. Not beating it is also a valid finding.

---

## Confidence Levels (Be Honest)

- **High confidence predictions:** Teams with rich historical data + well-known players in Big 5 leagues + stable coaching setups
- **Medium confidence:** Teams with some data gaps (players in non-Big-5 leagues, new coaching staff)
- **Low confidence / flag for refusal:** First-time qualifiers with limited international history AND players in leagues we don't cover. Say "insufficient data" rather than hallucinate a prediction.
