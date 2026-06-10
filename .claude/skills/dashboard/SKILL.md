---
name: dashboard
description: Streamlit dashboard for WorldCupIQ — match previews, live predictions, lineup sensitivity charts, and a running calibration scorecard. The visual output that makes this project demo-worthy.
---

# Dashboard Skill

## What This Skill Covers

A Streamlit multi-page app that serves as the public face of WorldCupIQ. Four pages covering pre-match, live, post-match, and meta-evaluation views.

## Pages

### 1. Match Preview (`pages/match_preview.py`)
Before each match, show:
- **Three-model comparison**: Elo baseline, aggregated player model, attention model (if ready) — side by side probability bars for win/draw/loss
- **Bookmaker line**: Current market odds converted to probabilities, overlaid
- **Disagreement flag**: Highlight where the player model diverges >5% from market
- **Key matchup insights**: Which position group drives the advantage
- **Lineup sensitivity chart**: Bar chart showing top 5 players whose absence changes win probability most

### 2. Live Tracker (`pages/live_tracker.py`)
During the tournament:
- **Today's matches**: Predictions for upcoming games
- **Recent results**: Actual outcomes vs predictions (color-coded: green=correct favorite, red=surprise)
- **Running scorecard**: Cumulative Brier score chart (line chart, all models + bookmakers)
- **Group standings**: Current standings with projected final positions

### 3. Lineup Analyzer (`pages/lineup_analyzer.py`)
Interactive tool:
- Select a team → see current 26-man squad
- Toggle players in/out of the starting 11
- See how team strength score and match predictions change
- Player data quality badges (full / partial / minimal)
- "What if" scenarios: "What if France starts Camavinga instead of Tchouaméni?"

### 4. Calibration Board (`pages/calibration_board.py`)
The honest scorecard:
- **Reliability diagram**: Interactive Plotly chart, selectable by model and outcome type
- **Brier score table**: All models + bookmakers, cumulative and rolling 10-match window
- **Surprise log**: Matches where the model was most wrong, with analysis
- **Model comparison**: Which model had the best Brier score after group stage, knockouts, overall?

## Design Direction

Sports analytics meets data journalism. Think FiveThirtyEight's election forecast aesthetic:
- Clean, data-forward, lots of white space
- Probability distributions visualized as density ridgeline plots or horizontal stacked bars
- Color coding: consistent palette across models (e.g., blue=Elo, orange=player model, green=attention, gray=bookmaker)
- Calibration charts with confidence bands
- Mobile-friendly (Streamlit handles this)

## Tech Notes

- Use `streamlit>=1.30` with native multipage support (`pages/` directory)
- Plotly for interactive charts (reliability diagrams, sensitivity bars)
- Cache data loads with `@st.cache_data`
- Store predictions in `data/predictions/` as timestamped JSON
- Auto-refresh during live matches (30-second poll interval)
- Sidebar: model selector, match day navigator, confidence threshold filter

## Data Flow

```
data/processed/*.parquet → src/models/ → data/predictions/*.json → dashboard reads predictions
                                                                     ↓
                                                              live API → real-time results
                                                                     ↓
                                                              evaluation metrics computed on-the-fly
```

The dashboard never trains models. It reads pre-computed predictions and compares to outcomes.
