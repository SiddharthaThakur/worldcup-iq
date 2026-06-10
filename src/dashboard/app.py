"""
WorldCupIQ Dashboard — main entry point.

What this does in simple English:
    This is the public-facing Streamlit app that lets anyone see our
    predictions, compare models, explore lineup sensitivities, and
    track our honest calibration scorecard during the World Cup.

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="WorldCupIQ",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ WorldCupIQ")
st.markdown(
    "**Player-composition predictions with honest evaluation** — "
    "FIFA World Cup 2026"
)

st.markdown("---")

st.markdown("""
### What is this?

WorldCupIQ predicts World Cup match outcomes using **actual player data** —
not just team-level statistics like every other predictor.

Three models run side by side:
- 🏆 **Elo Baseline** — Simple team-level ratings (the champion to beat)
- 👥 **Player Composition** — Team strength derived from the actual starting 11
- 🧠 **Attention Model** — Transformer-based player interaction (stretch goal)

Every prediction is compared to **bookmaker odds** and tracked with a
**running calibration scorecard**.

### Pages

Use the sidebar to navigate:
- **Match Preview** — Pre-match predictions for upcoming games
- **Lineup Analyzer** — What happens if a key player is injured?
- **Calibration Board** — Running Brier score and reliability diagrams
- **Live Tracker** — Real-time results during the tournament

### Honest by Design

We track our own accuracy. If simple Elo beats the fancy player model,
we'll say so. If the bookmakers beat us all, we'll report that too.
The evaluation IS the deliverable.
""")

st.markdown("---")
st.caption(
    "Data: StatsBomb Open Data, FBRef, Understat, Kaggle. "
    "Built with epistemic honesty. "
    "[GitHub](https://github.com/your-repo/worldcup-iq)"
)
