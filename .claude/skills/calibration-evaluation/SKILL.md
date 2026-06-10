---
name: calibration-evaluation
description: Framework for honestly evaluating prediction quality — Brier score, log-loss, RPS, reliability diagrams, market comparison, and Closing Line Value tracking. The evaluation IS the deliverable.
---

# Calibration & Evaluation Skill

## What This Skill Covers

The honest evaluation framework that makes this project different from every other WC predictor. Most projects say "our model predicts Argentina wins!" This project says "here's our calibration curve, Brier score, and how we compare to bookmakers over 104 matches."

## Core Metrics

### Brier Score
Mean squared error of probability forecasts. For win/draw/loss (3 outcomes):
```
BS = (1/N) * Σ (1/3) * Σ_k (p_k - o_k)²
```
where p_k is predicted probability for outcome k, o_k is 1 if outcome k occurred else 0.
- Perfect = 0, Uniform random (1/3, 1/3, 1/3) = 0.667
- A Brier score of 0.54 (like the Dixon-Coles baseline in the literature) indicates real signal

### Log-Loss (Cross-Entropy)
More punishing than Brier for confident wrong predictions:
```
LL = -(1/N) * Σ log(p_actual_outcome)
```
- Perfect = 0, Uniform = log(3) ≈ 1.099
- Clip probabilities at [0.01, 0.99] to avoid infinity

### Ranked Probability Score (RPS)
Better than Brier for ordinal outcomes (it penalizes predicting a win when a draw happened less than predicting a win when a loss happened):
```
RPS = (1/2) * Σ_k (CDF_predicted_k - CDF_actual_k)²
```
- Preferred metric in football prediction literature

### Accuracy
Simply: did the highest-probability outcome occur? Less informative than the above but intuitive.
- Baseline: always picking the favorite ≈ 45-50% for international football
- Random = 33%

## Market Comparison

### Removing the Overround
Bookmaker odds include a margin (the "vig" or "overround"). To get implied probabilities:
1. Convert decimal odds to probabilities: `p = 1/odds`
2. Sum all three: `total = p_home + p_draw + p_away` (typically 1.05-1.10)
3. Normalize: `p_adjusted = p / total`

### Closing Line Value (CLV)
The gold standard for whether a prediction model has genuine edge:
- Compare your pre-match prediction to the closing bookmaker line
- If your model consistently predicts higher probabilities for outcomes that the market later moves toward, that's CLV
- Track: for each match, did the closing line move toward or away from your prediction?

### Head-to-Head Scoring
For each match, compute Brier score for:
1. Your Elo baseline model
2. Your player-composition model
3. Bookmaker-implied probabilities

Running cumulative comparison. Chart it. Be honest about who's winning.

## Reliability Diagrams

Bin predictions by confidence level (e.g., 0-10%, 10-20%, ..., 90-100%). For each bin:
- x-axis: average predicted probability
- y-axis: actual frequency of that outcome

Perfect calibration = diagonal line. Deviations show overconfidence or underconfidence.

Generate separate reliability diagrams for:
- Home wins
- Draws (hardest to calibrate)
- Away wins

## Backtest Protocol

### 2022 World Cup Backtest
- Use data available before November 2022 to generate predictions
- Compare against actual 2022 WC results (64 matches)
- This is the primary validation

### 2018 World Cup Backtest
- Use data available before June 2018
- Compare against actual 2018 WC results (64 matches)
- Secondary validation

### Rules
- **Strict temporal ordering** — Never use future data. If backtesting 2022, use only data available before the tournament.
- **Report all metrics** — Don't cherry-pick the metric that makes the model look best.
- **Compare to bookmakers** — Get historical odds from football-data.co.uk for 2018/2022 WC matches.

## Tournament Scorecard (Live Dashboard)

During the 2026 World Cup, maintain a running scorecard:

```
Match Day 3 Scorecard (June 14, 2026)
──────────────────────────────────────
Matches predicted: 6
Elo Baseline Brier:        0.52
Player Model Brier:         0.49
Bookmaker Brier:            0.48
──────────────────────────────────────
Biggest surprise: Saudi Arabia 2-1 Spain (our model: 8% SA win)
Best call: Argentina 3-0 Morocco (our model: 62% ARG, market: 55%)
──────────────────────────────────────
```

Update after every match. Publish daily during the group stage.

## Post-Tournament Analysis

After July 19 (final), produce:
1. Full Brier/RPS/log-loss comparison table (3 models + bookmakers)
2. Reliability diagrams per model
3. Calibration statistical test (Hosmer-Lemeshow or similar)
4. CLV analysis: did the player model identify market inefficiencies?
5. Lineup sensitivity validation: did injury impacts match predictions?
6. Honest narrative: what worked, what failed, and why

This post-mortem is the LinkedIn post / blog post deliverable.
