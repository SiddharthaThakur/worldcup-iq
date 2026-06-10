---
name: prediction-models
description: Architecture and implementation guidance for the champion-challenger prediction model framework — Elo+Dixon-Coles baseline, aggregated player model, and attention-based composition model for World Cup match prediction.
---

# Prediction Models Skill

## What This Skill Covers

Three models predicting World Cup match outcomes, structured as champion vs challengers. The point is NOT to build the fanciest model — it's to honestly measure whether player-level composition adds value over simple team-level Elo.

## Model 1: Champion — Elo + Dixon-Coles (Baseline)

### Elo Rating System
- Initialize from Kaggle pre-computed Elo or compute from scratch using international results
- K-factor: 40 for World Cup, 30 for continental tournaments, 20 for qualifiers, 10 for friendlies
- Home advantage: +100 Elo for home team (0 for neutral venues — all WC 2026 matches are neutral except USA/Canada/Mexico games)
- Mean reversion: regress 1/3 toward 1500 at the start of each calendar year

### Dixon-Coles Model
The standard for football score prediction. Bivariate Poisson with a correction for low-scoring draws.

- Two teams with attack/defense parameters: `α_i` (attack), `β_i` (defense)
- Expected goals: `λ_home = α_home * β_away * γ` where `γ` is home advantage
- Dixon-Coles correction `ρ` adjusts P(0-0), P(1-0), P(0-1), P(1-1)
- Fit via maximum likelihood on recent international results (weighted by recency)
- For the Elo variant: derive `λ` directly from Elo difference rather than fitting per-team parameters

### Output
Full probability distribution over scorelines (0-0 through 5-5), aggregated to:
- P(home_win), P(draw), P(away_win)
- Expected goals for each team
- Most likely scoreline

## Model 2: Challenger 1 — Aggregated Player Composition

### Concept
Replace Elo with a team strength score derived from the actual starting 11's club season stats.

### Player Embedding
Per-player feature vector from club stats (all per-90):
- Offensive: goals, assists, xG, xA, shots, key_passes
- Creative: progressive_passes, progressive_carries, xGChain, xGBuildup
- Defensive: tackles, interceptions, blocks, clearances, pressures
- Physical: minutes_played (as a proxy for fitness/importance)

Missing features filled with position-group median. Features standardized (z-score) within each position group (GK, DEF, MID, FWD).

### Team Composition
For a given 11-player lineup:
1. Group players by position (GK, DEF, MID, FWD)
2. Average embeddings within each position group
3. Concatenate the 4 position-group averages → team vector (4 × feature_dim)
4. Team strength = learned linear projection of team vector → scalar

### Match Prediction
Feed team strengths into Dixon-Coles in place of Elo ratings. Same bivariate Poisson framework, different strength inputs.

### Training
- Train the linear projection on club matches with known lineups (schochastics data)
- Validate on held-out international matches
- The strength score should correlate with match outcomes

## Model 3: Challenger 2 — Attention-Based Composition (Stretch Goal)

### Concept
Instead of averaging player embeddings by position, use a small transformer to learn how players interact. A defensive midfielder's value depends on who the center-backs are.

### Architecture
```
Input: 11 player embeddings (each dim=D) + positional encoding (formation position)
       ↓
Multi-head self-attention (2 layers, 4 heads, dim=64)
       ↓
[CLS] token pooling → team representation (dim=64)
       ↓
Match head: concat two team representations → MLP → (expected_goals_home, expected_goals_away)
```

### Training
- Train on club match lineups + outcomes (schochastics data, ~100K+ matches)
- Loss: Poisson negative log-likelihood on goals
- Validation: held-out seasons

### Why This Might Not Work
- International teams train together rarely. Club chemistry ≠ national team chemistry
- 11-player attention with 100K training examples may overfit
- The signal-to-noise ratio in football is inherently low

If it doesn't outperform the simple aggregated model, that's a finding, not a failure. Document it.

## Champion-Challenger Evaluation Protocol

For every match, all three models produce predictions. Track:
1. **Brier score** per model (lower = better)
2. **Log-loss** per model
3. **Ranked Probability Score (RPS)** per model
4. **Accuracy** (did the highest-probability outcome occur?)
5. **Calibration** (are 60% predictions right 60% of the time?)

Compare all models against **bookmaker-implied probabilities** (after removing the overround/vig).

A model earns "challenger → champion" promotion only if it shows statistically significant improvement over the Elo baseline across 50+ matches. For 104 WC matches, this is a high bar.

## Key Implementation Notes

- All models must produce full probability distributions, not just point predictions
- Confidence intervals via bootstrap (1000 resamples of the training data)
- For lineup sensitivity: re-run the player models with one player swapped out, report Δ in win probability
- Cache model predictions in `data/predictions/` with timestamps
- Never use future data when generating predictions (strict temporal ordering)
