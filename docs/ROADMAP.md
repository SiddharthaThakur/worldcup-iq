# WorldCupIQ Roadmap (v2 — descoped and re-sequenced after critique)

## What This Project Does (Plain English)

A research question with a demo attached:

**"Does player-level composition modeling add measurable predictive value over
team-level ratings in international football — and can either beat the market?"**

We predict World Cup matches three ways (simple Elo baseline, player-stat
composition, attention over player interactions), compare all three against
bookmaker odds with proper calibration metrics, simulate the whole tournament
to get champion probabilities, and publish a verifiably honest scorecard —
every live prediction is git-committed before kickoff.

The answer is interesting whether it's yes or no. The honesty is the product.

Phases are sequential with no calendar dates. The 2026 tournament (June 11 –
July 19) is an evaluation set, not a deadline: models may enter the live
scorecard mid-tournament with their entry date marked; matches played before
a model's first lock-in are backtest data for that model, never predictions.

---

## Phase 1: Foundation
**Status: COMPLETE (2026-06-10) — see D013 for backtest numbers**

- [ ] Load international results (Kaggle martj42), team alias resolution
- [ ] Load 2026 fixtures, groups, 26-man squads
- [ ] Fit Elo system; FIT Dixon-Coles params via `src/models/fit_params.py`
      (Poisson regression + rho MLE — see D007; predict refuses unfitted params)
- [ ] Backtest fitted baseline on 2018 + 2022 WC (128 matches); record
      Brier/RPS/log-loss in DECISIONS.md
- [ ] Load 2018/2022 bookmaker odds (football-data.co.uk), de-vig, score the
      market on the same backtests
- [ ] Tests green throughout

**Kill criterion:** baseline backtest Brier worse than 0.60 → something is
broken; stop and debug before anything else.

**Deliverable:** fitted, backtested baseline + market benchmark numbers.

---

## Phase 2: Simulator + Market Layer + Lock-In
**Status: SCAFFOLDED (code written, untested against real data)**

- [ ] Validate tournament simulator: 12 groups, third-place advancement,
      knockout ET/penalty resolution (structural bracket per D010)
- [ ] Champion probabilities from Elo strengths — first headline output
- [ ] Odds ingestion for live matches (manual CSV path is the default;
      The Odds API free tier optional)
- [ ] Lock-in flow end-to-end: predict → JSON → git commit → verify_lock
- [ ] Begin locking real predictions for remaining 2026 matches

**Deliverable:** champion probability table + verifiable prediction trail.

---

## Phase 3: Player Composition Model (Challenger 1)
**Status: NOT STARTED**

- [ ] Tiered player data per D011 (club stats + xG → FBRef basic → EA FC floor)
- [ ] Entity resolution for the ~1,200 WC squad players (the long pole)
- [ ] Position-group aggregation → team strength → same Dixon-Coles likelihood
- [ ] Backtest gate per D009 (correlation ≥ 0.3, Brier within 0.02 of baseline)
- [ ] If gate passes: join live scorecard (entry date marked) + lineup
      sensitivity goes live. If it fails: publish the failure analysis.

**Deliverable:** validated challenger or documented negative result.

---

## Phase 4: Attention Composition (Challenger 2)
**Status: MODEL CODE SCAFFOLDED**

- [ ] Entity resolution at club scale (schochastics lineups ↔ player stats)
- [ ] Train TeamCompositionTransformer end-to-end with Poisson NLL
- [ ] Ablation: attention vs aggregation on held-out club seasons (H3)
- [ ] Kill criteria per D009; deploy only if earned

**Deliverable:** answer to "do player interactions transfer to international
football?" — publishable either way.

---

## Phase 5: Live Operation + Dashboard
**Status: SHELL ONLY**

- [ ] Streamlit pages: match preview (all models + market), lineup analyzer
      (gated), calibration board, champion-probability tracker
- [ ] Daily: lock predictions, enter closing odds, score completed matches
- [ ] Running scorecard: cumulative Brier/RPS, all models vs market

---

## Phase 6: The Write-Up
**Status: NOT STARTED**

- [ ] Pooled evaluation (2018 + 2022 + locked 2026) with bootstrap CIs — H1/H2/H3 verdicts
- [ ] CLV analysis, reliability diagrams, lock-in verification report
- [ ] Honest narrative: what beat what, what failed, where the data boundary
      between backtest and prediction sits
- [ ] Open-source release

**This is the actual deliverable.** Strong regardless of which hypothesis wins.

---

## Explicitly Cut

- RAG/tactical-retrieval layer (different project)
- Video/head-pose anything
- Exact FIFA bracket lookup table (tracked refinement, D010 — only if time allows)
