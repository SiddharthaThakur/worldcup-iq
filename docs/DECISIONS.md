# WorldCupIQ Decisions Log

<!-- Append-only. Each entry documents a decision, its rationale, and alternatives considered. -->

## D001: Champion-Challenger Framework over Single Model (2026-06-04)

**Decision:** Run three models (Elo baseline, aggregated player, attention composition) as champion vs challengers rather than picking one "best" model.

**Rationale:** The honest finding is whether player-level modeling adds value. If we only build the player model, we can't measure improvement. The Elo baseline is the null hypothesis. HIGFormer (KDD 2025) showed player-level GNN+transformer works for club football, but no one has tested whether it adds value for World Cup prediction specifically — where sample sizes are tiny and variance is enormous.

**Alternatives considered:**
- Single best model (rejected: can't measure improvement)
- Only Elo baseline (rejected: no technical novelty)
- Only player-level (rejected: no honest evaluation without a baseline)

---

## D002: Dixon-Coles over Plain Poisson (2026-06-04)

**Decision:** Use Dixon-Coles bivariate Poisson for score prediction in all models.

**Rationale:** Dixon-Coles (1997) corrects the well-known undercount of low-scoring draws (0-0, 1-1) in plain bivariate Poisson. This is the standard in football prediction literature and the basis of the best transparent WC prediction model to date (Brier 0.54 on backtest). Using it allows apples-to-apples comparison — the only variable that changes between models is how team strength is derived (Elo vs player composition).

---

## D003: FBRef Basic Stats + Understat xG as Player Data (2026-06-04)

**Decision:** Use FBRef for basic per-90 stats (goals, assists, shots, tackles) and Understat for xG/xA, rather than FIFA video game ratings or Transfermarkt market values.

**Rationale:** FBRef scouting reports are dead (discontinued Jan 2026), but basic stats are still scrapable. Understat provides genuine xG from a neural network model trained on 100K+ shots. FIFA video game ratings are subjective and noisy (used by multiple existing projects — not a differentiator). Market values from Transfermarkt correlate with quality but conflate many factors (age, contract length, marketability).

**Tradeoff:** Understat only covers Big 5 European leagues. ~40% of WC players will lack xG data. Player data_quality flags handle this.

---

## D004: Brier Score as Primary Metric, Not Accuracy (2026-06-04)

**Decision:** Evaluate models primarily on Brier score and RPS, not classification accuracy.

**Rationale:** Accuracy rewards overconfident predictions. A model that says "80% Argentina" for every Argentina match gets the same accuracy as one that says "55% Argentina" when both pick the right winner — but the calibrated model is far more useful. Brier score rewards honest probability estimation. This aligns with epistemic honesty design principle.

---

## D005: Neutral Venue for All WC Matches Except Host Nations (2026-06-04)

**Decision:** Treat all 2026 WC matches as neutral venue in the Elo system, EXCEPT matches involving USA, Canada, or Mexico playing in their own country.

**Rationale:** The 2026 WC is spread across 3 host nations. USA/Canada/Mexico will have genuine home crowd advantage when playing in their own stadiums. Other teams playing in, say, New York do not get home advantage. This is a modeling choice — some approaches give no home advantage at all in World Cups, but empirical evidence shows hosts outperform at ~+0.3 goals.

---

## D006: Entity Resolution via Deterministic + Fuzzy Two-Pass (2026-06-04)

**Decision:** Match players across sources using normalized name + birth year (deterministic), then rapidfuzz token_sort_ratio ≥ 88 (fuzzy), with manual overrides CSV.

**Rationale:** Simple string matching fails on accented names, name order differences (Asian names), and transliterations. Birth year + nationality constraint prevents false positives. The 88% threshold was chosen to minimize false matches while catching common variations (tested on Premier League player lists across FBRef/Understat).

---

## D007: All Dixon-Coles Parameters Must Be Fitted, Never Hand-Set (2026-06-10)

**Decision:** The Elo→expected-goals mapping (intercept, elo_coef, home_adv) is fitted by Poisson regression on post-2010 internationals using strictly PRE-match Elo ratings; rho is fitted by profile likelihood. `predict_match` refuses to run without a fitted params file.

**Rationale:** The original scaffold hardcoded `avg_goals=1.35`, `scale=400`, `rho=-0.13` (borrowed from club football). Since calibration is the project's deliverable, unfitted constants would make the scorecard measure arbitrary choices rather than model quality. Self-identified flaw, fixed.

**Implementation note:** `build_training_rows` captures Elo BEFORE each match updates the system — no outcome leakage into features.

---

## D008: Pre-Registered Hypotheses (2026-06-10 — BEFORE any 2026 match is scored)

Registered before evaluating any 2026 World Cup outcome:

- **H1:** The aggregated player-composition model achieves lower pooled Brier than the Elo baseline (pooled = 2018 WC + 2022 WC backtests + 2026 lock-ins; 95% bootstrap CI on the difference excludes zero).
- **H2 (expected to hold):** Neither model beats de-vigged closing-line probabilities on pooled Brier.
- **H3:** The attention model achieves lower Poisson NLL than the aggregated model on held-out club seasons.
- **Power note:** With per-match Brier SD ≈ 0.2 and plausible effects of 0.01–0.03, 104 matches alone cannot reach significance. The 2026 tournament is out-of-sample evidence, not a standalone hypothesis test. Stated now so the post-mortem can't be accused of moving goalposts.

**Primary metric:** RPS. Secondary: Brier, log-loss, reliability diagrams, CLV.

---

## D009: Kill Criteria for Challengers (2026-06-10)

- **Aggregated player model:** if player-derived team strength correlates < 0.3 with club match outcomes, OR backtest Brier exceeds Elo baseline by > 0.02 — do not deploy to live scorecard; publish the failure analysis instead. Lineup sensitivity ships ONLY if this gate passes (sensitivity from an unvalidated model is noise with false precision).
- **Attention model:** if it fails to beat the aggregated model on held-out club Poisson NLL, OR degrades WC backtest Brier vs Challenger 1 — report the negative result, do not deploy. A "no, interactions don't transfer to international football" finding is publishable.

---

## D010: Bracket Assignment Approximation (2026-06-10)

**Decision:** Round-of-32 pairing uses the correct structural skeleton (8 group winners vs the 8 third-place qualifiers; remaining winners vs runners-up; runners-up pair off) with randomized within-structure assignment, instead of FIFA's exact slot lookup table.

**Rationale:** Exact third-place slotting depends on which group combination qualifies (a published lookup of cases). The approximation preserves champion probabilities to within Monte Carlo noise. Tracked refinement: implement the exact table; effect expected to be visible only in round-specific opponent distributions, not champion odds.

---

## D011: EA FC Ratings as Universal Coverage Floor (2026-06-10)

**Decision:** Player data is tiered: Tier 1 = per-90 club stats + Understat xG (Big-5 leagues, ~55-60% of WC players); Tier 2 = FBRef basic stats only; Tier 3 = EA FC 26 ratings (universal coverage incl. all 48 squads). `data_quality` flags record each player's tier.

**Rationale:** Reverses an earlier dismissal of EA ratings, and the reversal is logged deliberately: EA ratings fail as a *differentiator* (existing projects use them) but succeed as *infrastructure* — the only free source covering Uzbekistan, Jordan, Cape Verde squads etc. Groll et al. found player-quality covariates among the most predictive features. Side question now testable: does the stats-based tier outperform the EA-only tier?

---

## D012: Prediction Lock-In via Git Commits (2026-06-10)

**Decision:** Every live prediction is written to a timestamped JSON (including the SHA-256 of the fitted-params file) and git-committed before kickoff. The lock script REFUSES matches whose kickoff has passed. Matches played before a lock exists are backtest data and are reported separately from predictions, with the boundary date stated explicitly in the write-up.

**Rationale:** "Honest evaluation" must be verifiable, not asserted. A public commit hash converts "trust me" into "check the git log" at near-zero cost.
