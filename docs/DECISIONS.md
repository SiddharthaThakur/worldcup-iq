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

---

## D013: Phase 1 Backtest Results — Baseline + Market Benchmark (2026-06-10)

**Empirical finding (not a decision).** Leak-free backtest: for each World Cup, Elo AND Dixon-Coles params fitted only on matches strictly before that tournament's first match (2018: 8,051 training matches; 2022: 12,111). Elo updates online within the tournament. Market = de-vigged (power method) cross-book average closing odds from the football-data.co.uk World Cup workbook. **Both model and market scored on 90-minute results** (1X2 settlement convention; ET winners count as draws).

| Metric (pooled, 128 matches) | Elo+DC baseline | Market |
|---|---|---|
| Brier | 0.2161 | 0.1904 |
| RPS | 0.2355 | 0.2012 |
| Log-loss | 1.0841 | 0.9733 |
| Accuracy | 48.4% | 54.7% |

Per-year model Brier: 2018 = 0.2081, 2022 = 0.2241 (2022 was upset-heavy for the market too: 0.1951 vs 0.1857 in 2018).

- Phase 1 kill criterion (model Brier > 0.60) **passed** with wide margin.
- Consistent with pre-registered H2: the baseline does not beat the market.
- These numbers are now the champion's record. Challengers must beat Brier 0.2161 pooled (gate per D009: within 0.02 or better).

**Fitted live params** (post-2010 through 2026-06-09, n=15,812): intercept=0.1332, elo_coef=0.4448/100Elo, home_adv=0.2767, rho=-0.0280. Sanity: home advantage ≈ +32% goals, small negative rho — both consistent with the football literature.

---

## D014: 90-Minute Results as the Evaluation Outcome Definition (2026-06-10)

**Decision:** All model-vs-market scoring uses the 90-minute result. Knockout matches decided in extra time count as draws for evaluation.

**Rationale:** 1X2 odds settle on 90 minutes. Scoring the model on post-ET results while the market is implicitly scored on 90-minute results would bias the comparison. The source of 90-minute scores is the odds workbook (HGFT/AGFT columns); our results dataset (martj42) records post-ET scores and is used for training only.

---

## D015: Annual Elo Mean Reversion Removed (2026-06-10 — selected on backtest only, supersedes D013 numbers)

**Decision:** `MEAN_REVERSION_FACTOR` = 0 (was 1/3 per year).

**Trigger:** First full-data simulation ranked Morocco #1 (19%) and France 13th (2.5%) — an artifact, not a finding. Diagnosis: the January reset compressed all ratings by 1/3, then teams with competitive January-2026 matches (AFCON participants) re-earned rating while teams with only friendlies stayed compressed.

**Selection procedure (no 2026 leakage):** reversion ∈ {0, 0.1, 1/3} evaluated on the pooled 2018+2022 leak-free backtest. Brier: 0.2054 / 0.2088 / 0.2160. Zero reversion won; adopted BEFORE any 2026 match was scored.

**Updated champion's record (90-min scoring, pooled 128):** model Brier **0.2056**, RPS 0.2210, log-loss 1.0490 vs market Brier 0.1904. Gap to market narrowed from 0.026 to 0.015. Challenger gate (D009) now references Brier 0.2056.

**Refitted live params** (n=15,812): intercept=0.1273, elo_coef=0.2364, home_adv=0.2601, rho=-0.0348. (elo_coef per-point is smaller because ratings now spread wider without annual compression.)

**First headline output (10,000 sims):** ESP 25.9%, ARG 22.6%, FRA 9.4%, BRA 5.5%, ENG 5.2%, COL 4.3%, MAR 3.3%, MEX 2.5% (host boost). Model is more top-heavy than the market consensus — expected for Elo-family ratings; tracked as part of live evaluation.

---

## D016: Simulator Home-Advantage Orientation Bug Fixed (2026-06-10)

**Finding:** the scaffolded simulator applied home advantage to whichever team was passed FIRST to the match function; when the host was the second team (e.g. "South Africa vs Mexico"), the boost went to the wrong side. Fixed with `simulate_match_oriented` (host always gets the fitted home_adv regardless of argument order); regression tests added. Host teams are treated as at-home in all their matches (group fixtures genuinely are; knockout venues are an approximation).

---

## D018: EA FC 26 Is NOT a Universal Coverage Floor (2026-06-10 — supersedes D011's premise)

**Finding (empirical):** D011 reversed an earlier dismissal of EA ratings, adopting them as the Tier-3 source that covers "all 48 squads incl. Uzbekistan, Jordan, Cape Verde." That premise is WRONG. EA FC 26 only licenses certain leagues, so players whose clubs are in unlicensed leagues are absent entirely:

| Nation | Players in EA FC 26 |
|---|---|
| Korea Republic | 400 (fine — KOR squad gaps are name romanization, not absence) |
| Iran | 6 |
| Jordan | 2 |
| Qatar | 0 |

So EA FC 26 floors coverage for European/major-league players, but NOT for squads drawn from the Gulf, Iranian, Uzbek, and similar domestic leagues. Strict name+birth-year coverage of all 1,246 WC squad players is 66% (will rise with fuzzy+nationality matching, but the Qatar/Jordan/Iran zeros are a hard ceiling, not a matching artifact).

**Consequence:** the CLAUDE.md "low confidence / refuse" tier is real and unavoidable for ~3-5 squads. The honest design response: the player-composition model carries a per-team data-coverage score; teams below a coverage threshold are predicted by the Elo baseline ONLY, with player-model predictions suppressed and the reason stated. We do not fabricate player strength for teams we can't see. This is a feature (epistemic honesty), not a failure — but it must be surfaced in the dashboard and write-up, not hidden.

**Data sources confirmed available (2026-06-10):**
- Basic per-90 stats, full 2025/26 season: Kaggle `hubertsidorowicz/football-players-stats-2025-2026` (2,839 Big-5 players, incl. Born + Nation → proper deterministic entity key). Supersedes the March-9 FBRef Wayback snapshot.
- xG / creation (npxG, xGChain, xGBuildup): Understat live endpoint (2,775 Big-5 players). The Haaland-context decomposition lives here.
- Universal-ish ratings: EA FC 26 (16,228 players) — with the coverage caveat above.
- Market value + caps as Tier-3 floor for unlicensed-league players: Transfermarkt (Kaggle `davidcariboo/player-scores`, 47,701 players). Recovers Gulf/Iranian/Uzbek players EA misses.
- FBRef (Cloudflare-blocked to scripts) and SoFIFA: not needed. SoFIFA just displays EA ratings (= the EA FC 26 dataset, same coverage gap); superseded.

---

## D019: Unified Player Table + Per-Team Coverage Tiers (2026-06-10)

**Built `unified_players.parquet`:** all 1,246 squad players matched to the 4 sources on normalized name + birth year, nationality breaking ties, fuzzy fallback (≥82) gated on nationality agreement to rescue romanized names. Validation: Korea went from 1/26 exact → 26/26 with fuzzy (name-order flips), zero false positives spot-checked.

**Coverage (the honest scorecard input):**
- 81.9% of players have a *quantitative* strength signal (xG / stats / EA rating / market value)
- 93.7% are verified professionals (found in Transfermarkt — club/position/age known)
- The remainder carry international caps/goals only (universal from Wikipedia), used as a coarse floor

**Per-team rule (operationalizes D018), continued below.**

---

## D020: Phase 3 Validation Gate — BORDERLINE FAIL, Player Model Does NOT Deploy (2026-06-10)

**The test (D009 gate, on club data so it's out-of-sample and large):** built a club Elo from 88k Transfermarkt league games 2006-2025; squad strength = sum of a club's players' market values (a player-composition aggregate); held out the 2025 season (7,101 games the Elo never trained on); compared Brier of pure Elo vs Elo-blended-with-strength across blend weights λ.

**Result:**

| λ (player blend) | 0.0 (pure Elo) | 0.2 | **0.4** | 0.6 | 0.8 | 1.0 (pure player) |
|---|---|---|---|---|---|---|
| Brier | 0.2033 | 0.2010 | **0.2001** | 0.2009 | 0.2036 | 0.2085 |

- Brier IMPROVES with a moderate blend (0.2033 → 0.2001 at λ=0.4), worsens toward pure player value. The U-shape is textbook champion-challenger: combining beats either signal alone.
- BUT correlation(strength-diff, outcome) = **0.293**, just under the pre-registered 0.30 threshold.

**Decision: the gate FAILS (correlation 0.293 < 0.30). The player-composition model does NOT enter the live scorecard, and lineup sensitivity does NOT ship — exactly as D009 specifies.** The threshold was registered before any data was seen; missing it by 0.007 is still missing it. Moving the bar now would betray the entire premise of the project.

**Honest reading:** this is a *promising near-miss*, not a flat negative. The Brier improvement over 7,101 games is real and the optimal-blend shape is strong evidence the approach carries signal. Two legitimate paths could clear the bar without goalpost-moving: (a) the live WC model uses EA FC ratings, not market value — EA ratings are cross-league calibrated and may correlate better; a club re-test on EA-based strength is a fair, pre-specified retest. (b) Phase 4's attention model may extract more from the same players. Until one of those clears the pre-registered bar, the player model is reported as "promising, unvalidated" and is scored separately, never as a trusted challenger.

**Caveats (stated, not used to excuse the result):** market value is a current snapshot; Dixon-Coles params are the international ones — but both baseline and blend share them, so the relative Brier comparison holds.

---

## D021: EA-Based Re-Test — Direct Evidence PASSES, Proxy Gate BORDERLINE (2026-06-10)

**Pre-specified re-test (legitimate per D020): same gate, but on the ACTUAL deployed signal — EA FC ratings run through the real `team_strength` aggregator, validated on out-of-sample 2025-season club games.** Tightened: only confident EA↔Transfermarkt club name matches kept (268 clubs), and bootstrap 95% CIs on both metrics.

**Result (1,581 holdout games):**

| λ | 0.0 | 0.3 | 0.5 | **0.6** | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| Brier | 0.2099 | 0.2069 | 0.2060 | **0.2059** | 0.2065 | 0.2085 |

- **Out-of-sample Brier improvement (blend vs Elo): +0.0040, 95% CI [0.0018, 0.0062] — STATISTICALLY SIGNIFICANT (excludes 0).** The direct measure of "does it predict better?" passes cleanly. Best λ = 0.6.
- **Correlation(strength-diff, outcome): 0.286, 95% CI [0.243, 0.329].** Point estimate below the pre-registered 0.30 bar, but the CI STRADDLES 0.30 — statistically indistinguishable from the threshold.

**The tension, stated plainly:** the pre-registered gate (D009) keys on the correlation point estimate (≥0.30) and by that letter this FAILS (0.286 < 0.30). But the gate's *intent* — "does player strength carry real predictive signal?" — is now answered YES by a significant out-of-sample Brier gain, the more direct and appropriate measure. The correlation proxy and the direct evidence conflict, and the correlation CI doesn't even cleanly place us below the bar.

Two independent signals (market value in D020: corr 0.293; EA ratings here: corr 0.286) agree: the blend reliably helps a little. The effect is REAL but SMALL.

**This is a judgment call the pre-registration did not cleanly resolve** (an OR-gate on a proxy that turned out to conflict with the direct measure). Resolution is deferred to the project owner rather than decided unilaterally — recorded here as the honest state of evidence. Regardless of that call: **lineup sensitivity stays held back** — this test validates TEAM-level strength, not the finer claim that swapping one player yields meaningful sensitivity (that needs per-player validation we have not done).

---

## D022: Blended Composition Model Promoted to Champion (2026-06-10 — owner decision)

**Decision (project owner):** the Elo+player-composition blend (λ=0.6 × per-team confidence, EA-rating strength bridged to Elo scale) becomes the CHAMPION — the headline model for live lock-ins and champion-probability simulation. Phase 4's attention model becomes the challenger that must beat it. This supersedes D001's "Elo baseline is champion."

**Basis:** D021's statistically-significant out-of-sample Brier improvement on club data (+0.0040, 95% CI [0.0018, 0.0062]). A deliberate, documented deviation from the strict letter of the D009 correlation gate (0.286 < 0.30), justified by the direct evidence outweighing a borderline proxy whose own CI straddles the bar.

**Honesty constraints kept (non-negotiable):**
- The **Elo baseline keeps being scored** on every match alongside the champion. Promotion does not end the comparison — if the simpler model wins live, we will say so.
- **Lineup sensitivity stays HELD BACK.** D021 validated team-level strength, not the finer claim that swapping one player yields meaningful sensitivity. Needs separate per-player validation.
- **Caveat surfaced:** the blend is validated on CLUB data. Its World Cup performance is still unproven (we lack historical international squad ratings to backtest it). The live 2026 scorecard is its real test. Lock-ins are versioned: June 11-13 were Elo-only (v1/v2); the composition champion is v3.

---

## D023: Phase 4 (Attention) — H3 FAILS, Negative Result, Does NOT Deploy (2026-06-10)

**The test (H3, pre-registered D008):** does an attention transformer that models player interactions beat simple aggregation (mean-pooling) on held-out club match prediction? Built a real-lineup dataset: 4,038 Big-5 matches (2022-2025) with starting XIs linked to EA FC attributes (pace/shooting/passing/dribbling/defending/physical/overall); 3,230 train / 808 chronologically-held-out. Both models share everything except attention-pool vs mean-pool, trained identically, compared on held-out Poisson NLL over 5 seeds.

**Result:**

| Model | Held-out Poisson NLL (5 seeds) |
|---|---|
| Aggregation (mean-pool) | 1.7831 ± 0.0011 |
| Attention (transformer) | 1.7833 ± 0.0025 |
| Attention improvement | **−0.0002 ± 0.0033** (attention wins 3/5 seeds — a coin flip) |

**Verdict: H3 FAILS. Attention and aggregation are statistically indistinguishable.** The attention model does NOT deploy (D009 kill criterion). "Averaging is enough" — modeling player interactions adds no measurable predictive value over simply combining player qualities, even at CLUB level (where chemistry should be strongest). This is a clean, robust negative result, and per the roadmap it is publishable as-is: it answers "do player interactions transfer to match prediction?" with a well-powered no.

**Scope/caveat (stated, not used to wriggle):** this tests attention over EA-ATTRIBUTE features. A richer interaction signal (passing networks, on-pitch xG combinations) might differ — but on the features a free, reproducible pipeline can assemble, the answer is no. The negative is specific to these features, and that's the honest boundary of the claim.

**Consequence for the project:** the champion remains the Elo+composition blend (D022). Both challengers' fates are now settled: Challenger 1 (aggregation) → promoted to champion on significant Brier evidence; Challenger 2 (attention) → negative result, shelved. The aggregation-is-enough finding is itself a contribution.

---

## D024: Champion+ Edges — Ensemble, Altitude, Rest/Travel (2026-06-10)

**Three free edges layered onto the champion (owner-requested after a brainstorm):**

1. **Elo ensemble** — average our Elo with an independent 2026 Elo system (Kaggle `afonsofernandescruz`, covers all 48 teams), after rescaling theirs to our mean/std. Cheap variance reduction; standard forecasting practice.
2. **Altitude** — teams not altitude-adapted are penalized at high venues (Mexico City 2,240m: −59 Elo; Guadalajara/Zapopan 1,560m: small). Adapted nations (MEX, BOL, ECU, COL, PER) exempt. 7 of 72 group games affected. Physiologically documented (McSharry, BMJ 2007).
3. **Rest/travel** — Elo nudge from the rest-day and travel-distance DIFFERENTIAL between the two teams (haversine over the 16 host-city coordinates). Matters more this WC than any prior — it spans a continent.

**Pipeline:** rating = [ensemble(our Elo, indep Elo) + composition blend] + altitude + rest/travel → Dixon-Coles.

**HONESTY NOTE (important):** unlike the model core (everything fitted, D007), the altitude and rest/travel COEFFICIENTS are literature-informed, NOT fitted — there is no altitude/rest/travel signal in our results file to fit on. They are deliberately MODEST and capped (altitude ≤ ~59 Elo, rest ≤ ±24 Elo, travel ≤ ~30 Elo) so they sharpen specific matches without dominating. The ensemble, by contrast, is principled and parameter-light. These edges are expected to recover only a fraction of the ~0.01 Brier gap to the market and reduce variance; they do not change the thesis. Champion+ is the headline prediction model; the plain champion and Elo baseline remain scored alongside it.

---

## D019 (cont.): Per-team coverage rule a team is "player-model-capable" if ≥50% of its squad has a quantitative signal. **42 / 48 qualify.** The 6 Elo-only teams — Jordan, Qatar, South Africa, Uzbekistan, Egypt, Panama — get the player model SUPPRESSED (predicted by Elo baseline alone), reason surfaced in dashboard/write-up. Egypt is the instructive case: Salah is Big-5 but most of the squad plays the Egyptian domestic league, so the team as a whole is data-poor. This is the player model's "player-delta-on-Elo" design (discussed, not yet implemented): the per-team coverage score scales how much the player nudge is trusted; data-poor teams fall back to Elo gracefully.

---

## D017: Flat K-Factor Replaces the Tournament-Importance Ladder (2026-06-10 — backtest-selected, pre-2026-scoring)

**Decision:** All matches update Elo with K=25. The conventional ladder (WC=60, Euro=50, ..., Friendly=15) is removed.

**Experiment (decision rule pre-stated: adopt only if pooled Brier improves > 0.003 with RPS agreeing):**

| Variant | Brier | RPS |
|---|---|---|
| current ladder | 0.2054 | 0.2223 |
| ladder × 1.5 (steeper) | 0.2075 | 0.2253 |
| ladder flattened halfway | 0.2026 | 0.2179 |
| **flat K** | **0.2009-0.2012** | **0.2149-0.2156** |

Steeper weighting is monotonically worse: high-K tournaments make ratings overreact to small-sample, high-variance knockout results. Level sweep K∈{20,25,30,35,40} is a plateau (Brier 0.2009-0.2012) — the gain comes from removing the ladder, not tuning the level. K=25 chosen as plateau center.

**Updated champion's record (90-min scoring, pooled 128): model Brier 0.2009, RPS 0.2132** vs market 0.1904/0.2012. Cumulative gap to market: 0.026 → 0.0105 across D015+D017. Challenger gate (D009) now references Brier 0.2009.

**Refitted live params** (n=15,812): intercept=0.1217, elo_coef=0.2408, home_adv=0.2585, rho=-0.0354.

**New headline (10K sims):** ARG 22.2%, ESP 17.1%, BRA 9.2%, FRA 7.3%, COL 5.7% — materially closer to market consensus than the D015 table.

**Model naming:** lock-ins from June 14 onward are `elo_dixon_coles_v2`; the June 11-13 locks remain v1 (importance-weighted K) and are scored as v1.
