# WorldCupIQ Progress Log

<!-- Newest entries at the top. 2-4 lines per entry. -->

## 2026-06-10 — Unified Player Table: 1,246 players × 4 sources (D019)
- Matched every squad player to Kaggle stats / Understat / EA FC 26 / Transfermarkt on name+birthyear+nationality, fuzzy fallback for romanization (Korea 1→26/26 recovered)
- Coverage: 81.9% have quantitative strength signal, 93.7% verified pro; 42/48 teams player-model-capable
- 6 Elo-only teams flagged: JOR/QAT/RSA/UZB/EGY/PAN (Egypt = Salah aside, squad is domestic-league)
- Transfermarkt recovers the Gulf/Iran/Uzbek players EA misses; SoFIFA confirmed moot (= EA data)
- 103 tests green. Next: Phase 3 player-composition model (player-delta-on-Elo design)

## 2026-06-10 — Player Data Secured: Kaggle + Understat + EA FC 26
- Kaggle via gitignored .env from 1Password (user convention, not kaggle.json); pulled 3 datasets
- Confirmed: Kaggle hubertsidorowicz stats are FULL 2025/26 season (June 1, fresher than my March Wayback FBRef) + Born/Nation → proper deterministic entity key; Understat = live xG/creation; EA FC 26 = 16,228 players
- Honest finding (D018): EA FC 26 is NOT universal — Qatar 0, Jordan 2, Iran 6 players (unlicensed leagues). ~3-5 squads will be Elo-only with player model suppressed + reason stated
- 95 tests green. Next: proper name+birthyear+nationality entity resolution across all sources, then Phase 3 model

## 2026-06-10 — K-Factor Experiment: Flat K=25 Beats the Importance Ladder (D017)
- User-prompted question ("are tournaments weighted more?") led to backtest experiment: flat K beats WC=60 ladder, Brier 0.2054 → 0.2009; steeper = monotonically worse
- Gap to market now 0.0105 (was 0.026 at first fit); new champion table: ARG 22%, ESP 17%, BRA 9% — closer to market consensus
- Lock-ins from June 14 are elo_dixon_coles_v2; June 11-13 locks remain v1
- Understat works via new POST endpoint (2025/26 xG confirmed); FBRef + SoFIFA 403 plain scripts — Wayback snapshots verified as clean fallback

## 2026-06-10 — Phase 2 Complete: Simulator, Champion Odds, First Live Lock-Ins
- Fixed simulator home-advantage orientation bug (D016); removed Elo mean reversion via backtest experiment (D015) — pooled Brier 0.2161 → 0.2056
- First champion table (10K sims): ESP 25.9%, ARG 22.6%, FRA 9.4%, BRA 5.5%, ENG 5.2%
- Lock-in flow live: June 11 + 12 predictions git-committed pre-kickoff (a604c75, 52b0749) with conservative midnight-UTC kickoff rule
- 77 tests green. Next: Phase 3 — player data (needs Kaggle creds or FBRef scrape), entity resolution

## 2026-06-10 — Phase 1 Complete: Fitted Baseline + Market Benchmark
- Real data loaded: 49K results (GitHub mirror, no Kaggle needed), WC odds workbook (football-data.co.uk — also covers 2018/2022/2014!), 2026 fixtures+groups (derived from fixture graph), 48 squads from Wikipedia (1,246 players)
- Live params fitted (D013); leak-free backtest pooled Brier 0.2161 vs market 0.1904 — H2 holds, kill criterion passed
- Fixed: Korea DPR mis-aliased to KOR, invalid build-backend, data/predictions/ was gitignored (broke D012)
- 69 tests green. Next: Phase 2 — validate simulator, champion probabilities, lock-in flow

## 2026-06-10 — Post-Critique Patch (v2)
- Fixed calibration landmine: Dixon-Coles params now fitted (Poisson regression + rho MLE), predict refuses unfitted params; dead code removed
- Added: tournament simulator (2026 48-team format, ET/penalty knockout model), odds loader (de-vig, CLV), git lock-in script, attention model scaffold
- Pre-registered hypotheses H1-H3 and kill criteria in DECISIONS.md (D007-D012); roadmap rewritten without calendar dates
- Next: Phase 1 — load real data, run fit_params, backtest baseline on 2018+2022

## 2026-06-04 — Project Scaffold
- Created full project structure: CLAUDE.md, 5 skill files, roadmap, source placeholders
- Defined champion-challenger architecture: Elo+Dixon-Coles baseline vs player-composition challengers
- Mapped 8 free data sources with coverage estimates
- Next: Phase 1 — data pipeline implementation (target June 7)
