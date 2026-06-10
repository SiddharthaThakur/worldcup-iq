# WorldCupIQ Progress Log

<!-- Newest entries at the top. 2-4 lines per entry. -->

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
