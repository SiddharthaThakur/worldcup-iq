---
name: data-pipeline
description: Guidance for ingesting, cleaning, and merging football data from multiple free sources (Kaggle, StatsBomb, FBRef, Understat, live APIs) into a unified schema for World Cup prediction.
---

# Data Pipeline Skill

## What This Skill Covers

Loading football data from 7+ free sources, resolving player/team names across them, and producing clean DataFrames ready for feature engineering. The core challenge is entity resolution — the same player appears as "Kylian Mbappé" in one source and "K. Mbappe" in another.

## Data Sources and Load Order

### Phase 1 (Pre-Tournament Essentials)
1. **International match results** — Kaggle CSV, 49K+ rows. Load with pandas. Filter to post-2010 for relevance. Key columns: date, home_team, away_team, home_score, away_score, tournament, neutral.
2. **Elo ratings** — Kaggle CSV for all 48 qualified teams. These seed the baseline model.
3. **2026 WC fixtures and squads** — From worldcup2026 API or Wikipedia scrape. Need: group assignments, match schedule, 26-man squads per team.

### Phase 2 (Player-Level Data)
4. **Player season stats** — Kaggle Top 5 League stats OR FBRef scraping. Per-90 stats: goals, assists, shots, key passes, tackles, interceptions.
5. **Understat xG/xA** — Kaggle or direct scraping. Per-player: xG, xA, xGChain, xGBuildup. Big 5 leagues only.
6. **Club match lineups** — schochastics/football-data GitHub repo, `data/formations/` folder. Parquet files with match lineups. Used to train composition models.

### Phase 3 (Evaluation + Live)
7. **Bookmaker odds** — football-data.co.uk CSVs. Historical odds for international matches. For 2026 WC: scrape or manual entry from OddsPortal.
8. **Live 2026 API** — `https://api.worldcup2026.com` or equivalent. Poll during matches for live scores.

## Entity Resolution Strategy

Player names are the hardest problem. Use a two-pass approach:

1. **Exact match first** — Normalize unicode, strip accents, lowercase. Match on `(normalized_name, birth_year, nationality)` triple.
2. **Fuzzy match second** — Use `rapidfuzz.fuzz.token_sort_ratio` with threshold ≥ 85. Manual review for matches between 85-92.
3. **Maintain a canonical player ID table** — `data/processed/player_ids.csv` maps source-specific IDs to a canonical UUID. This file is version-controlled and grows over time.

Team names are easier but still need a lookup: "Korea Republic" = "South Korea" = "KOR".

## Schema Conventions

- All dates as `YYYY-MM-DD` strings or pandas Timestamps
- Player stats always per-90-minutes (not raw totals)
- Match IDs: `{date}_{home_team_code}_{away_team_code}` (e.g., `2026-06-11_ARG_MAR`)
- Player IDs: canonical UUIDs in `player_ids.csv`
- Team codes: FIFA 3-letter codes (ARG, BRA, FRA, etc.)

## Rate Limiting

- FBRef: 3-second delay between requests. Max ~20 pages per minute.
- Understat: No aggressive protection but be respectful. 1-second delays.
- worldcup2026 API: Check rate limit headers. Cache responses.

## Data Quality Flags

Every player row should have a `data_quality` field:
- `"full"` — Big 5 league player with FBRef + Understat data
- `"partial"` — FBRef basic stats only (non-Big-5 league or Understat missing)
- `"minimal"` — Only squad membership known, no club stats available
- `"none"` — Player exists in squad but no stats found

These flags feed into the model's confidence/trust gauge.

## File Outputs

- `data/processed/international_results.parquet` — Clean match results
- `data/processed/player_stats.parquet` — Unified player stats with canonical IDs
- `data/processed/player_ids.csv` — Entity resolution mapping table
- `data/processed/wc2026_squads.json` — 48 teams × 26 players with mapped IDs
- `data/processed/wc2026_fixtures.json` — 104 match schedule
- `data/processed/club_lineups.parquet` — Club match lineups for composition training
- `data/processed/bookmaker_odds.parquet` — Historical + 2026 WC odds
