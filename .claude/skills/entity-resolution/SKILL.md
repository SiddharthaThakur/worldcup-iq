---
name: entity-resolution
description: Matching player and team names across FBRef, Understat, StatsBomb, Transfermarkt, and other football data sources. The unglamorous but critical plumbing that makes multi-source player-level analysis possible.
---

# Entity Resolution Skill

## What This Skill Covers

The same player appears differently across sources:
- FBRef: "Kylian Mbappé Lottin"
- Understat: "Kylian Mbappe"
- StatsBomb: "Kylian Mbappé"
- Transfermarkt: "Kylian Mbappé"
- Wikipedia squad lists: "Mbappé"

This skill handles canonicalizing all of these to one player ID.

## Strategy: Two-Pass Matching

### Pass 1: Deterministic Match
Normalize all names:
1. Unicode NFKD decomposition → strip combining marks (accents)
2. Lowercase
3. Strip punctuation
4. Collapse whitespace

Match on `(normalized_name, birth_year)` or `(normalized_name, nationality, position)`.

This resolves ~80% of players.

### Pass 2: Fuzzy Match
For remaining unmatched players:
1. Use `rapidfuzz.fuzz.token_sort_ratio` (handles word reordering)
2. Threshold: ≥ 88 = auto-match, 80-87 = manual review, <80 = no match
3. Additional constraint: same nationality AND same position group
4. Log all fuzzy matches for human review in `data/processed/fuzzy_matches_review.csv`

### Pass 3: Manual Overrides
Maintain `data/processed/manual_mappings.csv`:
```csv
source,source_id,source_name,canonical_id,canonical_name,notes
fbref,abc123,"Son Heung-Min",uuid-xyz,"Son Heung-min","Capitalization difference"
understat,456,"Heung-Min Son",uuid-xyz,"Son Heung-min","Name order reversed"
```

This file is version-controlled and grows as new edge cases are found.

## Team Name Resolution

Simpler but necessary. Maintain `data/processed/team_aliases.json`:
```json
{
  "ARG": ["Argentina", "Argentine"],
  "KOR": ["Korea Republic", "South Korea", "Korea Rep."],
  "USA": ["United States", "USA", "US", "United States of America"],
  "CRC": ["Costa Rica"],
  "CIV": ["Ivory Coast", "Côte d'Ivoire"]
}
```

All internal code uses FIFA 3-letter codes.

## Canonical ID Table

`data/processed/player_ids.csv`:
```csv
canonical_id,canonical_name,birth_year,nationality,position,fbref_id,understat_id,statsbomb_id,transfermarkt_id
uuid-001,"Kylian Mbappé",1998,FRA,FWD,abc123,789,sb-456,tm-012
```

This is the single source of truth. All data pipeline modules join through this table.

## Coverage Tracking

After resolution, report coverage statistics:
```
Total WC squad players: 1,248
Matched to FBRef: 1,150 (92.1%)
Matched to Understat: 720 (57.7%) — Big 5 leagues only
Matched to StatsBomb: 380 (30.4%) — past WC/Euro participants only
Full coverage (FBRef + Understat): 700 (56.1%)
Partial coverage (FBRef only): 450 (36.1%)
No stats found: 98 (7.9%)
```

These numbers feed directly into per-player `data_quality` flags and model confidence.

## Common Pitfalls

- Players who changed names (marriage, legal name change)
- Players with identical names (need birth year or club to disambiguate)
- Transliteration differences for Arabic/Asian/Cyrillic names
- Players who switched nationality (FIFA allows one switch under certain conditions)
- Very young players with no prior club stats (first senior season)
