# Outlier NCAAFB Props & Insights Discovery Probe Report

- **Probe Execution Date:** 2026-09-07T06:19:17+00:00
- **Target Slate Date:** 2026-09-07 (US Eastern calendar date)
- **Probe Script:** `scripts/probe_ncaafb_outlier.py` (Git Commit: `58fb6b2`, Branch: `feat/outlier-props-insights`)
- **HTTP Client Mode:** LIVE
- **Authentication State:** LIVE_SESSION
- **Reference Slate Fixtures:** `tests/fixtures/outlier/`

## 1. Executive Summary & Gate Verdict

- **Milestone M0 Discovery Gate:** PASS
- **GAMELINE:** OFFERED (34 cards; 6 propositions)
- **TEAM_PROP:** OFFERED (49 cards; 14 propositions)
- **PLAYER_PROP:** OFFERED (117 cards; 28 propositions)
- **GAME_PROP:** OFFERED (24 cards; 16 propositions)
- **Insights:** OFFERED (35 insights)
- **Scope decision:** The live feed offers every probed market family and insights. R2 still admits only its explicit gameline and team-prop whitelist; player props and unwhitelisted game props remain excluded.

## 2. Slate & Sampled Events Context

- **Schedule status:** HTTP 200
- **Events on target slate:** 1
- **Events in returned schedule:** 95
- **Events probed:** 1

| Event ID | Kickoff (UTC) | Matchup |
|---|---|---|
| `d6478002c35b26c983511db774db09a5d27b9f90` | `2026-09-07T23:30:00+00:00` | Mustangs @ Seminoles |

## 3. Findings by Probed Token

### 3.1 Token: `GAMELINE`
- **Endpoint:** `GET /sportsdata/events/{eventId}/markets?marketType=GAMELINE`
- **HTTP status:** HTTP 200
- **Market cards:** 34
- **Discovered propositions:** `DOUBLE_RESULT`, `MONEYLINE`, `MONEYLINE_THREE_WAY`, `SPREAD`, `TOTAL`, `WINNING_MARGIN`
- **Resolution:** OFFERED (34 cards; 6 propositions)

### 3.2 Token: `TEAM_PROP`
- **Endpoint:** `GET /sportsdata/events/{eventId}/markets?marketType=TEAM_PROP`
- **HTTP status:** HTTP 200
- **Market cards:** 49
- **Discovered propositions:** `MADE_FIELD_GOALS`, `OFFENSIVE_TOUCHDOWNS`, `OFFENSIVE_YARDS`, `PASSING_TOUCHDOWNS`, `PASSING_YARDS`, `POINTS`, `RECEIVING_TOUCHDOWNS`, `RECEIVING_YARDS`, `RECEPTIONS`, `RUSHING_ATTEMPTS`, `RUSHING_RECEIVING_YARDS`, `RUSHING_TOUCHDOWNS`, `RUSHING_YARDS`, `TOUCHDOWNS`
- **Resolution:** OFFERED (49 cards; 14 propositions)

### 3.3 Token: `PLAYER_PROP`
- **Endpoint:** `GET /sportsdata/events/{eventId}/markets?marketType=PLAYER_PROP`
- **HTTP status:** HTTP 200
- **Market cards:** 117
- **Discovered propositions:** `EXTRA_POINTS`, `FANTASY_SCORE_PP`, `FANTASY_SCORE_UD`, `FIRST_TEAM_TOUCHDOWN`, `FIRST_TOUCHDOWN`, `INTERCEPTIONS_THROWN`, `KICKING_POINTS`, `LAST_TOUCHDOWN`, `LONGEST_PASSING_COMPLETION`, `LONGEST_RECEPTION`, `LONGEST_RUSH`, `MADE_FIELD_GOALS`, `MOST_PASSING_YARDS`, `MOST_RECEIVING_YARDS`, `MOST_RUSHING_YARDS`, `PASSING_ATTEMPTS`, `PASSING_COMPLETIONS`, `PASSING_RUSHING_YARDS`, `PASSING_TOUCHDOWNS`, `PASSING_YARDS`, `RECEIVING_TOUCHDOWNS`, `RECEIVING_YARDS`, `RECEPTIONS`, `RUSHING_ATTEMPTS`, `RUSHING_RECEIVING_YARDS`, `RUSHING_TOUCHDOWNS`, `RUSHING_YARDS`, `TOUCHDOWNS`
- **Resolution:** OFFERED (117 cards; 28 propositions)

### 3.4 Token: `GAME_PROP`
- **Endpoint:** `GET /sportsdata/events/{eventId}/markets?marketType=GAME_PROP`
- **HTTP status:** HTTP 200
- **Market cards:** 24
- **Discovered propositions:** `BTTS`, `MADE_FIELD_GOALS`, `OFFENSIVE_TOUCHDOWNS`, `OFFENSIVE_YARDS`, `PASSING_ATTEMPTS`, `PASSING_COMPLETIONS`, `PASSING_RUSHING_YARDS`, `PASSING_TOUCHDOWNS`, `PASSING_YARDS`, `RECEIVING_TOUCHDOWNS`, `RECEIVING_YARDS`, `RECEPTIONS`, `RUSHING_RECEIVING_YARDS`, `RUSHING_TOUCHDOWNS`, `RUSHING_YARDS`, `TOUCHDOWNS`
- **Resolution:** OFFERED (24 cards; 16 propositions)

### 3.5 Recorded parser traps
- **Non-parallel book lists:** Verified in event `d6478002c35b26c983511db774db09a5d27b9f90`: `outcome.books[0]` was `FANATICS` while `outcome.odds[0].book` was `DRAFTKINGS`.
- **Multi-row propositions:** `SPREAD` used 9 cards and `TOTAL` used 9 cards; consumers must union cards and deduplicate `(book, side, line)`.

## 4. Proposition & Sportsbook Depth

| Market Type | Proposition | Cards | Event Frequency | Distinct Books | Book Count | Median Books/Game | Consensus Eligible | R2 Action |
|---|---|---:|---:|---|---:|---:|---|---|
| `GAMELINE` | `DOUBLE_RESULT` | 1 | 1/1 (100%) | BETRIVERS, DRAFTKINGS, FANATICS, MIDNITE | 4 | 4.0 | `YES` | `DROP_UNWHITELISTED` |
| `GAMELINE` | `MONEYLINE` | 9 | 1/1 (100%) | BETRIVERS, CAESARS, DRAFTKINGS, FANATICS, ... | 13 | 13.0 | `YES` | `ADMIT_AS_ML` |
| `GAMELINE` | `MONEYLINE_THREE_WAY` | 4 | 1/1 (100%) | BETRIVERS, DRAFTKINGS, FANATICS, THESCOREBET | 4 | 4.0 | `YES` | `DROP_UNWHITELISTED` |
| `GAMELINE` | `SPREAD` | 9 | 1/1 (100%) | BETRIVERS, CAESARS, DRAFTKINGS, FANATICS, ... | 13 | 13.0 | `YES` | `ADMIT` |
| `GAMELINE` | `TOTAL` | 9 | 1/1 (100%) | BETRIVERS, CAESARS, DRAFTKINGS, FANATICS, ... | 13 | 13.0 | `YES` | `ADMIT` |
| `GAMELINE` | `WINNING_MARGIN` | 2 | 1/1 (100%) | DRAFTKINGS, FANDUEL | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `MADE_FIELD_GOALS` | 2 | 1/1 (100%) | FANATICS | 1 | 1.0 | `NO` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `OFFENSIVE_TOUCHDOWNS` | 2 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `OFFENSIVE_YARDS` | 2 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `ADMIT_FULL_GAME_ONLY` |
| `TEAM_PROP` | `PASSING_TOUCHDOWNS` | 4 | 1/1 (100%) | FANATICS, HARDROCK, HARDROCK_R | 3 | 3.0 | `YES` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `PASSING_YARDS` | 4 | 1/1 (100%) | FANATICS, HARDROCK, HARDROCK_R | 3 | 3.0 | `YES` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `POINTS` | 14 | 1/1 (100%) | BETRIVERS, CAESARS, DRAFTKINGS, FANATICS, ... | 11 | 11.0 | `YES` | `ADMIT_FULL_GAME_ONLY` |
| `TEAM_PROP` | `RECEIVING_TOUCHDOWNS` | 2 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `RECEIVING_YARDS` | 2 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `ADMIT_FULL_GAME_ONLY` |
| `TEAM_PROP` | `RECEPTIONS` | 2 | 1/1 (100%) | FANATICS, HARDROCK, HARDROCK_R | 3 | 3.0 | `YES` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `RUSHING_ATTEMPTS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `RUSHING_RECEIVING_YARDS` | 2 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `RUSHING_TOUCHDOWNS` | 4 | 1/1 (100%) | FANATICS, HARDROCK, HARDROCK_R | 3 | 3.0 | `YES` | `DROP_UNWHITELISTED` |
| `TEAM_PROP` | `RUSHING_YARDS` | 4 | 1/1 (100%) | FANATICS, HARDROCK, HARDROCK_R | 3 | 3.0 | `YES` | `ADMIT_FULL_GAME_ONLY` |
| `TEAM_PROP` | `TOUCHDOWNS` | 4 | 1/1 (100%) | DRAFTKINGS, FANATICS | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `PLAYER_PROP` | `EXTRA_POINTS` | 1 | 1/1 (100%) | FLIFF, UNDERDOG | 2 | 2.0 | `NO` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `FANTASY_SCORE_PP` | 8 | 1/1 (100%) | PRIZEPICKS | 1 | 1.0 | `NO` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `FANTASY_SCORE_UD` | 7 | 1/1 (100%) | UNDERDOG | 1 | 1.0 | `NO` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `FIRST_TEAM_TOUCHDOWN` | 1 | 1/1 (100%) | DRAFTKINGS, FANATICS, MIDNITE | 3 | 3.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `FIRST_TOUCHDOWN` | 1 | 1/1 (100%) | CAESARS, DRAFTKINGS, FANATICS, HARDROCK, ... | 7 | 7.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `INTERCEPTIONS_THROWN` | 2 | 1/1 (100%) | FLIFF, HARDROCK, HARDROCK_R, PRIZEPICKS, ... | 5 | 5.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `KICKING_POINTS` | 1 | 1/1 (100%) | FLIFF, HARDROCK, HARDROCK_R, PRIZEPICKS, ... | 6 | 6.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `LAST_TOUCHDOWN` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R, MIDNITE | 3 | 3.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `LONGEST_PASSING_COMPLETION` | 2 | 1/1 (100%) | FLIFF, HARDROCK, HARDROCK_R, PRIZEPICKS, ... | 5 | 5.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `LONGEST_RECEPTION` | 8 | 1/1 (100%) | FLIFF, HARDROCK, HARDROCK_R, PRIZEPICKS, ... | 5 | 5.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `LONGEST_RUSH` | 5 | 1/1 (100%) | FLIFF, PRIZEPICKS, UNDERDOG | 3 | 3.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `MADE_FIELD_GOALS` | 1 | 1/1 (100%) | FLIFF, HARDROCK, HARDROCK_R, PRIZEPICKS, ... | 5 | 5.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `MOST_PASSING_YARDS` | 1 | 1/1 (100%) | UNDERDOG | 1 | 1.0 | `NO` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `MOST_RECEIVING_YARDS` | 1 | 1/1 (100%) | UNDERDOG | 1 | 1.0 | `NO` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `MOST_RUSHING_YARDS` | 1 | 1/1 (100%) | UNDERDOG | 1 | 1.0 | `NO` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `PASSING_ATTEMPTS` | 2 | 1/1 (100%) | FANATICS, FLIFF, HARDROCK, HARDROCK_R, ... | 6 | 6.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `PASSING_COMPLETIONS` | 2 | 1/1 (100%) | FANATICS, FLIFF, HARDROCK, HARDROCK_R, ... | 7 | 7.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `PASSING_RUSHING_YARDS` | 2 | 1/1 (100%) | DRAFTKINGS, FLIFF, PRIZEPICKS, UNDERDOG | 4 | 4.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `PASSING_TOUCHDOWNS` | 2 | 1/1 (100%) | CAESARS, DRAFTKINGS, FANATICS, FLIFF, ... | 11 | 11.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `PASSING_YARDS` | 2 | 1/1 (100%) | CAESARS, DRAFTKINGS, FANATICS, FLIFF, ... | 11 | 11.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `RECEIVING_TOUCHDOWNS` | 14 | 1/1 (100%) | FLIFF, PRIZEPICKS | 2 | 2.0 | `NO` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `RECEIVING_YARDS` | 8 | 1/1 (100%) | CAESARS, DRAFTKINGS, FANATICS, FLIFF, ... | 11 | 11.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `RECEPTIONS` | 7 | 1/1 (100%) | FANATICS, FLIFF, HARDROCK, HARDROCK_R, ... | 7 | 7.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `RUSHING_ATTEMPTS` | 3 | 1/1 (100%) | FLIFF, PRIZEPICKS, UNDERDOG | 3 | 3.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `RUSHING_RECEIVING_YARDS` | 2 | 1/1 (100%) | DRAFTKINGS, FANATICS, FLIFF, HARDROCK, ... | 7 | 7.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `RUSHING_TOUCHDOWNS` | 7 | 1/1 (100%) | FLIFF, PRIZEPICKS | 2 | 2.0 | `NO` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `RUSHING_YARDS` | 5 | 1/1 (100%) | CAESARS, DRAFTKINGS, FANATICS, FLIFF, ... | 11 | 11.0 | `YES` | `EXCLUDED_R2` |
| `PLAYER_PROP` | `TOUCHDOWNS` | 20 | 1/1 (100%) | DRAFTKINGS, FANATICS, FLIFF, HARDROCK, ... | 8 | 8.0 | `YES` | `EXCLUDED_R2` |
| `GAME_PROP` | `BTTS` | 4 | 1/1 (100%) | DRAFTKINGS, FANATICS | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `MADE_FIELD_GOALS` | 1 | 1/1 (100%) | FANATICS | 1 | 1.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `OFFENSIVE_TOUCHDOWNS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `OFFENSIVE_YARDS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `PASSING_ATTEMPTS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `PASSING_COMPLETIONS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `PASSING_RUSHING_YARDS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `PASSING_TOUCHDOWNS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `PASSING_YARDS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `RECEIVING_TOUCHDOWNS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `RECEIVING_YARDS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `RECEPTIONS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `RUSHING_RECEIVING_YARDS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `RUSHING_TOUCHDOWNS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `RUSHING_YARDS` | 1 | 1/1 (100%) | HARDROCK, HARDROCK_R | 2 | 2.0 | `NO` | `DROP_UNWHITELISTED` |
| `GAME_PROP` | `TOUCHDOWNS` | 6 | 1/1 (100%) | DRAFTKINGS, FANATICS, FLIFF | 3 | 3.0 | `YES` | `DROP_UNWHITELISTED` |

## 5. Player Identity Field Assessment

- **Stable `playerId` present:** True
- **Display name present:** True
- **Field location:** `markets[].player.playerId` and `markets[].player.fullName` (outcome-level fields are also accepted by the probe).
- **Representative samples:**
  - `035834d907073001ba5920c93168858d5818dad9` — Desirrio Riles
  - `10081c490fdf6d8831d67292dd5b11a50597b248` — Quintrevion Wisner
  - `10081c490fdf6d8831d67292dd5b11a50597b248` — Quintrevion Wisner
  - `10081c490fdf6d8831d67292dd5b11a50597b248` — Quintrevion Wisner
  - `1f19cbea535d39a9fbb4f8e084fea601aea3fa8c` — Duce Robinson

## 6. Insights Endpoint Technical Assessment

- **HTTP status:** HTTP 200
- **Insights returned:** 35
- **Subject types:** PLAYER=32, TEAM=3
- **Market types:** GAMELINE=3, PLAYER_PROP=32
- **Propositions:** FANTASY_SCORE_PP=3, LONGEST_RECEPTION=2, LONGEST_RUSH=1, MONEYLINE=1, PASSING_ATTEMPTS=1, PASSING_COMPLETIONS=1, PASSING_TOUCHDOWNS=1, PASSING_YARDS=1, RECEIVING_YARDS=3, RECEPTIONS=5, RUSHING_RECEIVING_YARDS=1, RUSHING_TOUCHDOWNS=1, RUSHING_YARDS=3, SPREAD=1, TOTAL=1, TOUCHDOWNS=9

| Event ID | HTTP Status | Source | Insight Count |
|---|---:|---|---:|
| `d6478002c35b26c983511db774db09a5d27b9f90` | 200 | live | 35 |

## 7. Hard Gate Conclusion

- **Verdict:** PASS
- The authenticated live capture confirms that NCAAFB gamelines, team props, player props, game props, and insights are offered for the sampled in-window event. The discovery evidence supports proceeding only with the R2-authorized gameline and team-prop scope.
