# cfb-analytics Repository Guidelines & Invariants

## Operational Invariants
1. **Shadow Mode**: Nothing in this repository is decision-grade until Phase 4 clears the two-key promotion gate in `config/promotion.json`. Always stamp generated outputs with `UNPROMOTED - shadow output, not decision-grade`.
2. **US Eastern Slate Dates**: A slate is a US Eastern calendar date, never a UTC date. Kickoffs must be converted to `FOOTBALL_TZ` (`America/New_York`) when determining slate membership.
3. **Outlier Session**: Outlier tokens live 24 hours. The pipeline degrades gracefully if expired; refresh the session via the `outlier` project when needed.
4. **Devig & Consensus**: Devig each book against itself first, then aggregate. Spreads group on absolute line values. Negative hold and placeholder quotes (-100000) must be dropped/flagged.
5. **Canonical Game Identity**: CFBD keys (`cfbd:<id>`) are canonical; Outlier resolves onto CFBD games and writes no games or teams of its own. Games link to venues by `venue_id`, never by venue name.
6. **Ranked OVER board**: `over-board` is AP Top 25 games plus each conference's top two. Markets are team rushing, game rushing, team receiving, game receiving — OVER only, ranked by calibrated model probability (capped at 84%), with US Eastern kickoff times. Favorite receiving is dropped at spread ≤ −20 (sit-QB). SP+/SRS must not substitute for the AP poll. Player props are prohibited. Publishing freezes `over_board_snapshots`. `settle` grades that snapshot against `team_game_box` (rushing_yards / net_passing_yards), never a live rebuild. Close games do not auto-inflate pass volume. Calibrate by family (`favorite_team_rush` / `team_rec` / `game_yards`); size up a family only after ≥70% over 3–4 Saturdays.
