"""Data sources.

Each source is isolated behind its own module so a brittle one (ESPN) failing
degrades confidence rather than crashing a run, and so a source can be dropped
without touching the ingest layer.

Verified status as of 2026-09-04:

* ``outlier``  - LIVE. League token ``NCAAFB``. Schedule, gameline odds
  (19-20 books incl. PS3838/Pinnacle and CIRCA), structured injuries.
* ``cfbd``     - LIVE where ``CFBD_API_KEY`` is set. Teams, venues, games,
  ratings, advanced stats, betting lines, and player data. The only source
  a scheduled job can rely on: a static key, no session, no OTP.
* ``weather``  - LIVE. Open-Meteo, no credential. Forecast + ERA5 archive,
  keyed to stadium coordinates via ``games.venue_id``.
* ``espn``     - **dropped.** Probed 2026-09-04: ESPN publishes no depth chart
  for college football. ``/teams/{id}/depthchart`` returns an empty object and
  the core-API variants return 400/404. Its roster endpoint does work, but
  carries no ``starter`` field, so it cannot confirm a starting quarterback --
  the one thing it was wanted for.

  Everything ESPN *could* have contributed, CFBD supplies with a proper API
  key and better structure: ``/roster`` (127 players with position),
  ``/player/usage`` (usage share), ``/player/returning`` (passing usage and
  PPA share), and ``/stats/player/season?category=passing`` (ATT, COMPLETIONS,
  YDS, TD, INT, PCT, YPA per player). QB experience and backup quality are
  therefore CFBD features, not ESPN ones.

  **The hard CORE blocker stands.** No free source confirms who starts at
  quarterback, so ``qb_status='unknown'`` continues to disqualify a leg from
  the CORE tier rather than being softened into a confidence deduction.
"""
