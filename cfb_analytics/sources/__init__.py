"""Data sources.

Each source is isolated behind its own module so a brittle one (ESPN) failing
degrades confidence rather than crashing a run, and so a source can be dropped
without touching the ingest layer.

Verified status as of 2026-08-31:

* ``outlier``  - LIVE. League token ``NCAAFB``. Schedule, gameline odds
  (19-20 books incl. PS3838/Pinnacle and CIRCA), structured injuries.
* ``cfbd``     - BLOCKED. Requires the ``CFBD_API_KEY`` environment variable.
* ``espn``     - not yet implemented. Needed only for starting-QB confirmation.
* ``weather``  - not yet implemented (Open-Meteo, no key required).
"""
