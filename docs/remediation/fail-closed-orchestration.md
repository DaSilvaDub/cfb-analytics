# CFB publication hold and orchestration repair

Deliverables 7 and 8 classify the current outputs as NOT READY. The checked-in
publication hold and the host quarantine hold are checked before a producer,
database write, export, or publication starts. There is no command-line bypass.
Release requires independent revalidation; passing these mechanical tests does
not make the CFB outputs decision-grade.

The daily and next-day runners now use terminating PowerShell errors and checked
native process exits. Standard output and error are drained separately to avoid
PowerShell 5.1 turning normal native stderr logging into a terminating error.
Reported HTTP 403, ENOSPC, database lock, slate mismatch and zero matches stop the
run even if a producer incorrectly returns zero. Database-journal presence and
the C: free-space floor are checked before execution. Next-day ingestion no
longer treats a recent database mtime as proof of a successful current run.

Before sync, the simulation must identify the requested Eastern slate, have a
current generation timestamp, and contain matching games. The export must be
fresh, nonempty, and have canonical game identities and matching kickoff dates.
All selected reports must be current and nonempty; files are hash-locked into a
staging directory before any publication directories or copies are created.
T-1 snapshots reject stale sources and refuse to overwrite an existing baseline.

`sync_to_drive.ps1` now requires `-RunStartedUtc`; other legacy callers must pass
their successful run's start time before they can publish. That deliberate
failure is safer than accepting old reports based on file existence.

Validation: 14 offline PowerShell subprocess regressions passed; Ruff passed;
all six PowerShell files parse successfully. The intraday refresher and nightly
audit also check the hold before their producers and check native exits.
No live CFB pipeline, database,
calibration, paid provider, or Drive publication was executed.

Limits: this is mechanical containment, not repaired model calibration or a
provider-authenticated provenance chain. Copy failures during the final commit
phase can leave a partial destination set; the active hold prevents that phase
in current operation. Cross-file transactional publication needs a versioned
bundle/pointer protocol before the hold can be released. Existing producer
schemas that omit the required identity evidence remain withheld.
