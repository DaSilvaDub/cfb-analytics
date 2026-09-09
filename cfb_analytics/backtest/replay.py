"""Game replay engine: re-creates GameState checkpoints from historical PBP.

Loads drives and plays from the database for a single game, feeds them
through ``LiveStateTracker`` to produce ``GameState`` snapshots at
configurable checkpoints (end-of-quarter, drive-start, halftime).

This is the bridge between stored CFBD play-by-play data and the live
micro-markets model's backtesting harness.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from cfb_analytics.features.live import (
    QUARTER_SECONDS,
    DriveOutcome,
    GameState,
    LiveStateTracker,
)

# Maps CFBD drive result strings to DriveOutcome enum values.
CFBD_RESULT_MAP: dict[str, DriveOutcome] = {
    "TOUCHDOWN": DriveOutcome.TOUCHDOWN,
    "TD": DriveOutcome.TOUCHDOWN,
    "FIELD GOAL": DriveOutcome.FIELD_GOAL,
    "FG": DriveOutcome.FIELD_GOAL,
    "FIELD_GOAL": DriveOutcome.FIELD_GOAL,
    "FG GOOD": DriveOutcome.FIELD_GOAL,
    "MADE FG": DriveOutcome.FIELD_GOAL,
    "PUNT": DriveOutcome.PUNT,
    "FUMBLE": DriveOutcome.TURNOVER,
    "INTERCEPTION": DriveOutcome.TURNOVER,
    "INT": DriveOutcome.TURNOVER,
    "FUMBLE RECOVERY (OWN)": DriveOutcome.TURNOVER,
    "FUMBLE RECOVERY (OPPONENT)": DriveOutcome.TURNOVER,
    "TURNOVER": DriveOutcome.TURNOVER,
    "TURNOVER ON DOWNS": DriveOutcome.DOWNS,
    "DOWNS": DriveOutcome.DOWNS,
    "MISSED FG": DriveOutcome.DOWNS,
    "MISSED FIELD GOAL": DriveOutcome.DOWNS,
    "FG MISSED": DriveOutcome.DOWNS,
    "SAFETY": DriveOutcome.SAFETY,
    "END OF HALF": DriveOutcome.END_HALF,
    "END OF 4TH QUARTER": DriveOutcome.END_GAME,
    "END OF GAME": DriveOutcome.END_GAME,
    "KICKOFF": DriveOutcome.PUNT,  # Treat kickoff drives as non-scoring
    "UNCATEGORIZED": DriveOutcome.DOWNS,
}


@dataclass(frozen=True)
class CheckpointLabel:
    """Identifies when/why a checkpoint was emitted."""

    kind: str  # "end_of_quarter" | "drive_start" | "halftime"
    quarter: int
    drive_number: int | None = None


@dataclass(frozen=True)
class ReplayCheckpoint:
    """A replayed game state snapshot with ground-truth labels."""

    label: CheckpointLabel
    state: GameState
    actual_drive_outcome: str | None = None  # for drive_start only
    actual_home_final: int = 0
    actual_away_final: int = 0


def _map_cfbd_result(result: str | None) -> DriveOutcome:
    """Map a CFBD drive result string to a DriveOutcome enum value."""
    if result is None:
        return DriveOutcome.DOWNS
    normalized = result.strip().upper()
    return CFBD_RESULT_MAP.get(normalized, DriveOutcome.DOWNS)


def _is_scrimmage_play(play_row: dict) -> bool:
    """Filter out non-scrimmage plays (kickoffs, PATs, etc.)."""
    down = play_row["down"]
    play_type = (play_row["play_type"] or "").lower()
    # Null/0 down indicates non-scrimmage (kickoff, extra point, etc.)
    if down is None or down == 0:
        return False
    return play_type not in (
        "kickoff",
        "kickoff return (touchback)",
        "extra point",
        "two point conversion",
        "pat",
    )


def _safe_yardline(yard_line: int | None, offense_team_id: str, home_team_id: str) -> int:
    """Normalize CFBD yardline to 'yards to opponent goal line' (1-99).

    CFBD yardLine is "yards from offense's own goal line" (0-100).
    We need distance to opponent goal = 100 - yardLine.
    """
    if yard_line is None:
        return 75  # Default: own 25, standard touchback
    yl = max(1, min(99, 100 - yard_line))
    return yl


def replay_game(
    conn: sqlite3.Connection,
    game_id: str,
    *,
    checkpoint_kinds: frozenset[str] = frozenset({"end_of_quarter", "drive_start", "halftime"}),
) -> list[ReplayCheckpoint]:
    """Replay a historical game from stored PBP, emitting checkpoints.

    Returns an empty list if the game has no drives/plays stored.
    """
    # 1. Load game metadata
    game_row = conn.execute(
        """SELECT game_id, home_team_id, away_team_id, home_points, away_points,
                  kickoff_utc
           FROM games WHERE game_id = ?""",
        (game_id,),
    ).fetchone()
    if game_row is None:
        return []

    home_team = str(game_row["home_team_id"])
    away_team = str(game_row["away_team_id"])
    home_final = int(game_row["home_points"] or 0)
    away_final = int(game_row["away_points"] or 0)
    kickoff_utc = str(game_row["kickoff_utc"] or "2020-01-01T00:00:00+00:00")

    # 2. Load drives ordered by drive_number
    drives = conn.execute(
        """SELECT drive_id, drive_number, offense_team_id, defense_team_id,
                  scoring, start_period, start_yardline, start_time_minutes,
                  start_time_seconds, result
           FROM drives WHERE game_id = ? ORDER BY drive_number""",
        (game_id,),
    ).fetchall()
    if not drives:
        return []

    # 3. Load all plays, group by drive_id
    all_plays = conn.execute(
        """SELECT play_id, drive_id, offense_team_id, defense_team_id,
                  play_number, period, clock_minutes, clock_seconds,
                  yard_line, down, distance, yards_gained, play_type,
                  scoring, ppa
           FROM plays WHERE game_id = ? ORDER BY period, play_number""",
        (game_id,),
    ).fetchall()

    plays_by_drive: dict[str, list[dict]] = {}
    for play in all_plays:
        did = str(play["drive_id"])
        plays_by_drive.setdefault(did, []).append(dict(play))

    # 4. Create LiveStateTracker
    first_drive = drives[0]
    initial_possession = str(first_drive["offense_team_id"])
    initial_yardline = _safe_yardline(first_drive["start_yardline"], initial_possession, home_team)

    tracker = LiveStateTracker(
        home_team_id=home_team,
        away_team_id=away_team,
        initial_possession_team_id=initial_possession,
        game_id=game_id,
        initial_observed_utc=kickoff_utc,
        initial_ingested_utc=kickoff_utc,
        source="replay",
        initial_source_sequence=0,
        initial_quarter=first_drive["start_period"] or 1,
        initial_clock_seconds=min(
            QUARTER_SECONDS,
            (first_drive["start_time_minutes"] or 15) * 60
            + (first_drive["start_time_seconds"] or 0),
        ),
        initial_yardline=initial_yardline,
    )

    checkpoints: list[ReplayCheckpoint] = []
    sequence = 1
    last_quarter_emitted: set[int] = set()

    for drive_row in drives:
        drive_id = str(drive_row["drive_id"])
        offense_id = str(drive_row["offense_team_id"])
        drive_number = int(drive_row["drive_number"])
        start_period = drive_row["start_period"] or 1
        start_yl = _safe_yardline(drive_row["start_yardline"], offense_id, home_team)
        start_mins = drive_row["start_time_minutes"]
        start_secs = drive_row["start_time_seconds"]
        start_clock = min(
            QUARTER_SECONDS,
            (start_mins or 15) * 60 + (start_secs or 0),
        )
        cfbd_result = drive_row["result"]
        drive_outcome = _map_cfbd_result(cfbd_result)

        # Start the drive
        try:
            tracker.start_drive(
                drive_id=drive_id,
                possession_team_id=offense_id,
                quarter=start_period,
                clock_seconds=start_clock,
                start_yardline=start_yl,
                observed_utc=kickoff_utc,
                ingested_utc=kickoff_utc,
                source_sequence=sequence,
            )
            sequence += 1
        except Exception:
            continue  # Skip drives with invalid data

        # Emit drive_start checkpoint
        if "drive_start" in checkpoint_kinds:
            state = tracker.get_current_state()
            checkpoints.append(
                ReplayCheckpoint(
                    label=CheckpointLabel(
                        kind="drive_start",
                        quarter=start_period,
                        drive_number=drive_number,
                    ),
                    state=state,
                    actual_drive_outcome=drive_outcome.value,
                    actual_home_final=home_final,
                    actual_away_final=away_final,
                )
            )

        # Feed plays
        drive_plays = plays_by_drive.get(drive_id, [])
        scrimmage_plays = [p for p in drive_plays if _is_scrimmage_play(p)]
        points_on_drive = 0

        for i, play in enumerate(scrimmage_plays):
            yards = play["yards_gained"] or 0
            is_last = i == len(scrimmage_plays) - 1
            play_period = play["period"] or start_period
            play_clock_mins = play["clock_minutes"]
            play_clock_secs = play["clock_seconds"]
            play_clock = min(
                QUARTER_SECONDS,
                (play_clock_mins or 0) * 60 + (play_clock_secs or 0),
            )
            play_yl = _safe_yardline(play["yard_line"], offense_id, home_team)

            is_scoring = bool(play["scoring"])
            pts = 0
            scoring_team = ""
            if is_scoring and is_last:
                # Attribute points based on drive outcome
                if drive_outcome == DriveOutcome.TOUCHDOWN:
                    pts = 7
                    scoring_team = offense_id
                elif drive_outcome == DriveOutcome.FIELD_GOAL:
                    pts = 3
                    scoring_team = offense_id
                elif drive_outcome == DriveOutcome.SAFETY:
                    pts = 2
                    defense_id = str(drive_row["defense_team_id"])
                    scoring_team = defense_id
                points_on_drive += pts

            # Determine next down/distance/yardline from next play, or from drive end
            if i + 1 < len(scrimmage_plays):
                next_play = scrimmage_plays[i + 1]
                next_down = next_play["down"] or 1
                next_dist = next_play["distance"] or 10
                next_yl = _safe_yardline(next_play["yard_line"], offense_id, home_team)
                next_q = next_play["period"] or play_period
                next_clock_mins = next_play["clock_minutes"]
                next_clock_secs = next_play["clock_seconds"]
                next_clock = min(
                    QUARTER_SECONDS,
                    (next_clock_mins or 0) * 60 + (next_clock_secs or 0),
                )
            else:
                # Last play of drive
                next_down = 1
                next_dist = 10
                next_yl = max(1, min(99, play_yl - yards))
                next_q = play_period
                next_clock = max(0, play_clock - 30)  # Estimate ~30 seconds

            try:
                tracker.record_play(
                    play_id=str(play["play_id"]),
                    yards_gained=float(yards),
                    next_down=max(1, min(4, next_down)),
                    next_distance=max(1, min(99, next_dist)),
                    next_yardline=max(1, min(99, next_yl)),
                    next_quarter=next_q,
                    next_clock_seconds=max(0, min(QUARTER_SECONDS, next_clock)),
                    play_type=play["play_type"] or "rush",
                    epa=float(play["ppa"] or 0.0),
                    is_scoring=is_scoring and pts > 0,
                    points_scored=pts,
                    scoring_team_id=scoring_team,
                    observed_utc=kickoff_utc,
                    ingested_utc=kickoff_utc,
                    source_sequence=sequence,
                )
                sequence += 1
            except Exception:
                continue

        # End the drive
        next_poss = away_team if offense_id == home_team else home_team
        try:
            tracker.end_drive(
                outcome=drive_outcome.value,
                next_possession_team_id=next_poss,
                points=points_on_drive,
                observed_utc=kickoff_utc,
                ingested_utc=kickoff_utc,
                source_sequence=sequence,
            )
            sequence += 1
        except Exception:
            # Force reset drive state if end_drive fails
            tracker.drive_active = False
            tracker.active_drive_plays.clear()
            continue

        # Emit end_of_quarter checkpoints
        current_q = tracker.quarter
        if (
            "end_of_quarter" in checkpoint_kinds
            and current_q not in last_quarter_emitted
            and (
                start_period != current_q
                or (start_period == current_q and tracker.clock_seconds == 0)
            )
        ):
            last_quarter_emitted.add(start_period)
            state = tracker.get_current_state()
            checkpoints.append(
                ReplayCheckpoint(
                    label=CheckpointLabel(
                        kind="end_of_quarter",
                        quarter=start_period,
                    ),
                    state=state,
                    actual_home_final=home_final,
                    actual_away_final=away_final,
                )
            )

        # Emit halftime checkpoint
        if (
            "halftime" in checkpoint_kinds
            and start_period == 2
            and 3 not in last_quarter_emitted
            and current_q >= 3
        ):
            state = tracker.get_current_state()
            checkpoints.append(
                ReplayCheckpoint(
                    label=CheckpointLabel(
                        kind="halftime",
                        quarter=2,
                    ),
                    state=state,
                    actual_home_final=home_final,
                    actual_away_final=away_final,
                )
            )

    return checkpoints
