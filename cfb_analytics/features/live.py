"""In-game play-by-play state tracking and momentum features.

Represents continuous in-game state (quarter, clock, down, distance,
yardline, score differential, possession, timeouts) and rolling
drive-level EPA and success-rate momentum. Strictly team-level;
no player props. Pure standard library design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from cfb_analytics.errors import SchemaError

QUARTER_SECONDS: int = 900
HALF_SECONDS: int = 1800
REGULATION_SECONDS: int = 3600

# FBS baseline benchmarks for collegiate football
BASELINE_SUCCESS_RATE: float = 0.42
BASELINE_EPA_PER_PLAY: float = 0.00
DEFAULT_MOMENTUM_WINDOW_DRIVES: int = 3


class DriveOutcome(StrEnum):
    """Primary terminal drive outcomes in college football."""

    TOUCHDOWN = "TD"
    FIELD_GOAL = "FG"
    PUNT = "PUNT"
    TURNOVER = "TURNOVER"
    DOWNS = "DOWNS"
    SAFETY = "SAFETY"
    END_HALF = "END_HALF"
    END_GAME = "END_GAME"


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise SchemaError(f"{name} must be numeric, got {value!r}")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(resolved):
        raise SchemaError(f"{name} must be finite, got {value!r}")
    return resolved


def _aware_timestamp(value: str, *, name: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} must be a parseable ISO timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise SchemaError(f"{name} must include a UTC offset")
    return stamp


def _required_text(value: str, *, name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise SchemaError(f"{name} is required")
    return resolved


def is_successful_play(down: int, distance: int, yards_gained: float) -> bool:
    """Standard college football analytics play success criteria.

    - 1st down: gain >= 50% of needed distance.
    - 2nd down: gain >= 70% of needed distance.
    - 3rd / 4th down: gain >= 100% of needed distance (first down conversion).
    """
    gained = _finite_float(yards_gained, name="yards_gained")
    if not (1 <= down <= 4) or isinstance(down, bool):
        raise SchemaError(f"Down must be in [1, 4], got {down}")
    if not (1 <= distance <= 99) or isinstance(distance, bool):
        raise SchemaError(f"Distance must be in [1, 99], got {distance}")
    if down == 1:
        return gained >= 0.5 * distance
    elif down == 2:
        return gained >= 0.7 * distance
    elif down in (3, 4):
        return gained >= float(distance)
    return gained > 0.0


@dataclass(frozen=True)
class LivePlay:
    """A single play-by-play event in an ongoing drive."""

    play_id: str
    drive_id: str
    team_id: str
    quarter: int
    clock_seconds: int
    down: int
    distance: int
    yardline: int
    yards_gained: float = 0.0
    play_type: str = "rush"
    epa: float = 0.0
    is_success: bool | None = None
    is_turnover: bool = False
    is_scoring: bool = False
    points_scored: int = 0
    scoring_team_id: str = ""
    game_id: str = ""
    observed_utc: str = ""
    ingested_utc: str = ""
    source: str = ""
    source_sequence: int = -1

    def __post_init__(self) -> None:
        _required_text(self.play_id, name="play_id")
        _required_text(self.drive_id, name="drive_id")
        _required_text(self.team_id, name="team_id")
        _required_text(self.game_id, name="game_id")
        _required_text(self.source, name="source")
        observed = _aware_timestamp(self.observed_utc, name="observed_utc")
        ingested = _aware_timestamp(self.ingested_utc, name="ingested_utc")
        if ingested < observed:
            raise SchemaError("ingested_utc cannot precede observed_utc")
        if self.source_sequence < 0 or isinstance(self.source_sequence, bool):
            raise SchemaError("source_sequence must be non-negative")
        if self.quarter < 1 or isinstance(self.quarter, bool):
            raise SchemaError(f"Quarter must be >= 1, got {self.quarter}")
        if not (0 <= self.clock_seconds <= QUARTER_SECONDS):
            raise SchemaError("clock_seconds must be within the current quarter")
        if not (1 <= self.down <= 4):
            raise SchemaError(f"Down must be in [1, 4], got {self.down}")
        if not (1 <= self.distance <= 99):
            raise SchemaError(f"Distance must be in [1, 99], got {self.distance}")
        if not (0 <= self.yardline <= 100):
            raise SchemaError(f"Yardline must be in [0, 100], got {self.yardline}")
        _finite_float(self.yards_gained, name="yards_gained")
        _finite_float(self.epa, name="epa")
        if self.points_scored < 0 or isinstance(self.points_scored, bool):
            raise SchemaError("points_scored must be a non-negative integer")
        if self.points_scored > 8:
            raise SchemaError(
                f"points_scored cannot exceed 8 in a single play, got {self.points_scored}"
            )
        if self.points_scored > 0 and not self.is_scoring:
            raise SchemaError("positive points_scored requires is_scoring=True")
        if self.points_scored > 0 and not self.scoring_team_id.strip():
            raise SchemaError("scoring_team_id is required when points are scored")
        if self.is_success is None:
            if self.is_turnover:
                success = False
            else:
                success = self.is_scoring or is_successful_play(
                    self.down, self.distance, self.yards_gained
                )
            object.__setattr__(self, "is_success", success)


@dataclass(frozen=True)
class LiveDrive:
    """An offensive drive comprising one or more plays."""

    drive_id: str
    team_id: str
    start_quarter: int
    start_clock_seconds: int
    start_yardline: int
    end_yardline: int | None = None
    outcome: str | None = None
    plays: tuple[LivePlay, ...] = ()
    drive_epa: float = 0.0
    play_count: int = 0
    success_count: int = 0
    yards_gained: float = 0.0
    points: int = 0

    def __post_init__(self) -> None:
        _required_text(self.drive_id, name="drive_id")
        _required_text(self.team_id, name="team_id")
        if self.outcome is not None:
            try:
                DriveOutcome(self.outcome)
            except ValueError as exc:
                raise SchemaError(f"Unknown drive outcome {self.outcome!r}") from exc
        if self.play_count != len(self.plays) or self.success_count > self.play_count:
            raise SchemaError("Drive play/success counts do not match its plays")
        _finite_float(self.drive_epa, name="drive_epa")
        _finite_float(self.yards_gained, name="yards_gained")
        if self.points < 0 or isinstance(self.points, bool):
            raise SchemaError("Drive points must be a non-negative integer")

    @property
    def success_rate(self) -> float:
        return self.success_count / self.play_count if self.play_count > 0 else 0.0

    @property
    def epa_per_play(self) -> float:
        return self.drive_epa / self.play_count if self.play_count > 0 else 0.0


@dataclass(frozen=True)
class TeamMomentum:
    """Rolling momentum metrics across recent drives."""

    team_id: str = ""
    rolling_drive_epa: float = 0.0
    rolling_epa_per_play: float = 0.0
    rolling_success_rate: float = BASELINE_SUCCESS_RATE
    drives_observed: int = 0
    plays_observed: int = 0
    momentum_factor: float = 0.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.rolling_drive_epa, "rolling_drive_epa"),
            (self.rolling_epa_per_play, "rolling_epa_per_play"),
            (self.rolling_success_rate, "rolling_success_rate"),
            (self.momentum_factor, "momentum_factor"),
        ):
            _finite_float(value, name=name)
        if not 0.0 <= self.rolling_success_rate <= 1.0:
            raise SchemaError("rolling_success_rate must be in [0, 1]")
        if self.drives_observed < 0 or self.plays_observed < 0:
            raise SchemaError("Observed drive and play counts must be non-negative")


@dataclass(frozen=True)
class GameState:
    """Point-in-time representation of live college football game state."""

    home_team_id: str
    away_team_id: str
    possession_team_id: str
    quarter: int
    clock_seconds: int
    down: int
    distance: int
    yardline: int  # Distance to opponent goal line (1 to 99 yards)
    home_score: int = 0
    away_score: int = 0
    home_timeouts: int = 3
    away_timeouts: int = 3
    home_momentum: TeamMomentum = field(default_factory=TeamMomentum)
    away_momentum: TeamMomentum = field(default_factory=TeamMomentum)
    game_id: str = ""
    observed_utc: str = ""
    ingested_utc: str = ""
    source: str = ""
    source_sequence: int = -1
    is_final: bool = False

    def __post_init__(self) -> None:
        _required_text(self.home_team_id, name="home_team_id")
        _required_text(self.away_team_id, name="away_team_id")
        _required_text(self.game_id, name="game_id")
        _required_text(self.source, name="source")
        if self.home_team_id == self.away_team_id:
            raise SchemaError("Home and away team IDs must differ")
        observed = _aware_timestamp(self.observed_utc, name="observed_utc")
        ingested = _aware_timestamp(self.ingested_utc, name="ingested_utc")
        if ingested < observed:
            raise SchemaError("ingested_utc cannot precede observed_utc")
        if self.source_sequence < 0 or isinstance(self.source_sequence, bool):
            raise SchemaError("source_sequence must be non-negative")
        if self.quarter < 1:
            raise SchemaError(f"Quarter must be >= 1, got {self.quarter}")
        if not (0 <= self.clock_seconds <= QUARTER_SECONDS):
            raise SchemaError(
                f"clock_seconds must be within [0, {QUARTER_SECONDS}], got {self.clock_seconds}"
            )
        if not (1 <= self.down <= 4):
            raise SchemaError(f"Down must be in [1, 4], got {self.down}")
        if not (1 <= self.distance <= 99):
            raise SchemaError(f"Distance must be in [1, 99], got {self.distance}")
        if not (0 <= self.yardline <= 100):
            raise SchemaError(f"Yardline must be in [0, 100], got {self.yardline}")
        if (
            self.home_score < 0
            or self.away_score < 0
            or isinstance(self.home_score, bool)
            or isinstance(self.away_score, bool)
        ):
            raise SchemaError("Scores must be non-negative integers")
        if not (0 <= self.home_timeouts <= 3) or not (0 <= self.away_timeouts <= 3):
            raise SchemaError("Timeouts must be between 0 and 3")
        if self.possession_team_id not in (self.home_team_id, self.away_team_id):
            raise SchemaError(
                f"Possession team {self.possession_team_id!r} must be either home "
                f"({self.home_team_id!r}) or away ({self.away_team_id!r})"
            )
        if self.home_momentum.team_id not in ("", self.home_team_id):
            raise SchemaError("home_momentum team_id does not match home_team_id")
        if self.away_momentum.team_id not in ("", self.away_team_id):
            raise SchemaError("away_momentum team_id does not match away_team_id")
        if self.is_final and self.home_score == self.away_score:
            raise SchemaError("A final college-football game cannot have a tied score")

    @property
    def seconds_remaining_in_quarter(self) -> int:
        """Seconds remaining in the current 15-minute quarter."""
        return self.clock_seconds

    @property
    def seconds_remaining_in_half(self) -> int:
        """Seconds remaining in the current half (Q1+Q2 or Q3+Q4)."""
        if self.quarter in (1, 3):
            return self.clock_seconds + QUARTER_SECONDS
        elif self.quarter in (2, 4):
            return self.clock_seconds
        return 0  # Overtime is untimed

    @property
    def seconds_remaining_in_game(self) -> int:
        """Seconds remaining in regulation (0 to 3600 seconds)."""
        if self.quarter == 1:
            return self.clock_seconds + 3 * QUARTER_SECONDS
        elif self.quarter == 2:
            return self.clock_seconds + 2 * QUARTER_SECONDS
        elif self.quarter == 3:
            return self.clock_seconds + QUARTER_SECONDS
        elif self.quarter == 4:
            return self.clock_seconds
        return 0  # Overtime

    @property
    def elapsed_game_seconds(self) -> int:
        """Seconds elapsed in regulation."""
        return max(0, min(REGULATION_SECONDS, REGULATION_SECONDS - self.seconds_remaining_in_game))

    @property
    def is_overtime(self) -> bool:
        """True if the game has progressed to overtime (quarter >= 5)."""
        return self.quarter >= 5

    @property
    def defense_team_id(self) -> str:
        """The team currently on defense."""
        return (
            self.away_team_id if self.possession_team_id == self.home_team_id else self.home_team_id
        )

    @property
    def score_differential(self) -> int:
        """Score differential from the perspective of the possession team."""
        if self.possession_team_id == self.home_team_id:
            return self.home_score - self.away_score
        return self.away_score - self.home_score

    @property
    def home_score_differential(self) -> int:
        """Score differential from home team perspective."""
        return self.home_score - self.away_score

    @property
    def away_score_differential(self) -> int:
        """Score differential from away team perspective."""
        return self.away_score - self.home_score

    @property
    def possession_timeouts(self) -> int:
        """Timeouts remaining for the team on offense."""
        return (
            self.home_timeouts
            if self.possession_team_id == self.home_team_id
            else self.away_timeouts
        )

    @property
    def defense_timeouts(self) -> int:
        """Timeouts remaining for the team on defense."""
        return (
            self.away_timeouts
            if self.possession_team_id == self.home_team_id
            else self.home_timeouts
        )

    @property
    def possession_momentum(self) -> TeamMomentum:
        """Momentum object of the team currently on offense."""
        return (
            self.home_momentum
            if self.possession_team_id == self.home_team_id
            else self.away_momentum
        )

    @property
    def defense_momentum(self) -> TeamMomentum:
        """Momentum object of the team currently on defense."""
        return (
            self.away_momentum
            if self.possession_team_id == self.home_team_id
            else self.home_momentum
        )


class LiveStateTracker:
    """Stateful engine tracking live in-game plays, drives, and momentum."""

    def __init__(
        self,
        home_team_id: str,
        away_team_id: str,
        initial_possession_team_id: str,
        *,
        game_id: str,
        initial_observed_utc: str,
        initial_ingested_utc: str,
        source: str,
        initial_source_sequence: int = 0,
        momentum_window_drives: int = DEFAULT_MOMENTUM_WINDOW_DRIVES,
        initial_quarter: int = 1,
        initial_clock_seconds: int = QUARTER_SECONDS,
        initial_down: int = 1,
        initial_distance: int = 10,
        initial_yardline: int = 75,  # Standard touchback to 25 (75 yards to goal)
    ) -> None:
        initial_state = GameState(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            possession_team_id=initial_possession_team_id,
            quarter=initial_quarter,
            clock_seconds=initial_clock_seconds,
            down=initial_down,
            distance=initial_distance,
            yardline=initial_yardline,
            game_id=game_id,
            observed_utc=initial_observed_utc,
            ingested_utc=initial_ingested_utc,
            source=source,
            source_sequence=initial_source_sequence,
        )
        if momentum_window_drives < 1 or isinstance(momentum_window_drives, bool):
            raise SchemaError("momentum_window_drives must be a positive integer")

        self.game_id = game_id
        self.source = source
        self.observed_utc = initial_observed_utc
        self.ingested_utc = initial_ingested_utc
        self.source_sequence = initial_source_sequence
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        self.possession_team_id = initial_possession_team_id
        self.momentum_window_drives = momentum_window_drives

        self.quarter = initial_quarter
        self.clock_seconds = initial_clock_seconds
        self.down = initial_down
        self.distance = initial_distance
        self.yardline = initial_yardline

        self.home_score = 0
        self.away_score = 0
        self.home_timeouts = 3
        self.away_timeouts = 3

        self.completed_drives_home: list[LiveDrive] = []
        self.completed_drives_away: list[LiveDrive] = []
        self.active_drive_plays: list[LivePlay] = []
        self.current_drive_id: str = "drive_1"
        self.current_drive_start_quarter: int = initial_quarter
        self.current_drive_start_clock: int = initial_clock_seconds
        self.current_drive_start_yardline: int = initial_yardline

        self.total_plays_observed: int = 0
        self.drive_active = False
        self.seen_play_ids: set[str] = set()
        self.seen_drive_ids: set[str] = set()
        self._event_log: list[dict[str, str | int]] = [
            {
                "event": "init",
                "source_sequence": initial_source_sequence,
                "observed_utc": initial_observed_utc,
            }
        ]

        self.quarter = initial_state.quarter
        self.clock_seconds = initial_state.clock_seconds
        self.down = initial_state.down
        self.distance = initial_state.distance
        self.yardline = initial_state.yardline

    def _validate_observation(self, observed_utc: str, ingested_utc: str, sequence: int) -> None:
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise SchemaError("source_sequence must be a non-negative integer")
        observed = _aware_timestamp(observed_utc, name="observed_utc")
        previous = _aware_timestamp(self.observed_utc, name="previous observed_utc")
        ingested = _aware_timestamp(ingested_utc, name="ingested_utc")
        if ingested < observed:
            raise SchemaError("ingested_utc cannot precede observed_utc")
        if sequence <= self.source_sequence:
            raise SchemaError("source_sequence must increase monotonically")
        if observed < previous:
            raise SchemaError("observed_utc cannot move backward")

    def _validate_game_clock(self, quarter: int, clock_seconds: int) -> None:
        if quarter < self.quarter or (
            quarter == self.quarter and clock_seconds > self.clock_seconds
        ):
            raise SchemaError("Game clock cannot move backward")

    def start_drive(
        self,
        drive_id: str,
        possession_team_id: str,
        quarter: int,
        clock_seconds: int,
        start_yardline: int,
        down: int = 1,
        distance: int = 10,
        *,
        observed_utc: str,
        ingested_utc: str,
        source_sequence: int,
    ) -> None:
        """Start tracking a new offensive drive."""
        if self.drive_active:
            raise SchemaError("Cannot start a new drive while another drive is active")
        if possession_team_id not in (self.home_team_id, self.away_team_id):
            raise SchemaError(f"Unknown team ID {possession_team_id!r}")
        _required_text(drive_id, name="drive_id")
        if drive_id in self.seen_drive_ids:
            raise SchemaError(f"Duplicate drive_id {drive_id!r}")
        if not (0 <= clock_seconds <= QUARTER_SECONDS):
            raise SchemaError(
                f"clock_seconds must be within [0, {QUARTER_SECONDS}], got {clock_seconds}"
            )
        if quarter < 1:
            raise SchemaError(f"quarter must be >= 1, got {quarter}")
        if not (1 <= start_yardline <= 99):
            raise SchemaError(f"start_yardline must be in [1, 99], got {start_yardline}")
        if not (1 <= down <= 4):
            raise SchemaError(f"down must be in [1, 4], got {down}")
        if not (1 <= distance <= 99):
            raise SchemaError(f"distance must be in [1, 99], got {distance}")

        self._validate_observation(observed_utc, ingested_utc, source_sequence)
        self._validate_game_clock(quarter, clock_seconds)

        self.current_drive_id = drive_id
        self.possession_team_id = possession_team_id
        self.current_drive_start_quarter = quarter
        self.current_drive_start_clock = clock_seconds
        self.current_drive_start_yardline = start_yardline
        self.active_drive_plays.clear()

        self.quarter = quarter
        self.clock_seconds = clock_seconds
        self.down = down
        self.distance = distance
        self.yardline = start_yardline
        self.observed_utc = observed_utc
        self.ingested_utc = ingested_utc
        self.source_sequence = source_sequence
        self.drive_active = True
        self.seen_drive_ids.add(drive_id)
        self._event_log.append(
            {
                "event": "start_drive",
                "drive_id": drive_id,
                "source_sequence": source_sequence,
                "observed_utc": observed_utc,
            }
        )

    def record_play(
        self,
        play_id: str,
        yards_gained: float,
        next_down: int,
        next_distance: int,
        next_yardline: int,
        next_quarter: int,
        next_clock_seconds: int,
        *,
        play_type: str = "rush",
        epa: float = 0.0,
        is_success: bool | None = None,
        is_turnover: bool = False,
        is_scoring: bool = False,
        points_scored: int = 0,
        scoring_team_id: str = "",
        observed_utc: str,
        ingested_utc: str,
        source_sequence: int,
    ) -> GameState:
        """Record an in-game play, advancing game clock and down/distance."""
        if not self.drive_active:
            raise SchemaError("Cannot record a play when no drive is active")
        if play_id in self.seen_play_ids:
            raise SchemaError(f"Duplicate play_id {play_id!r}")
        if not (0 <= next_yardline <= 100):
            raise SchemaError(f"next_yardline must be in [0, 100], got {next_yardline}")

        self._validate_observation(observed_utc, ingested_utc, source_sequence)
        self._validate_game_clock(next_quarter, next_clock_seconds)
        play = LivePlay(
            play_id=play_id,
            drive_id=self.current_drive_id,
            team_id=self.possession_team_id,
            quarter=self.quarter,
            clock_seconds=self.clock_seconds,
            down=self.down,
            distance=self.distance,
            yardline=self.yardline,
            yards_gained=yards_gained,
            play_type=play_type,
            epa=epa,
            is_success=is_success,
            is_turnover=is_turnover,
            is_scoring=is_scoring,
            points_scored=points_scored,
            scoring_team_id=scoring_team_id,
            game_id=self.game_id,
            observed_utc=observed_utc,
            ingested_utc=ingested_utc,
            source=self.source,
            source_sequence=source_sequence,
        )
        next_home_score = self.home_score
        next_away_score = self.away_score
        if points_scored > 0:
            if scoring_team_id == self.home_team_id:
                next_home_score += points_scored
            elif scoring_team_id == self.away_team_id:
                next_away_score += points_scored
            else:
                raise SchemaError(
                    f"scoring_team_id {scoring_team_id!r} must be either home "
                    f"({self.home_team_id!r}) or away ({self.away_team_id!r})"
                )

        GameState(
            home_team_id=self.home_team_id,
            away_team_id=self.away_team_id,
            possession_team_id=self.possession_team_id,
            quarter=next_quarter,
            clock_seconds=next_clock_seconds,
            down=next_down,
            distance=next_distance,
            yardline=next_yardline,
            home_score=next_home_score,
            away_score=next_away_score,
            game_id=self.game_id,
            observed_utc=observed_utc,
            ingested_utc=ingested_utc,
            source=self.source,
            source_sequence=source_sequence,
        )

        self.active_drive_plays.append(play)
        self.seen_play_ids.add(play_id)
        self.total_plays_observed += 1
        self.home_score = next_home_score
        self.away_score = next_away_score

        self.quarter = next_quarter
        self.clock_seconds = next_clock_seconds
        self.down = next_down
        self.distance = next_distance
        self.yardline = next_yardline
        self.observed_utc = observed_utc
        self.ingested_utc = ingested_utc
        self.source_sequence = source_sequence
        self._event_log.append(
            {
                "event": "record_play",
                "play_id": play_id,
                "source_sequence": source_sequence,
                "observed_utc": observed_utc,
            }
        )

        return self.get_current_state()

    def end_drive(
        self,
        outcome: str,
        *,
        next_possession_team_id: str | None = None,
        points: int = 0,
        observed_utc: str,
        ingested_utc: str,
        source_sequence: int,
    ) -> LiveDrive:
        """Conclude the active offensive drive and update rolling team momentum."""
        if not self.drive_active:
            raise SchemaError("Cannot end a drive when no drive is active")
        if next_possession_team_id is not None and next_possession_team_id not in (
            self.home_team_id,
            self.away_team_id,
        ):
            raise SchemaError(f"Unknown next possession team {next_possession_team_id!r}")
        try:
            DriveOutcome(outcome)
        except ValueError as exc:
            raise SchemaError(f"Unknown drive outcome {outcome!r}") from exc
        if points < 0 or isinstance(points, bool):
            raise SchemaError("Drive points must be a non-negative integer")
        recorded_points = sum(play.points_scored for play in self.active_drive_plays)
        if points != recorded_points:
            raise SchemaError(
                f"Drive points {points} do not match recorded scoring plays {recorded_points}"
            )
        self._validate_observation(observed_utc, ingested_utc, source_sequence)
        plays_tuple = tuple(self.active_drive_plays)
        drive_epa = sum(p.epa for p in plays_tuple)
        success_count = sum(1 for p in plays_tuple if p.is_success)
        yards_gained = sum(p.yards_gained for p in plays_tuple)

        drive = LiveDrive(
            drive_id=self.current_drive_id,
            team_id=self.possession_team_id,
            start_quarter=self.current_drive_start_quarter,
            start_clock_seconds=self.current_drive_start_clock,
            start_yardline=self.current_drive_start_yardline,
            end_yardline=self.yardline,
            outcome=outcome,
            plays=plays_tuple,
            drive_epa=drive_epa,
            play_count=len(plays_tuple),
            success_count=success_count,
            yards_gained=yards_gained,
            points=points,
        )

        if self.possession_team_id == self.home_team_id:
            self.completed_drives_home.append(drive)
        else:
            self.completed_drives_away.append(drive)

        self.active_drive_plays.clear()
        self.drive_active = False

        if next_possession_team_id:
            self.possession_team_id = next_possession_team_id
        self.observed_utc = observed_utc
        self.ingested_utc = ingested_utc
        self.source_sequence = source_sequence
        self._event_log.append(
            {
                "event": "end_drive",
                "drive_id": self.current_drive_id,
                "outcome": outcome,
                "source_sequence": source_sequence,
                "observed_utc": observed_utc,
            }
        )

        return drive

    def set_score(
        self,
        home_score: int,
        away_score: int,
        *,
        observed_utc: str,
        ingested_utc: str,
        source_sequence: int,
    ) -> None:
        """Explicitly override game score (e.g. sync from external scoreboard)."""
        if (
            home_score < 0
            or away_score < 0
            or isinstance(home_score, bool)
            or isinstance(away_score, bool)
        ):
            raise SchemaError("Scores cannot be negative")
        self._validate_observation(observed_utc, ingested_utc, source_sequence)
        self.home_score = home_score
        self.away_score = away_score
        self.observed_utc = observed_utc
        self.ingested_utc = ingested_utc
        self.source_sequence = source_sequence
        self._event_log.append(
            {
                "event": "set_score",
                "source_sequence": source_sequence,
                "observed_utc": observed_utc,
            }
        )

    def set_timeouts(
        self,
        home_timeouts: int,
        away_timeouts: int,
        *,
        observed_utc: str,
        ingested_utc: str,
        source_sequence: int,
    ) -> None:
        """Update timeouts remaining for both teams."""
        if (
            not (0 <= home_timeouts <= 3)
            or not (0 <= away_timeouts <= 3)
            or isinstance(home_timeouts, bool)
            or isinstance(away_timeouts, bool)
        ):
            raise SchemaError("Timeouts must be between 0 and 3")
        self._validate_observation(observed_utc, ingested_utc, source_sequence)
        self.home_timeouts = home_timeouts
        self.away_timeouts = away_timeouts
        self.observed_utc = observed_utc
        self.ingested_utc = ingested_utc
        self.source_sequence = source_sequence
        self._event_log.append(
            {
                "event": "set_timeouts",
                "source_sequence": source_sequence,
                "observed_utc": observed_utc,
            }
        )

    def compute_momentum(self, team_id: str) -> TeamMomentum:
        """Calculate rolling momentum over the trailing N drives for a team."""
        if team_id not in (self.home_team_id, self.away_team_id):
            raise SchemaError(f"Unknown team ID {team_id!r}")
        drives = (
            self.completed_drives_home
            if team_id == self.home_team_id
            else self.completed_drives_away
        )
        if not drives:
            return TeamMomentum(
                team_id=team_id,
                rolling_drive_epa=0.0,
                rolling_epa_per_play=BASELINE_EPA_PER_PLAY,
                rolling_success_rate=BASELINE_SUCCESS_RATE,
                drives_observed=0,
                plays_observed=0,
                momentum_factor=0.0,
            )

        window = drives[-self.momentum_window_drives :]
        total_plays = sum(d.play_count for d in window)
        total_epa = sum(d.drive_epa for d in window)
        total_success = sum(d.success_count for d in window)

        if total_plays == 0:
            return TeamMomentum(
                team_id=team_id,
                rolling_drive_epa=0.0,
                rolling_epa_per_play=BASELINE_EPA_PER_PLAY,
                rolling_success_rate=BASELINE_SUCCESS_RATE,
                drives_observed=len(window),
                plays_observed=0,
                momentum_factor=0.0,
            )

        epa_per_play = total_epa / total_plays
        success_rate = total_success / total_plays
        mean_drive_epa = total_epa / len(window)

        # Normalized momentum index centered at 0.0:
        # Positive represents above-baseline live offensive rhythm and success.
        raw_factor = 1.2 * (epa_per_play - BASELINE_EPA_PER_PLAY) + 2.0 * (
            success_rate - BASELINE_SUCCESS_RATE
        )
        bounded_factor = max(-2.0, min(2.0, raw_factor))

        return TeamMomentum(
            team_id=team_id,
            rolling_drive_epa=round(mean_drive_epa, 4),
            rolling_epa_per_play=round(epa_per_play, 4),
            rolling_success_rate=round(success_rate, 4),
            drives_observed=len(window),
            plays_observed=total_plays,
            momentum_factor=round(bounded_factor, 4),
        )

    def get_current_state(self) -> GameState:
        """Generate a complete immutable GameState snapshot."""
        return GameState(
            home_team_id=self.home_team_id,
            away_team_id=self.away_team_id,
            possession_team_id=self.possession_team_id,
            quarter=self.quarter,
            clock_seconds=self.clock_seconds,
            down=self.down,
            distance=self.distance,
            yardline=self.yardline,
            home_score=self.home_score,
            away_score=self.away_score,
            home_timeouts=self.home_timeouts,
            away_timeouts=self.away_timeouts,
            home_momentum=self.compute_momentum(self.home_team_id),
            away_momentum=self.compute_momentum(self.away_team_id),
            game_id=self.game_id,
            observed_utc=self.observed_utc,
            ingested_utc=self.ingested_utc,
            source=self.source,
            source_sequence=self.source_sequence,
        )

    def get_event_log(self) -> list[dict[str, str | int]]:
        """Return the immutable audit event log for replay verification."""
        return list(self._event_log)

    def tempo_summary(self) -> dict[str, float]:
        """Compute live tempo statistics (seconds per play, possessions per hour)."""
        state = self.get_current_state()
        elapsed = state.elapsed_game_seconds
        total_possessions = len(self.completed_drives_home) + len(self.completed_drives_away)

        sec_per_play = (
            round(elapsed / self.total_plays_observed, 2) if self.total_plays_observed > 0 else 0.0
        )
        poss_per_60 = (
            round((total_possessions / elapsed) * REGULATION_SECONDS, 2) if elapsed > 0 else 0.0
        )

        return {
            "elapsed_game_seconds": float(elapsed),
            "total_plays": float(self.total_plays_observed),
            "seconds_per_play": sec_per_play,
            "total_possessions_completed": float(total_possessions),
            "possessions_per_60_min": poss_per_60,
        }
