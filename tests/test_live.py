"""Comprehensive unit tests for Live Micro-Markets and in-game state tracking.

Covers:
1. GameState representation, time conversions, differentials, and validation invariants.
2. Play-by-play tracking, drive completion, and rolling drive EPA/success-rate momentum.
3. Analytical Expected Points (EP) model across down, distance, and field position.
4. Real-time in-game win probability model (blowouts, kneel-downs, possession, momentum, OT).
5. Next-drive outcome distribution (TD, FG, Punt, Turnover, Safety) across field positions.
6. Live in-game team totals and dynamic possession tempo projections.
7. Strict prohibition and rejection of player props (team micro-markets only).
"""

from __future__ import annotations

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.features.live import (
    DriveOutcome,
    GameState,
    LivePlay,
    LiveStateTracker,
    TeamMomentum,
    is_successful_play,
)
from cfb_analytics.models.live import (
    calculate_expected_points,
    calculate_win_probability,
    estimate_drive_points_distribution,
    estimate_next_drive_outcomes,
    is_player_prop,
    project_live_spread,
    project_live_team_totals,
    validate_live_market,
    win_probability_to_american_odds,
)

_GameState = GameState
_LivePlay = LivePlay
_LiveStateTracker = LiveStateTracker
_STAMP = "2026-09-07T19:00:00+00:00"


def GameState(*args, **kwargs):
    metadata = {
        "game_id": "test-game",
        "observed_utc": _STAMP,
        "ingested_utc": _STAMP,
        "source": "test-fixture",
        "source_sequence": 0,
    }
    metadata.update(kwargs)
    return _GameState(*args, **metadata)


def LivePlay(*args, **kwargs):
    metadata = {
        "game_id": "test-game",
        "observed_utc": _STAMP,
        "ingested_utc": _STAMP,
        "source": "test-fixture",
        "source_sequence": 0,
    }
    metadata.update(kwargs)
    return _LivePlay(*args, **metadata)


def LiveStateTracker(*args, **kwargs):
    metadata = {
        "game_id": "test-game",
        "initial_observed_utc": _STAMP,
        "initial_ingested_utc": _STAMP,
        "source": "test-fixture",
    }
    metadata.update(kwargs)
    return _LiveStateTracker(*args, **metadata)


# ============================================================================
# 1. State Representation & Schema Validation Tests
# ============================================================================


def test_game_state_valid_initialization() -> None:
    """Proves GameState exposes correct time and differential properties."""
    state = GameState(
        home_team_id="OHIO_STATE",
        away_team_id="MICHIGAN",
        possession_team_id="OHIO_STATE",
        quarter=2,
        clock_seconds=450,  # 7:30 left in Q2
        down=2,
        distance=6,
        yardline=35,
        home_score=14,
        away_score=10,
        home_timeouts=2,
        away_timeouts=3,
    )

    assert state.home_team_id == "OHIO_STATE"
    assert state.away_team_id == "MICHIGAN"
    assert state.possession_team_id == "OHIO_STATE"
    assert state.defense_team_id == "MICHIGAN"
    assert state.quarter == 2
    assert state.clock_seconds == 450

    # Time calculations: Q2 450s left -> 450s in half, 450 + 1800 = 2250s in game
    assert state.seconds_remaining_in_quarter == 450
    assert state.seconds_remaining_in_half == 450
    assert state.seconds_remaining_in_game == 2250
    assert state.elapsed_game_seconds == 1350
    assert not state.is_overtime

    # Score differentials & timeouts
    assert state.home_score_differential == 4
    assert state.score_differential == 4
    assert state.possession_timeouts == 2
    assert state.defense_timeouts == 3


def test_game_state_away_possession_differentials() -> None:
    """Proves differentials and timeouts flip appropriately when away team possesses ball."""
    state = GameState(
        home_team_id="GEORGIA",
        away_team_id="ALABAMA",
        possession_team_id="ALABAMA",
        quarter=3,
        clock_seconds=600,  # 10:00 left in Q3
        down=1,
        distance=10,
        yardline=75,
        home_score=21,
        away_score=14,
        home_timeouts=3,
        away_timeouts=1,
    )

    assert state.defense_team_id == "GEORGIA"
    assert state.home_score_differential == 7
    # Away is trailing by 7, so possession score diff is -7
    assert state.score_differential == -7
    assert state.possession_timeouts == 1
    assert state.defense_timeouts == 3

    # Half: Q3 has 600 + 900 = 1500s remaining in half 2
    assert state.seconds_remaining_in_half == 1500
    # Game: Q3 has 600 + 900 = 1500s remaining in game
    assert state.seconds_remaining_in_game == 1500


def test_game_state_time_calculations_all_quarters() -> None:
    """Proves seconds remaining across Q1, Q2, Q3, Q4, and OT."""
    q1 = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=1,
        clock_seconds=300,
        down=1,
        distance=10,
        yardline=50,
    )
    assert q1.seconds_remaining_in_quarter == 300
    assert q1.seconds_remaining_in_half == 1200  # 300 + 900
    assert q1.seconds_remaining_in_game == 3000  # 300 + 2700

    q4 = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=4,
        clock_seconds=120,
        down=1,
        distance=10,
        yardline=50,
    )
    assert q4.seconds_remaining_in_quarter == 120
    assert q4.seconds_remaining_in_half == 120
    assert q4.seconds_remaining_in_game == 120

    ot = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=5,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=25,
    )
    assert ot.is_overtime
    assert ot.seconds_remaining_in_game == 0
    assert ot.seconds_remaining_in_half == 0


def test_game_state_validation_guards() -> None:
    """Proves SchemaError is raised on out-of-bounds inputs."""
    # Invalid quarter
    with pytest.raises(SchemaError, match="Quarter must be >= 1"):
        GameState("H", "A", "H", quarter=0, clock_seconds=500, down=1, distance=10, yardline=50)

    # Invalid clock seconds
    with pytest.raises(SchemaError, match="clock_seconds must be within"):
        GameState("H", "A", "H", quarter=1, clock_seconds=901, down=1, distance=10, yardline=50)
    with pytest.raises(SchemaError, match="clock_seconds must be within"):
        GameState("H", "A", "H", quarter=1, clock_seconds=-5, down=1, distance=10, yardline=50)

    # Invalid down
    with pytest.raises(SchemaError, match="Down must be in"):
        GameState("H", "A", "H", quarter=1, clock_seconds=500, down=0, distance=10, yardline=50)
    with pytest.raises(SchemaError, match="Down must be in"):
        GameState("H", "A", "H", quarter=1, clock_seconds=500, down=5, distance=10, yardline=50)

    # Invalid distance
    with pytest.raises(SchemaError, match="Distance must be in"):
        GameState("H", "A", "H", quarter=1, clock_seconds=500, down=1, distance=0, yardline=50)
    with pytest.raises(SchemaError, match="Distance must be in"):
        GameState("H", "A", "H", quarter=1, clock_seconds=500, down=1, distance=100, yardline=50)

    # Invalid yardline
    with pytest.raises(SchemaError, match="Yardline must be in"):
        GameState("H", "A", "H", quarter=1, clock_seconds=500, down=1, distance=10, yardline=-1)
    with pytest.raises(SchemaError, match="Yardline must be in"):
        GameState("H", "A", "H", quarter=1, clock_seconds=500, down=1, distance=10, yardline=101)

    # Negative scores
    with pytest.raises(SchemaError, match="Scores must be non-negative"):
        GameState(
            "H",
            "A",
            "H",
            quarter=1,
            clock_seconds=500,
            down=1,
            distance=10,
            yardline=50,
            home_score=-1,
        )

    # Invalid timeouts
    with pytest.raises(SchemaError, match="Timeouts must be between 0 and 3"):
        GameState(
            "H",
            "A",
            "H",
            quarter=1,
            clock_seconds=500,
            down=1,
            distance=10,
            yardline=50,
            home_timeouts=4,
        )

    # Unknown possession team
    with pytest.raises(SchemaError, match="Possession team 'NOT_A_TEAM' must be either home"):
        GameState(
            "H", "A", "NOT_A_TEAM", quarter=1, clock_seconds=500, down=1, distance=10, yardline=50
        )


# ============================================================================
# 2. Play-by-Play Tracking & Momentum Features Tests
# ============================================================================


def test_is_successful_play_logic() -> None:
    """Proves football analytics play success criteria across downs."""
    # 1st down: needs >= 50%
    assert is_successful_play(down=1, distance=10, yards_gained=5.0) is True
    assert is_successful_play(down=1, distance=10, yards_gained=4.5) is False

    # 2nd down: needs >= 70%
    assert is_successful_play(down=2, distance=10, yards_gained=7.0) is True
    assert is_successful_play(down=2, distance=10, yards_gained=6.5) is False

    # 3rd down: needs >= 100% (conversion)
    assert is_successful_play(down=3, distance=4, yards_gained=4.0) is True
    assert is_successful_play(down=3, distance=4, yards_gained=3.5) is False

    # 4th down: needs >= 100% (conversion)
    assert is_successful_play(down=4, distance=1, yards_gained=1.0) is True
    assert is_successful_play(down=4, distance=1, yards_gained=0.5) is False


def test_live_play_auto_computes_success() -> None:
    """Proves LivePlay determines success automatically if not explicitly given."""
    p1 = LivePlay(
        play_id="p1",
        drive_id="d1",
        team_id="TEXAS",
        quarter=1,
        clock_seconds=800,
        down=1,
        distance=10,
        yardline=75,
        yards_gained=6.0,
        epa=0.35,
    )
    assert p1.is_success is True

    p2 = LivePlay(
        play_id="p2",
        drive_id="d1",
        team_id="TEXAS",
        quarter=1,
        clock_seconds=760,
        down=2,
        distance=4,
        yardline=69,
        yards_gained=1.0,
        epa=-0.25,
    )
    assert p2.is_success is False  # 1.0 < 0.7 * 4 = 2.8


def test_live_state_tracker_drives_and_momentum() -> None:
    """Proves LiveStateTracker accumulates plays, updates momentum, and tracks tempo."""
    tracker = LiveStateTracker(
        home_team_id="TEXAS",
        away_team_id="OKLAHOMA",
        initial_possession_team_id="TEXAS",
        momentum_window_drives=3,
        initial_quarter=1,
        initial_clock_seconds=900,
        initial_yardline=75,
    )

    initial_state = tracker.get_current_state()
    assert initial_state.possession_team_id == "TEXAS"
    assert initial_state.yardline == 75
    assert initial_state.home_momentum.drives_observed == 0

    # Drive 1 for Texas: 3 plays resulting in a Touchdown
    tracker.start_drive(
        "d1",
        "TEXAS",
        quarter=1,
        clock_seconds=900,
        start_yardline=75,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=1,
    )

    # Play 1: 1st & 10 at 75 -> 25 yd pass, success
    tracker.record_play(
        "p1",
        yards_gained=25.0,
        next_down=1,
        next_distance=10,
        next_yardline=50,
        next_quarter=1,
        next_clock_seconds=860,
        play_type="pass",
        epa=1.2,
        is_success=True,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=2,
    )

    # Play 2: 1st & 10 at 50 -> 30 yd rush, success
    tracker.record_play(
        "p2",
        yards_gained=30.0,
        next_down=1,
        next_distance=10,
        next_yardline=20,
        next_quarter=1,
        next_clock_seconds=820,
        play_type="rush",
        epa=1.4,
        is_success=True,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=3,
    )

    # Play 3: 1st & 10 at 20 -> 20 yd TD pass, success
    tracker.record_play(
        "p3",
        yards_gained=20.0,
        next_down=1,
        next_distance=10,
        next_yardline=0,
        next_quarter=1,
        next_clock_seconds=780,
        play_type="pass",
        epa=2.0,
        is_success=True,
        is_scoring=True,
        points_scored=7,
        scoring_team_id="TEXAS",
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=4,
    )

    d1 = tracker.end_drive(
        outcome=DriveOutcome.TOUCHDOWN.value,
        next_possession_team_id="OKLAHOMA",
        points=7,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=5,
    )

    assert d1.outcome == "TD"
    assert d1.play_count == 3
    assert d1.success_count == 3
    assert d1.success_rate == 1.0
    assert d1.drive_epa == 4.6
    assert d1.epa_per_play == pytest.approx(4.6 / 3, 0.01)

    # State check after drive 1
    state_after_d1 = tracker.get_current_state()
    assert state_after_d1.home_score == 7
    assert state_after_d1.away_score == 0
    assert state_after_d1.possession_team_id == "OKLAHOMA"

    # Texas momentum should now be strongly positive
    texas_mom = state_after_d1.home_momentum
    assert texas_mom.drives_observed == 1
    assert texas_mom.plays_observed == 3
    assert texas_mom.rolling_success_rate == 1.0
    assert texas_mom.momentum_factor > 1.0

    # Oklahoma momentum is still default
    okla_mom = state_after_d1.away_momentum
    assert okla_mom.drives_observed == 0
    assert okla_mom.momentum_factor == 0.0

    # Now simulate Oklahoma 3-and-out
    tracker.start_drive(
        "d2",
        "OKLAHOMA",
        quarter=1,
        clock_seconds=780,
        start_yardline=75,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=6,
    )
    tracker.record_play(
        "p4",
        yards_gained=1.0,
        next_down=2,
        next_distance=9,
        next_yardline=74,
        next_quarter=1,
        next_clock_seconds=740,
        epa=-0.5,
        is_success=False,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=7,
    )
    tracker.record_play(
        "p5",
        yards_gained=2.0,
        next_down=3,
        next_distance=7,
        next_yardline=72,
        next_quarter=1,
        next_clock_seconds=700,
        epa=-0.6,
        is_success=False,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=8,
    )
    tracker.record_play(
        "p6",
        yards_gained=0.0,
        next_down=4,
        next_distance=7,
        next_yardline=72,
        next_quarter=1,
        next_clock_seconds=660,
        epa=-1.2,
        is_success=False,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=9,
    )
    tracker.end_drive(
        DriveOutcome.PUNT.value,
        next_possession_team_id="TEXAS",
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=10,
    )

    state_after_d2 = tracker.get_current_state()
    assert state_after_d2.away_momentum.drives_observed == 1
    assert state_after_d2.away_momentum.rolling_success_rate == 0.0
    assert state_after_d2.away_momentum.momentum_factor < 0.0

    # Tempo stats verification
    tempo = tracker.tempo_summary()
    assert tempo["total_plays"] == 6.0
    assert tempo["total_possessions_completed"] == 2.0
    assert tempo["elapsed_game_seconds"] == 240.0  # 900 - 660


# ============================================================================
# 3. Expected Points (EP) Scrimmage Model Tests
# ============================================================================


def test_expected_points_monotonic_field_progression() -> None:
    """Proves Expected Points increases monotonically as team nears opponent endzone on 1st & 10."""
    ep_at_1 = calculate_expected_points(down=1, distance=10, yardline=1)
    ep_at_10 = calculate_expected_points(down=1, distance=10, yardline=10)
    ep_at_25 = calculate_expected_points(down=1, distance=10, yardline=25)
    ep_at_50 = calculate_expected_points(down=1, distance=10, yardline=50)
    ep_at_75 = calculate_expected_points(down=1, distance=10, yardline=75)
    ep_at_90 = calculate_expected_points(down=1, distance=10, yardline=90)
    ep_at_99 = calculate_expected_points(down=1, distance=10, yardline=99)

    # Goal line to backed up ordering
    assert ep_at_1 > ep_at_10 > ep_at_25 > ep_at_50 > ep_at_75 > ep_at_90 > ep_at_99
    # Values reflect realistic CFB Expected Points
    assert ep_at_1 > 5.5  # Goal-line 1st down ~6 pts
    assert 1.0 < ep_at_75 < 2.5  # Touchback at 25 ~1.6 pts
    assert ep_at_99 < 0.0  # Backed up inside 1 is negative (safety / short field risk)


def test_expected_points_down_penalty() -> None:
    """Proves Expected Points decreases as down increases with identical distance and yardline."""
    ep_1st = calculate_expected_points(down=1, distance=10, yardline=40)
    ep_2nd = calculate_expected_points(down=2, distance=10, yardline=40)
    ep_3rd = calculate_expected_points(down=3, distance=10, yardline=40)
    ep_4th = calculate_expected_points(down=4, distance=10, yardline=40)

    assert ep_1st > ep_2nd > ep_3rd > ep_4th


def test_expected_points_4th_down_decisions() -> None:
    """Proves 4th down EP reflects FG range vs punt territory vs go-for-it short yardage."""
    # 4th & 2 at opponent 15 (easy FG territory) -> high positive EP
    ep_fg = calculate_expected_points(down=4, distance=2, yardline=15)
    assert ep_fg > 1.8

    # 4th & 10 at own 20 (yardline 80) -> negative EP (punt gives ball to opponent)
    ep_punt = calculate_expected_points(down=4, distance=10, yardline=80)
    assert ep_punt < 0.0

    # 4th & 1 conversion attempt should beat long distance on 4th down
    ep_short = calculate_expected_points(down=4, distance=1, yardline=45)
    ep_long = calculate_expected_points(down=4, distance=10, yardline=45)
    assert ep_short > ep_long


# ============================================================================
# 4. Real-time Win Probability Model Tests
# ============================================================================


def test_win_probability_kickoff_neutral_teams() -> None:
    """Proves two equal teams tied at kickoff produce ~50% win probability."""
    state = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=1,
        clock_seconds=900,
        down=1,
        distance=10,
        yardline=75,
        home_score=0,
        away_score=0,
    )
    wp_home = calculate_win_probability(state, pregame_home_margin=0.0, team_id="H")
    wp_away = calculate_win_probability(state, pregame_home_margin=0.0, team_id="A")

    # Slight possession edge at kickoff (~52-54%)
    assert 0.50 <= wp_home <= 0.56
    assert wp_home + wp_away == pytest.approx(1.0, abs=1e-3)


def test_win_probability_pregame_favorite_influence() -> None:
    """Proves pregame expected margin influences win probability even when tied."""
    state = GameState(
        home_team_id="FAV",
        away_team_id="DOG",
        possession_team_id="FAV",
        quarter=1,
        clock_seconds=900,
        down=1,
        distance=10,
        yardline=75,
        home_score=0,
        away_score=0,
    )
    # Heavy favorite by 17 points
    wp_fav = calculate_win_probability(state, pregame_home_margin=17.0, team_id="FAV")
    assert wp_fav > 0.80

    # Large underdog (-17 points)
    wp_dog = calculate_win_probability(state, pregame_home_margin=-17.0, team_id="FAV")
    assert wp_dog < 0.20


def test_win_probability_blowout_late() -> None:
    """Proves large lead late in game approaches 1.0 for leader and 0.0 for trailer."""
    # Home team up 28 with 3:00 left in Q4
    state = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=4,
        clock_seconds=180,
        down=1,
        distance=10,
        yardline=50,
        home_score=38,
        away_score=10,
    )
    wp_home = calculate_win_probability(state, pregame_home_margin=0.0, team_id="H")
    wp_away = calculate_win_probability(state, pregame_home_margin=0.0, team_id="A")

    assert wp_home >= 0.999
    assert wp_away <= 0.001


def test_win_probability_possession_and_field_position_impact() -> None:
    """Proves possession and red zone position increase win probability in close game."""
    # Tied with 4:00 left in Q4, Home has ball at opponent 5-yard line (yardline=5)
    state_redzone = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=4,
        clock_seconds=240,
        down=1,
        distance=5,
        yardline=5,
        home_score=24,
        away_score=24,
    )
    # Tied with 4:00 left in Q4, Home backed up at own 5-yard line (yardline=95)
    state_backedup = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=4,
        clock_seconds=240,
        down=1,
        distance=10,
        yardline=95,
        home_score=24,
        away_score=24,
    )

    wp_rz = calculate_win_probability(state_redzone, pregame_home_margin=0.0, team_id="H")
    wp_bu = calculate_win_probability(state_backedup, pregame_home_margin=0.0, team_id="H")

    assert wp_rz > 0.80  # 1st & goal tied late is a huge advantage
    assert wp_bu < wp_rz
    assert wp_bu < 0.50  # Backed up at own 5 with ball late has safety and turnover risk


def test_win_probability_kneel_down_rule() -> None:
    """Proves win probability hits 0.9999 when leading team can kneel out remaining clock."""
    # Up by 3, 1st & 10 with 45 seconds left, opponent has 0 timeouts
    # Offense can kneel 2 times for 80 seconds > 45 seconds
    state = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=4,
        clock_seconds=45,
        down=1,
        distance=10,
        yardline=40,
        home_score=24,
        away_score=21,
        home_timeouts=3,
        away_timeouts=0,
    )
    wp = calculate_win_probability(state, pregame_home_margin=0.0, team_id="H")
    assert wp == 0.9999

    # Away team leading and kneeling out
    state_away_kneel = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="A",
        quarter=4,
        clock_seconds=30,
        down=2,
        distance=12,
        yardline=50,
        home_score=21,
        away_score=24,
        home_timeouts=0,
        away_timeouts=2,
    )
    wp_away = calculate_win_probability(state_away_kneel, pregame_home_margin=0.0, team_id="A")
    assert wp_away == 0.9999


def test_win_probability_clock_expired_and_overtime() -> None:
    """Proves win probability at clock expiration and in overtime."""
    # Regulation ended, home won
    final_home = GameState(
        "H",
        "A",
        "H",
        quarter=4,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=50,
        home_score=27,
        away_score=24,
        is_final=True,
    )
    assert calculate_win_probability(final_home, team_id="H") == 1.0
    assert calculate_win_probability(final_home, team_id="A") == 0.0

    # Overtime is refused without possession-series state.
    ot_tied = GameState(
        "H",
        "A",
        "H",
        quarter=5,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=25,
        home_score=24,
        away_score=24,
    )
    with pytest.raises(SchemaError, match="Unresolved overtime"):
        calculate_win_probability(ot_tied, pregame_home_margin=3.0, team_id="H")


def test_win_probability_momentum_boost() -> None:
    """Proves positive in-game rolling momentum increases win probability."""
    base_state = GameState(
        "H",
        "A",
        "H",
        quarter=3,
        clock_seconds=600,
        down=1,
        distance=10,
        yardline=50,
        home_score=14,
        away_score=14,
    )
    wp_base = calculate_win_probability(base_state, pregame_home_margin=0.0, team_id="H")

    hot_momentum = TeamMomentum(team_id="H", rolling_success_rate=0.75, momentum_factor=1.5)
    momentum_state = GameState(
        "H",
        "A",
        "H",
        quarter=3,
        clock_seconds=600,
        down=1,
        distance=10,
        yardline=50,
        home_score=14,
        away_score=14,
        home_momentum=hot_momentum,
    )
    wp_momentum = calculate_win_probability(momentum_state, pregame_home_margin=0.0, team_id="H")

    assert wp_momentum > wp_base


# ============================================================================
# 5. Next Drive Outcome Estimation Tests
# ============================================================================


def test_drive_outcome_distribution_sums_to_one() -> None:
    """Proves drive outcome probabilities sum to 1.0 across various yardlines."""
    for y in (1, 10, 25, 50, 75, 90, 99):
        dist = estimate_next_drive_outcomes(start_yardline=y)
        total_p = dist.touchdown + dist.field_goal + dist.punt + dist.turnover_downs + dist.safety
        assert total_p == pytest.approx(1.0, abs=1e-3)
        assert dist.touchdown > 0.0
        assert dist.field_goal > 0.0
        assert dist.punt > 0.0
        assert dist.turnover_downs > 0.0
        assert dist.safety >= 0.0


def test_drive_outcome_goal_to_go_vs_backed_up() -> None:
    """Proves field position shifts TD, punt, and safety probabilities."""
    goal_to_go = estimate_next_drive_outcomes(start_yardline=5)
    backed_up = estimate_next_drive_outcomes(start_yardline=99)

    # Goal to go: TD is plurality/majority, punt is practically zero
    assert goal_to_go.touchdown > 0.55
    assert goal_to_go.punt < 0.01
    assert goal_to_go.safety < 0.001
    assert goal_to_go.expected_points > 4.5

    # Backed up: Punt dominates, safety risk is non-zero
    assert backed_up.punt > 0.55
    assert backed_up.safety > 0.015  # At own 1-yard line, safety is a genuine risk
    assert backed_up.touchdown < 0.20
    assert backed_up.expected_points < 2.0


def test_drive_outcome_efficiency_and_momentum_shift() -> None:
    """Proves offensive efficiency advantage shifts mass to scoring outcomes (TD/FG)."""
    neutral = estimate_next_drive_outcomes(start_yardline=50)
    elite_offense = estimate_next_drive_outcomes(
        start_yardline=50, off_efficiency=0.8, def_efficiency=-0.5, momentum=1.0
    )
    elite_defense = estimate_next_drive_outcomes(
        start_yardline=50, off_efficiency=-0.5, def_efficiency=0.8, momentum=-1.0
    )

    # Elite offense: higher TD and FG, fewer punts, higher expected points
    assert elite_offense.touchdown > neutral.touchdown
    assert elite_offense.punt < neutral.punt
    assert elite_offense.expected_points > neutral.expected_points

    # Elite defense: lower TD, more punts and turnovers
    assert elite_defense.touchdown < neutral.touchdown
    assert elite_defense.punt > neutral.punt
    assert elite_defense.expected_points < neutral.expected_points


def test_drive_outcome_as_dict_property() -> None:
    """Proves as_dict returns correct keys mapping to DriveOutcome enum."""
    dist = estimate_next_drive_outcomes(start_yardline=75)
    d = dist.as_dict

    assert DriveOutcome.TOUCHDOWN.value in d
    assert DriveOutcome.FIELD_GOAL.value in d
    assert DriveOutcome.PUNT.value in d
    assert DriveOutcome.TURNOVER.value in d
    assert DriveOutcome.SAFETY.value in d
    assert sum(d.values()) == pytest.approx(1.0, abs=1e-3)


# ============================================================================
# 6. Live In-Game Team Totals Tests
# ============================================================================


def test_project_live_team_totals_active_game() -> None:
    """Proves live totals project possessions, points, and line probabilities."""
    state = GameState(
        home_team_id="CLEMSON",
        away_team_id="FSU",
        possession_team_id="CLEMSON",
        quarter=3,
        clock_seconds=900,  # Halftime / Start of Q3 (1800s remaining)
        down=1,
        distance=10,
        yardline=75,
        home_score=17,
        away_score=14,
    )

    proj = project_live_team_totals(
        state,
        pregame_pace=12.0,
        off_eff_home=0.3,
        off_eff_away=0.1,
        lines_home=(27.5, 31.5),
        lines_away=(24.5,),
        lines_game=(52.5, 56.5),
    )

    # 1800 seconds remaining = half of regulation -> approx 6 remaining possessions
    assert proj.remaining_game_seconds == 1800
    assert 6.0 <= proj.home.remaining_possessions <= 8.0
    assert proj.home.projected_total > 17.0
    assert proj.away.projected_total > 14.0
    assert proj.projected_game_total == pytest.approx(
        proj.home.projected_total + proj.away.projected_total, abs=0.01
    )

    # Check Over/Under probabilities exist and sum to 1.0
    assert 27.5 in proj.home.prob_over
    assert 27.5 in proj.home.prob_under
    assert proj.home.prob_over[27.5] + proj.home.prob_under[27.5] == pytest.approx(1.0, abs=1e-3)

    assert 52.5 in proj.prob_game_over
    assert 52.5 in proj.prob_game_under
    assert proj.prob_game_over[52.5] + proj.prob_game_under[52.5] == pytest.approx(1.0, abs=1e-3)


def test_project_live_team_totals_game_completed() -> None:
    """Proves projected totals equal final scores when regulation clock has expired."""
    final_state = GameState(
        home_team_id="CLEMSON",
        away_team_id="FSU",
        possession_team_id="CLEMSON",
        quarter=4,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=50,
        home_score=31,
        away_score=28,
        is_final=True,
    )

    proj = project_live_team_totals(
        final_state,
        lines_home=(30.5,),
        lines_away=(30.5,),
        lines_game=(58.5,),
    )

    assert proj.remaining_game_seconds == 0
    assert proj.home.remaining_possessions == 0.0
    assert proj.home.remaining_expected_points == 0.0
    assert proj.home.projected_total == 31.0
    assert proj.away.projected_total == 28.0
    assert proj.projected_game_total == 59.0

    # Probabilities should be deterministic at game end
    assert proj.home.prob_over[30.5] == 1.0  # 31 > 30.5
    assert proj.home.prob_under[30.5] == 0.0
    assert proj.away.prob_over[30.5] == 0.0  # 28 < 30.5
    assert proj.away.prob_under[30.5] == 1.0
    assert proj.prob_game_over[58.5] == 1.0  # 59 > 58.5


def test_project_live_team_totals_tempo_adaptation() -> None:
    """Proves fast pace increases projected possessions and points relative to slow pace."""
    state = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=2,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=75,
        home_score=14,
        away_score=14,
    )  # Halftime (1800s elapsed, 1800s remaining)

    # Fast game: 16 possessions completed in first half (8 each)
    proj_fast = project_live_team_totals(
        state, pregame_pace=12.0, observed_possessions_home=8, observed_possessions_away=8
    )

    # Slow game: 8 possessions completed in first half (4 each)
    proj_slow = project_live_team_totals(
        state, pregame_pace=12.0, observed_possessions_home=4, observed_possessions_away=4
    )

    assert proj_fast.blended_tempo_poss_per_60 > proj_slow.blended_tempo_poss_per_60
    assert proj_fast.home.remaining_possessions > proj_slow.home.remaining_possessions
    assert proj_fast.projected_game_total > proj_slow.projected_game_total


# ============================================================================
# 7. Invariants & Strict Player Prop Prohibition Tests
# ============================================================================


def test_is_player_prop_identification() -> None:
    """Proves is_player_prop detects player props and admits team micro-markets."""
    # Player prop indicators
    assert is_player_prop("player_passing_yards") is True
    assert is_player_prop("player_touchdowns") is True
    assert is_player_prop("anytime_touchdown") is True
    assert is_player_prop("first_td_scorer") is True
    assert is_player_prop("passer_rating") is True
    assert is_player_prop("rusher_yards") is True
    assert is_player_prop("receiver_receptions") is True
    assert is_player_prop("pass_yards") is True
    assert is_player_prop("rush_attempts") is True

    # Team micro-market indicators
    assert is_player_prop("team_total") is False
    assert is_player_prop("team_total_points") is False
    assert is_player_prop("live_team_total") is False
    assert is_player_prop("live_win_probability") is False
    assert is_player_prop("next_drive_outcome") is False


def test_validate_live_market_rejects_player_props() -> None:
    """Proves validate_live_market raises SchemaError on any player prop."""
    with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
        validate_live_market("player_passing_yards")

    with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
        validate_live_market("anytime_touchdown")

    with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
        validate_live_market("live_win_probability", market_type="PLAYER_PROP")


def test_validate_live_market_whitelists_supported_markets() -> None:
    """Proves supported live micro-markets and aliases validate and normalize cleanly."""
    assert validate_live_market("live_win_probability") == "live_win_probability"
    assert validate_live_market("win_prob") == "live_win_probability"
    assert validate_live_market("ml") == "live_moneyline"
    assert validate_live_market("next_drive_outcome") == "next_drive_outcome"
    assert validate_live_market("drive_outcome") == "next_drive_outcome"
    assert validate_live_market("team_total") == "live_team_total"
    assert validate_live_market("game_total") == "live_game_total"
    assert validate_live_market("spread") == "live_spread"

    # Unsupported market
    with pytest.raises(SchemaError, match="Unsupported live market"):
        validate_live_market("total_corners")


# ============================================================================
# 8. Extended Edge Cases & Robustness Tests
# ============================================================================


def test_game_state_away_score_differential() -> None:
    """Proves away_score_differential correctly computes away_score - home_score."""
    state = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=2,
        clock_seconds=500,
        down=1,
        distance=10,
        yardline=50,
        home_score=21,
        away_score=14,
    )
    assert state.home_score_differential == 7
    assert state.away_score_differential == -7


def test_live_play_turnover_not_successful() -> None:
    """Proves plays resulting in turnovers are never marked successful."""
    p_turnover = LivePlay(
        play_id="to_1",
        drive_id="d1",
        team_id="A",
        quarter=1,
        clock_seconds=500,
        down=1,
        distance=10,
        yardline=50,
        yards_gained=8.0,  # > 5.0, but fumbled/intercepted
        is_turnover=True,
    )
    assert p_turnover.is_success is False


def test_live_state_tracker_preserves_goal_line_transition() -> None:
    """Proves a terminal scoring play can preserve its natural goal-line coordinate."""
    tracker = LiveStateTracker(
        home_team_id="H",
        away_team_id="A",
        initial_possession_team_id="H",
        initial_yardline=20,
    )
    tracker.start_drive(
        "d1",
        "H",
        quarter=1,
        clock_seconds=900,
        start_yardline=20,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=1,
    )
    # 20-yard TD play advancing yardline to 0
    state = tracker.record_play(
        "p_td",
        yards_gained=20.0,
        next_down=1,
        next_distance=10,
        next_yardline=0,
        next_quarter=1,
        next_clock_seconds=600,
        is_scoring=True,
        points_scored=6,
        scoring_team_id="H",
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=2,
    )
    assert state.yardline == 0
    assert tracker.yardline == 0

    drive = tracker.end_drive(
        DriveOutcome.TOUCHDOWN.value,
        points=6,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=3,
    )
    assert drive.end_yardline == 0


def test_live_state_tracker_start_drive_validation_guards() -> None:
    """Proves start_drive raises SchemaError on invalid inputs."""
    tracker = LiveStateTracker("H", "A", "H")
    observation = {
        "observed_utc": _STAMP,
        "ingested_utc": _STAMP,
        "source_sequence": 1,
    }
    with pytest.raises(SchemaError, match="Unknown team ID"):
        tracker.start_drive(
            "d1",
            "BAD_TEAM",
            quarter=1,
            clock_seconds=900,
            start_yardline=75,
            **observation,
        )
    with pytest.raises(SchemaError, match="start_yardline must be in"):
        tracker.start_drive(
            "d1",
            "H",
            quarter=1,
            clock_seconds=900,
            start_yardline=-1,
            **observation,
        )
    with pytest.raises(SchemaError, match="start_yardline must be in"):
        tracker.start_drive(
            "d1",
            "H",
            quarter=1,
            clock_seconds=900,
            start_yardline=101,
            **observation,
        )
    with pytest.raises(SchemaError, match="clock_seconds must be within"):
        tracker.start_drive(
            "d1",
            "H",
            quarter=1,
            clock_seconds=901,
            start_yardline=75,
            **observation,
        )
    with pytest.raises(SchemaError, match="quarter must be >= 1"):
        tracker.start_drive(
            "d1",
            "H",
            quarter=0,
            clock_seconds=900,
            start_yardline=75,
            **observation,
        )
    with pytest.raises(SchemaError, match="down must be in"):
        tracker.start_drive(
            "d1",
            "H",
            quarter=1,
            clock_seconds=900,
            start_yardline=75,
            down=5,
            **observation,
        )
    with pytest.raises(SchemaError, match="distance must be in"):
        tracker.start_drive(
            "d1",
            "H",
            quarter=1,
            clock_seconds=900,
            start_yardline=75,
            distance=0,
            **observation,
        )


def test_win_probability_refuses_unresolved_overtime_state() -> None:
    """Proves an incomplete overtime state cannot emit false probability certainty."""
    state_reg = GameState(
        "H",
        "A",
        "H",
        quarter=4,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=50,
        home_score=24,
        away_score=24,
    )
    with pytest.raises(SchemaError, match="requires is_final=True"):
        calculate_win_probability(state_reg, pregame_home_margin=40.0, team_id="H")

    state_ot = GameState(
        "H",
        "A",
        "H",
        quarter=5,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=25,
        home_score=24,
        away_score=24,
    )
    with pytest.raises(SchemaError, match="Unresolved overtime"):
        calculate_win_probability(state_ot, pregame_home_margin=40.0, team_id="H")


def test_win_probability_to_american_odds() -> None:
    """Proves win_probability_to_american_odds maps probabilities to American prices."""
    assert win_probability_to_american_odds(0.50) in (100, -100)
    assert win_probability_to_american_odds(0.80) == -400
    assert win_probability_to_american_odds(0.25) == 300
    assert win_probability_to_american_odds(0.0) is None
    assert win_probability_to_american_odds(1.0) is None
    assert win_probability_to_american_odds(-0.1) is None
    assert win_probability_to_american_odds(1.2) is None


def test_project_live_spread() -> None:
    """Proves project_live_spread computes projected spread and cover probabilities."""
    state = GameState(
        "H",
        "A",
        "H",
        quarter=3,
        clock_seconds=450,
        down=1,
        distance=10,
        yardline=50,
        home_score=21,
        away_score=14,
    )
    spread_res = project_live_spread(state, pregame_home_margin=3.5, lines=(-7.5, -3.5, 3.5))

    assert "projected_home_margin" in spread_res
    assert "projected_home_spread" in spread_res
    assert spread_res["projected_home_margin"] > 0
    assert spread_res["projected_home_spread"] == pytest.approx(
        -spread_res["projected_home_margin"], abs=0.01
    )

    covers = spread_res["prob_cover"]
    assert covers[-7.5] < covers[3.5]
    for p in covers.values():
        assert 0.0 < p < 1.0

    # Finished game determinism
    final_state = GameState(
        "H",
        "A",
        "H",
        quarter=4,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=50,
        home_score=28,
        away_score=24,
        is_final=True,
    )
    final_spread = project_live_spread(final_state, lines=(-3.5, -4.5))
    assert final_spread["prob_cover"][-3.5] == 1.0  # margin +4 covers -3.5 (4 - 3.5 = +0.5 > 0)
    assert final_spread["prob_cover"][-4.5] == 0.0  # margin +4 fails -4.5 (4 - 4.5 = -0.5 < 0)


def test_estimate_drive_points_distribution() -> None:
    """Proves estimate_drive_points_distribution partitions probabilities across discrete scores."""
    dist = estimate_drive_points_distribution(start_yardline=25)
    assert 7 in dist
    assert 3 in dist
    assert 0 in dist
    assert 8 in dist
    assert 6 in dist
    assert -2 in dist

    total_p = sum(dist.values())
    assert total_p == pytest.approx(1.0, abs=1e-3)
    rz_dist = estimate_drive_points_distribution(start_yardline=10)
    bu_dist = estimate_drive_points_distribution(start_yardline=95)
    assert rz_dist[7] > bu_dist[7]
    assert bu_dist[0] > rz_dist[0]


def test_drive_outcome_distribution_property_aliases() -> None:
    """Proves turnover, downs, and turnover_or_downs property accessors work."""
    dist = estimate_next_drive_outcomes(start_yardline=50)
    assert dist.turnover == dist.turnover_downs
    assert dist.downs == dist.turnover_downs
    assert dist.turnover_or_downs == dist.turnover_downs


def test_is_player_prop_extended_catalog() -> None:
    """Proves is_player_prop catches passing, rushing, receiving, interceptions, and sacks."""
    assert is_player_prop("passing_yards") is True
    assert is_player_prop("rushing_yards") is True
    assert is_player_prop("receiving_yards") is True
    assert is_player_prop("receptions") is True
    assert is_player_prop("interceptions") is True
    assert is_player_prop("sacks") is True
    assert is_player_prop("longest_rush") is True
    assert is_player_prop("field_goals_made") is True

    # Team markets remain allowed
    assert is_player_prop("game_total") is False
    assert is_player_prop("live_game_total") is False
    assert is_player_prop("team_total") is False
    assert is_player_prop("live_team_total") is False


def test_project_live_team_totals_refuses_unresolved_overtime() -> None:
    """Proves totals refuse overtime without possession-series state."""
    ot_state = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=5,
        clock_seconds=0,
        down=1,
        distance=10,
        yardline=25,
        home_score=24,
        away_score=24,
    )
    with pytest.raises(SchemaError, match="Unresolved overtime"):
        project_live_team_totals(ot_state, lines_home=(27.5,), lines_away=(27.5,))


def test_game_state_with_provenance_metadata() -> None:
    """Proves GameState accepts valid metadata and rejects malformed metadata."""
    state = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=1,
        clock_seconds=900,
        down=1,
        distance=10,
        yardline=50,
        game_id="game_123",
        source="cfbd",
        observed_utc="2026-09-07T19:00:00+00:00",
        ingested_utc="2026-09-07T19:00:05+00:00",
        source_sequence=42,
    )
    assert state.game_id == "game_123"
    assert state.source == "cfbd"
    assert state.source_sequence == 42

    # Ingested precedes observed
    with pytest.raises(SchemaError, match="ingested_utc cannot precede observed_utc"):
        GameState(
            home_team_id="H",
            away_team_id="A",
            possession_team_id="H",
            quarter=1,
            clock_seconds=900,
            down=1,
            distance=10,
            yardline=50,
            observed_utc="2026-09-07T19:00:05+00:00",
            ingested_utc="2026-09-07T19:00:00+00:00",
        )

    # Negative source sequence
    with pytest.raises(SchemaError, match="source_sequence must be non-negative"):
        GameState(
            home_team_id="H",
            away_team_id="A",
            possession_team_id="H",
            quarter=1,
            clock_seconds=900,
            down=1,
            distance=10,
            yardline=50,
            source_sequence=-5,
        )


def test_live_play_with_provenance_metadata() -> None:
    """Proves LivePlay accepts valid metadata and rejects invalid sequences/timestamps."""
    play = LivePlay(
        play_id="p1",
        drive_id="d1",
        team_id="H",
        quarter=1,
        clock_seconds=900,
        down=1,
        distance=10,
        yardline=50,
        game_id="game_123",
        source="pbp_feed",
        observed_utc="2026-09-07T19:00:00+00:00",
        ingested_utc="2026-09-07T19:00:01+00:00",
        source_sequence=1,
    )
    assert play.game_id == "game_123"
    assert play.source_sequence == 1

    with pytest.raises(SchemaError, match="source_sequence must be non-negative"):
        LivePlay(
            play_id="p1",
            drive_id="d1",
            team_id="H",
            quarter=1,
            clock_seconds=900,
            down=1,
            distance=10,
            yardline=50,
            source_sequence=-2,
        )


def test_live_state_tracker_compute_momentum_unknown_team() -> None:
    """Proves compute_momentum raises SchemaError when asked about a non-participating team."""
    tracker = LiveStateTracker(
        home_team_id="TEXAS", away_team_id="OU", initial_possession_team_id="TEXAS"
    )
    with pytest.raises(SchemaError, match="Unknown team ID 'BAMA'"):
        tracker.compute_momentum("BAMA")


def test_win_probability_and_spread_final_state() -> None:
    """Proves calculate_win_probability and project_live_spread respect is_final=True."""
    state = GameState(
        home_team_id="H",
        away_team_id="A",
        possession_team_id="H",
        quarter=4,
        clock_seconds=300,  # 5 min remaining on clock, but called final
        down=1,
        distance=10,
        yardline=50,
        home_score=35,
        away_score=14,
        is_final=True,
    )
    assert calculate_win_probability(state, team_id="H") == 1.0
    assert calculate_win_probability(state, team_id="A") == 0.0

    spread = project_live_spread(state, lines=(-14.5, -21.5))
    assert spread["projected_home_margin"] == 21.0
    assert spread["projected_home_spread"] == -21.0
    assert spread["prob_cover"][-14.5] == 1.0  # 21 - 14.5 > 0
    assert spread["prob_cover"][-21.5] == 0.0  # 21 - 21.5 < 0


def test_win_probability_to_american_odds_edge_cases() -> None:
    """Proves win_probability_to_american_odds returns None for NaN, inf, and non-numeric inputs."""
    assert win_probability_to_american_odds(float("nan")) is None
    assert win_probability_to_american_odds(float("inf")) is None
    assert win_probability_to_american_odds(float("-inf")) is None
    assert win_probability_to_american_odds("not_a_number") is None  # type: ignore


def test_live_records_require_complete_provenance() -> None:
    """Point-in-time records fail closed when source identity is absent."""
    with pytest.raises(SchemaError, match="game_id is required"):
        _GameState(
            "H",
            "A",
            "H",
            quarter=1,
            clock_seconds=900,
            down=1,
            distance=10,
            yardline=75,
        )
    with pytest.raises(SchemaError, match="game_id is required"):
        _LivePlay(
            "p1",
            "d1",
            "H",
            quarter=1,
            clock_seconds=900,
            down=1,
            distance=10,
            yardline=75,
        )


def test_record_play_validation_is_atomic() -> None:
    """A rejected terminal state cannot partially mutate the tracker."""
    tracker = LiveStateTracker("H", "A", "H")
    tracker.start_drive(
        "d1",
        "H",
        quarter=1,
        clock_seconds=900,
        start_yardline=75,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=1,
    )
    before = tracker.get_current_state()

    with pytest.raises(SchemaError, match="Down must be in"):
        tracker.record_play(
            "p1",
            yards_gained=5.0,
            next_down=5,
            next_distance=5,
            next_yardline=70,
            next_quarter=1,
            next_clock_seconds=860,
            observed_utc=_STAMP,
            ingested_utc=_STAMP,
            source_sequence=2,
        )

    assert tracker.get_current_state() == before
    assert tracker.active_drive_plays == []
    assert tracker.total_plays_observed == 0
    assert tracker.seen_play_ids == set()


def test_tracker_rejects_duplicate_and_out_of_order_events() -> None:
    """Replay identity, sequence, observation time, and game clock are monotonic."""
    tracker = LiveStateTracker("H", "A", "H")
    tracker.start_drive(
        "d1",
        "H",
        quarter=1,
        clock_seconds=900,
        start_yardline=75,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=1,
    )
    tracker.record_play(
        "p1",
        yards_gained=5.0,
        next_down=2,
        next_distance=5,
        next_yardline=70,
        next_quarter=1,
        next_clock_seconds=860,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=2,
    )
    before = tracker.get_current_state()

    with pytest.raises(SchemaError, match="Duplicate play_id"):
        tracker.record_play(
            "p1",
            1.0,
            3,
            4,
            69,
            1,
            850,
            observed_utc=_STAMP,
            ingested_utc=_STAMP,
            source_sequence=3,
        )
    with pytest.raises(SchemaError, match="source_sequence must increase"):
        tracker.record_play(
            "p2",
            1.0,
            3,
            4,
            69,
            1,
            850,
            observed_utc=_STAMP,
            ingested_utc=_STAMP,
            source_sequence=2,
        )
    with pytest.raises(SchemaError, match="Game clock cannot move backward"):
        tracker.record_play(
            "p2",
            1.0,
            3,
            4,
            69,
            1,
            880,
            observed_utc=_STAMP,
            ingested_utc=_STAMP,
            source_sequence=3,
        )

    assert tracker.get_current_state() == before
    assert len(tracker.active_drive_plays) == 1


def test_end_drive_validation_is_atomic() -> None:
    """A rejected drive close preserves the active drive and possession."""
    tracker = LiveStateTracker("H", "A", "H")
    tracker.start_drive(
        "d1",
        "H",
        quarter=1,
        clock_seconds=900,
        start_yardline=75,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=1,
    )
    before = tracker.get_current_state()

    with pytest.raises(SchemaError, match="Unknown next possession team"):
        tracker.end_drive(
            DriveOutcome.PUNT.value,
            next_possession_team_id="X",
            observed_utc=_STAMP,
            ingested_utc=_STAMP,
            source_sequence=2,
        )
    with pytest.raises(SchemaError, match="source_sequence must increase"):
        tracker.end_drive(
            DriveOutcome.PUNT.value,
            next_possession_team_id="A",
            observed_utc=_STAMP,
            ingested_utc=_STAMP,
            source_sequence=1,
        )

    assert tracker.get_current_state() == before
    assert tracker.drive_active is True
    assert tracker.completed_drives_home == []


def test_live_models_reject_nonfinite_and_malformed_inputs() -> None:
    """NaN, infinity, boolean numerics, and fractional possession counts fail closed."""
    state = GameState("H", "A", "H", 2, 600, 1, 10, 50)
    with pytest.raises(SchemaError, match="pregame_pace must be finite"):
        project_live_team_totals(state, pregame_pace=float("nan"))
    with pytest.raises(SchemaError, match="lines_game must be finite"):
        project_live_team_totals(state, lines_game=(float("inf"),))
    with pytest.raises(SchemaError, match="non-negative integer"):
        project_live_team_totals(state, observed_possessions_home=2.5)  # type: ignore[arg-type]
    with pytest.raises(SchemaError, match="off_efficiency must be numeric"):
        estimate_next_drive_outcomes(75, off_efficiency=True)


def test_live_market_family_mapping_fails_closed() -> None:
    """Market names cannot be relabeled into incompatible market families."""
    with pytest.raises(SchemaError, match="incompatible"):
        validate_live_market("live_team_total", market_type="GAMELINE")
    with pytest.raises(SchemaError, match="incompatible"):
        validate_live_market("live_game_total", market_type="TEAM_PROP")
    with pytest.raises(SchemaError, match="incompatible"):
        validate_live_market("next_drive_outcome", market_type="GAMELINE")


def test_final_pushes_and_shadow_metadata_are_explicit() -> None:
    """Integer-line pushes are represented and all model outputs remain shadow-only."""
    final_state = GameState(
        "H",
        "A",
        "H",
        4,
        0,
        1,
        10,
        50,
        home_score=31,
        away_score=28,
        is_final=True,
    )
    totals = project_live_team_totals(
        final_state,
        lines_home=(31.0,),
        lines_away=(28.0,),
        lines_game=(59.0,),
    )
    assert totals.home.prob_push[31.0] == 1.0
    assert totals.away.prob_push[28.0] == 1.0
    assert totals.prob_game_push[59.0] == 1.0
    assert totals.model_status == "uncalibrated_shadow"
    assert totals.is_actionable is False
    assert totals.home.is_actionable is False

    spread = project_live_spread(final_state, lines=(-3.0,))
    assert spread["prob_cover"][-3.0] == 0.0
    assert spread["prob_push"][-3.0] == 1.0
    assert spread["model_status"] == "uncalibrated_shadow"
    assert spread["is_actionable"] is False

    drive = estimate_next_drive_outcomes(75)
    assert drive.model_status == "uncalibrated_shadow"
    assert drive.is_actionable is False


# ============================================================================
# 9. Defensive Scoring Attribution Tests
# ============================================================================


def test_defensive_score_attributed_to_correct_team() -> None:
    """Proves a pick-6 credits the defensive team, not the offensive team."""
    tracker = LiveStateTracker("H", "A", "H")
    tracker.start_drive(
        "d1",
        "H",
        quarter=2,
        clock_seconds=600,
        start_yardline=70,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=1,
    )
    # Interception returned for a touchdown — away team (defense) scores
    state = tracker.record_play(
        "p_pick6",
        yards_gained=-5.0,
        next_down=1,
        next_distance=10,
        next_yardline=75,
        next_quarter=2,
        next_clock_seconds=580,
        is_turnover=True,
        is_scoring=True,
        points_scored=6,
        scoring_team_id="A",  # defense scored
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=2,
    )
    assert state.home_score == 0
    assert state.away_score == 6


def test_scoring_team_id_required_when_points_scored() -> None:
    """Proves scoring_team_id must be non-empty when points_scored > 0."""
    with pytest.raises(SchemaError, match="scoring_team_id is required"):
        LivePlay(
            play_id="p1",
            drive_id="d1",
            team_id="H",
            quarter=1,
            clock_seconds=900,
            down=1,
            distance=10,
            yardline=20,
            is_scoring=True,
            points_scored=7,
            scoring_team_id="",
        )


def test_scoring_team_id_must_be_game_participant() -> None:
    """Proves record_play rejects scoring_team_id not matching home/away."""
    tracker = LiveStateTracker("H", "A", "H")
    tracker.start_drive(
        "d1",
        "H",
        quarter=1,
        clock_seconds=900,
        start_yardline=75,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=1,
    )
    with pytest.raises(SchemaError, match="scoring_team_id 'X' must be either"):
        tracker.record_play(
            "p1",
            yards_gained=10.0,
            next_down=1,
            next_distance=10,
            next_yardline=65,
            next_quarter=1,
            next_clock_seconds=860,
            is_scoring=True,
            points_scored=7,
            scoring_team_id="X",
            observed_utc=_STAMP,
            ingested_utc=_STAMP,
            source_sequence=2,
        )


def test_points_scored_upper_bound() -> None:
    """Proves points_scored cannot exceed 8 (max TD + 2pt conversion)."""
    with pytest.raises(SchemaError, match="cannot exceed 8"):
        LivePlay(
            play_id="p1",
            drive_id="d1",
            team_id="H",
            quarter=1,
            clock_seconds=900,
            down=1,
            distance=10,
            yardline=20,
            is_scoring=True,
            points_scored=9,
            scoring_team_id="H",
        )


# ============================================================================
# 10. Event Log & Audit Trail Tests
# ============================================================================


def test_event_log_tracks_complete_mutation_history() -> None:
    """Proves the tracker's event log records every state transition in order."""
    tracker = LiveStateTracker("H", "A", "H")
    log = tracker.get_event_log()
    assert len(log) == 1
    assert log[0]["event"] == "init"

    tracker.start_drive(
        "d1",
        "H",
        quarter=1,
        clock_seconds=900,
        start_yardline=75,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=1,
    )
    tracker.record_play(
        "p1",
        yards_gained=5.0,
        next_down=2,
        next_distance=5,
        next_yardline=70,
        next_quarter=1,
        next_clock_seconds=860,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=2,
    )
    tracker.end_drive(
        DriveOutcome.PUNT.value,
        next_possession_team_id="A",
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=3,
    )
    tracker.set_score(
        7,
        0,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=4,
    )
    tracker.set_timeouts(
        2,
        3,
        observed_utc=_STAMP,
        ingested_utc=_STAMP,
        source_sequence=5,
    )

    log = tracker.get_event_log()
    events = [entry["event"] for entry in log]
    expected = [
        "init",
        "start_drive",
        "record_play",
        "end_drive",
        "set_score",
        "set_timeouts",
    ]
    assert events == expected

    # Sequences must be monotonically increasing
    sequences = [entry["source_sequence"] for entry in log]
    assert sequences == [0, 1, 2, 3, 4, 5]


def test_event_log_is_immutable_copy() -> None:
    """Proves get_event_log returns a copy, not a mutable reference."""
    tracker = LiveStateTracker("H", "A", "H")
    log1 = tracker.get_event_log()
    log1.clear()
    log2 = tracker.get_event_log()
    assert len(log2) == 1  # Original unaffected
