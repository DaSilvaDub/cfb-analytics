"""Walk-forward live model prediction collection.

Iterates seasons -> weeks -> games, replays each game's PBP data to
produce ``GameState`` checkpoints, then calls the live micro-markets
model at each checkpoint to collect predictions against actual outcomes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from cfb_analytics.backtest.replay import (
    CheckpointLabel,
    replay_game,
)
from cfb_analytics.models.live import (
    calculate_win_probability,
    estimate_next_drive_outcomes,
    project_live_team_totals,
)
from cfb_analytics.models.ridge import DEFAULT_MIN_GAMES, DEFAULT_RIDGE_LAMBDA, fit_ratings


@dataclass(frozen=True)
class WinProbPrediction:
    game_id: str
    season: int
    week: int
    checkpoint: CheckpointLabel
    predicted_home_win_prob: float
    actual_home_won: bool
    pregame_home_margin: float
    seconds_remaining: int


@dataclass(frozen=True)
class DriveOutcomePrediction:
    game_id: str
    season: int
    week: int
    drive_number: int
    start_yardline: int
    predicted_td: float
    predicted_fg: float
    predicted_punt: float
    predicted_turnover: float
    predicted_safety: float
    actual_outcome: str


@dataclass(frozen=True)
class TotalsPrediction:
    game_id: str
    season: int
    week: int
    checkpoint: CheckpointLabel
    team_id: str
    projected_total: float
    actual_total: int
    current_score: int
    remaining_seconds: int


@dataclass
class LiveWalkForwardRun:
    win_prob_predictions: list[WinProbPrediction] = field(default_factory=list)
    drive_outcome_predictions: list[DriveOutcomePrediction] = field(default_factory=list)
    totals_predictions: list[TotalsPrediction] = field(default_factory=list)
    games_replayed: int = 0
    games_skipped_no_pbp: int = 0


def _regular_season_weeks(conn: sqlite3.Connection, season: int) -> list[int]:
    rows = conn.execute(
        """SELECT DISTINCT week FROM games
           WHERE source = 'cfbd' AND season = ? AND season_type = 'regular'
             AND completed = 1 AND week IS NOT NULL
           ORDER BY week""",
        (season,),
    ).fetchall()
    return [int(row["week"]) for row in rows]


def _row_to_history_game(row: dict) -> dict:
    return {
        "home_team_id": row["home_team_id"],
        "away_team_id": row["away_team_id"],
        "home_points": row["home_points"],
        "away_points": row["away_points"],
        "neutral_site": row["neutral_site"],
        "kickoff_utc": row["kickoff_utc"],
    }


def run_live_walk_forward(
    conn: sqlite3.Connection,
    seasons: list[int],
    *,
    checkpoint_kinds: frozenset[str] = frozenset({"end_of_quarter", "drive_start", "halftime"}),
    min_games: int = DEFAULT_MIN_GAMES,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
) -> LiveWalkForwardRun:
    """Walk-forward live model backtesting across seasons.

    For each game, computes a pregame home margin from ridge ratings
    (with the same leakage cutoff as harness.py: only games with
    kickoff strictly before the week's earliest kickoff), then replays
    the game to generate checkpoints and score predictions.
    """
    run = LiveWalkForwardRun()

    for season in seasons:
        for week in _regular_season_weeks(conn, season):
            week_games = conn.execute(
                """SELECT game_id, home_team_id, away_team_id, home_points,
                          away_points, neutral_site, kickoff_utc
                   FROM games
                   WHERE source = 'cfbd' AND season = ? AND season_type = 'regular'
                     AND week = ? AND completed = 1
                     AND home_points IS NOT NULL AND away_points IS NOT NULL""",
                (season, week),
            ).fetchall()
            if not week_games:
                continue

            as_of_utc = min(row["kickoff_utc"] for row in week_games)

            # Fit ridge ratings using only prior games this season
            history_rows = conn.execute(
                """SELECT home_team_id, away_team_id, home_points, away_points,
                          neutral_site, kickoff_utc
                   FROM games
                   WHERE source = 'cfbd' AND season = ? AND completed = 1
                     AND kickoff_utc < ?""",
                (season, as_of_utc),
            ).fetchall()
            ratings = fit_ratings(
                [_row_to_history_game(dict(row)) for row in history_rows],
                ridge_lambda=ridge_lambda,
                min_games=min_games,
                as_of=datetime.fromisoformat(as_of_utc),
            )

            for row in week_games:
                game_id = str(row["game_id"])
                home_id = str(row["home_team_id"])
                away_id = str(row["away_team_id"])
                home_final = int(row["home_points"])
                away_final = int(row["away_points"])
                neutral = bool(row["neutral_site"])
                actual_home_won = home_final > away_final

                # Pregame margin from ridge (None if team not rated)
                margin = ratings.margin(home_id, away_id, neutral_site=neutral)
                pregame_margin = margin if margin is not None else 0.0

                # Replay the game
                checkpoints = replay_game(conn, game_id, checkpoint_kinds=checkpoint_kinds)
                if not checkpoints:
                    run.games_skipped_no_pbp += 1
                    continue

                run.games_replayed += 1

                for cp in checkpoints:
                    state = cp.state

                    # Win probability prediction
                    try:
                        home_wp = calculate_win_probability(
                            state,
                            pregame_home_margin=pregame_margin,
                            team_id=home_id,
                        )
                    except Exception:
                        continue

                    run.win_prob_predictions.append(
                        WinProbPrediction(
                            game_id=game_id,
                            season=season,
                            week=week,
                            checkpoint=cp.label,
                            predicted_home_win_prob=home_wp,
                            actual_home_won=actual_home_won,
                            pregame_home_margin=pregame_margin,
                            seconds_remaining=state.seconds_remaining_in_game,
                        )
                    )

                    # Drive outcome prediction (only at drive_start)
                    if cp.label.kind == "drive_start" and cp.actual_drive_outcome:
                        try:
                            dist = estimate_next_drive_outcomes(state.yardline)
                            run.drive_outcome_predictions.append(
                                DriveOutcomePrediction(
                                    game_id=game_id,
                                    season=season,
                                    week=week,
                                    drive_number=cp.label.drive_number or 0,
                                    start_yardline=state.yardline,
                                    predicted_td=dist.touchdown,
                                    predicted_fg=dist.field_goal,
                                    predicted_punt=dist.punt,
                                    predicted_turnover=dist.turnover_downs,
                                    predicted_safety=dist.safety,
                                    actual_outcome=cp.actual_drive_outcome,
                                )
                            )
                        except Exception:
                            pass

                    # Team totals prediction
                    try:
                        totals = project_live_team_totals(state)
                        run.totals_predictions.append(
                            TotalsPrediction(
                                game_id=game_id,
                                season=season,
                                week=week,
                                checkpoint=cp.label,
                                team_id=home_id,
                                projected_total=totals.home.projected_total,
                                actual_total=home_final,
                                current_score=state.home_score,
                                remaining_seconds=state.seconds_remaining_in_game,
                            )
                        )
                        run.totals_predictions.append(
                            TotalsPrediction(
                                game_id=game_id,
                                season=season,
                                week=week,
                                checkpoint=cp.label,
                                team_id=away_id,
                                projected_total=totals.away.projected_total,
                                actual_total=away_final,
                                current_score=state.away_score,
                                remaining_seconds=state.seconds_remaining_in_game,
                            )
                        )
                    except Exception:
                        pass

    return run
