"""Promotion gate evaluation for the moneyline backtest (plan section 8).

``config/promotion.json`` is the two-key gate: sample-size floors AND
demonstrated out-of-sample skill. This module turns a
``MoneylineBacktestReport`` (including its same-game-set market evidence)
into structured evidence and a fail-closed pass/fail. Status flips to
``promoted`` only when every configured gate passes; otherwise it stays
``shadow``. Callers must never hand-edit the status field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfb_analytics import config, paths
from cfb_analytics.backtest.moneyline import MoneylineBacktestReport, SameGameSetEvidence
from cfb_analytics.utils import utc_now_iso


@dataclass(frozen=True)
class PromotionDecision:
    """Result of evaluating ``promotion.json`` gates against backtest evidence."""

    passed: bool
    status: str  # "promoted" or "shadow"
    failures: tuple[str, ...]
    evidence: dict[str, Any]

    def as_text(self) -> str:
        lines = [
            "cfb-analytics promotion gate",
            f"  decision : {'PASS -> promoted' if self.passed else 'FAIL -> shadow (fail-closed)'}",
            f"  status   : {self.status}",
        ]
        if self.failures:
            lines.append("  failures:")
            for failure in self.failures:
                lines.append(f"    - {failure}")
        else:
            lines.append("  failures: (none)")
        return "\n".join(lines)


def _calibration_gap_from_buckets(
    buckets: list[Any], *, min_bucket_n: int
) -> float | None:
    """Max |win_rate - bucket midpoint| over reliability buckets with n >= floor.

    Returns None when no bucket clears ``min_bucket_n`` (fail-closed: gap
    unknown is not a pass).
    """
    gaps: list[float] = []
    for bucket in buckets:
        if bucket.n < min_bucket_n or bucket.win_rate is None:
            continue
        mid = 0.5 * (bucket.low + min(bucket.high, 1.0))
        gaps.append(abs(bucket.win_rate - mid))
    if not gaps:
        return None
    return max(gaps)


def build_promotion_evidence(
    report: MoneylineBacktestReport,
    *,
    seasons: tuple[int, ...],
) -> dict[str, Any]:
    """Structured evidence JSON for ``--promote`` (and promotion.json)."""
    sgs: SameGameSetEvidence | None = report.same_game_set
    ensemble = report.ensemble_seasons
    cal_gap = None
    # Prefer overlap-set ensemble reliability; fall back to full ensemble.
    cal_source = (
        sgs.ensemble if sgs is not None and sgs.ensemble is not None else ensemble
    )
    promo = config.promotion()
    min_bucket_n = int(promo.get("min_bucket_n_for_gap_test", 100))
    if cal_source is not None:
        cal_gap = _calibration_gap_from_buckets(
            cal_source.reliability, min_bucket_n=min_bucket_n
        )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "promotion_candidate": "ensemble",
        "seasons_requested": list(seasons),
        "n_full": report.seasons.n_games,
        "n_overlap": sgs.n_overlap if sgs is not None else 0,
        "n_skipped_no_market": sgs.n_skipped if sgs is not None else report.skipped_market_unpriced,
        "seasons_with_overlap": list(sgs.seasons_covered) if sgs is not None else [],
        "market_min_books": sgs.market_min_books if sgs is not None else None,
        "clv_formula": (
            sgs.clv_formula
            if sgs is not None
            else SameGameSetEvidence.CLV_FORMULA
        ),
        "clv_units": "probability_points",
        "clv_bps_scale": 10_000,
        "beat_market": sgs.beat_market if sgs is not None else None,
        "median_clv": sgs.median_clv if sgs is not None else None,
        "median_clv_bps": sgs.median_clv_bps if sgs is not None else None,
        "median_clv_non_negative": (
            sgs.median_clv_non_negative if sgs is not None else None
        ),
        "calibration_gap": cal_gap,
        "logloss": {},
        "brier": {},
    }
    if sgs is not None:
        if sgs.market is not None:
            evidence["logloss"]["market"] = sgs.market.log_loss
            evidence["brier"]["market"] = sgs.market.brier
        if sgs.ensemble is not None:
            evidence["logloss"]["ensemble"] = sgs.ensemble.log_loss
            evidence["brier"]["ensemble"] = sgs.ensemble.brier
        if sgs.ensemble_raw is not None:
            evidence["logloss"]["ensemble_raw"] = sgs.ensemble_raw.log_loss
            evidence["brier"]["ensemble_raw"] = sgs.ensemble_raw.brier
        if sgs.calibration_method is not None:
            evidence["ensemble_calibrator"] = sgs.calibration_method
        if sgs.ridge is not None:
            evidence["logloss"]["ridge"] = sgs.ridge.log_loss
            evidence["brier"]["ridge"] = sgs.ridge.brier
        if sgs.elo is not None:
            evidence["logloss"]["elo"] = sgs.elo.log_loss
            evidence["brier"]["elo"] = sgs.elo.brier
    return evidence


def evaluate_promotion(
    report: MoneylineBacktestReport,
    *,
    seasons: tuple[int, ...],
    promotion_cfg: dict[str, Any] | None = None,
) -> PromotionDecision:
    """Fail-closed two-key evaluation. Missing evidence => shadow."""
    cfg = promotion_cfg if promotion_cfg is not None else config.promotion()
    evidence = build_promotion_evidence(report, seasons=seasons)
    failures: list[str] = []

    min_games = int(cfg.get("min_settled_games", 1500))
    n_overlap = int(evidence["n_overlap"] or 0)
    if n_overlap < min_games:
        failures.append(
            f"n_overlap {n_overlap} < min_settled_games {min_games}"
        )

    min_seasons = int(cfg.get("min_seasons_backtested", 3))
    seasons_covered = list(evidence.get("seasons_with_overlap") or [])
    if len(seasons_covered) < min_seasons:
        failures.append(
            f"seasons_with_overlap {seasons_covered!r} "
            f"({len(seasons_covered)}) < min_seasons_backtested {min_seasons}"
        )

    if cfg.get("require_oos_logloss_beat_market", True):
        beat = evidence.get("beat_market")
        if beat is not True:
            failures.append(
                f"require_oos_logloss_beat_market: beat_market={beat!r} "
                f"(need ensemble_logloss < market_logloss on same-game set)"
            )

    if cfg.get("require_median_clv_non_negative", True):
        ok = evidence.get("median_clv_non_negative")
        if ok is not True:
            failures.append(
                f"require_median_clv_non_negative: "
                f"median_clv={evidence.get('median_clv')!r} "
                f"(non_negative={ok!r})"
            )

    max_gap = float(cfg.get("max_abs_calibration_gap", 0.03))
    gap = evidence.get("calibration_gap")
    if gap is None:
        failures.append(
            "calibration_gap unavailable "
            f"(need reliability buckets with n>={cfg.get('min_bucket_n_for_gap_test', 100)})"
        )
    elif gap > max_gap:
        failures.append(
            f"calibration_gap {gap:.4f} > max_abs_calibration_gap {max_gap}"
        )

    passed = not failures
    status = "promoted" if passed else "shadow"
    evidence = {
        **evidence,
        "gate_passed": passed,
        "failures": list(failures),
        "evaluated_utc": utc_now_iso(),
    }
    return PromotionDecision(
        passed=passed,
        status=status,
        failures=tuple(failures),
        evidence=evidence,
    )


def write_promotion_result(decision: PromotionDecision) -> Path:
    """Persist evidence; flip status only when gates pass (else force shadow)."""
    path = paths.CONFIG_DIR / "promotion.json"
    current = config.promotion()
    # Fail-closed: never leave a prior 'promoted' if this run failed.
    new_status = "promoted" if decision.passed else "shadow"
    payload = {
        **{k: v for k, v in current.items() if not k.startswith("_")},
        "status": new_status,
        "changed_utc": (
            utc_now_iso()
            if new_status != current.get("status")
            else current.get("changed_utc")
        ),
        "evidence": decision.evidence,
        "_comment": current.get(
            "_comment",
            "Two-key gate: BOTH a sample-size floor AND demonstrated "
            "out-of-sample skill. While status is 'shadow', no CORE tier is "
            "emitted and every artifact is stamped UNPROMOTED. Never "
            "hand-edit status to 'promoted' - it is set by "
            "`cfb-analytics backtest --promote` on passing evidence.",
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config._read_json.cache_clear()
    return path
