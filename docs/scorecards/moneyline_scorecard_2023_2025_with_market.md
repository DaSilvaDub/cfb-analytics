# Moneyline scorecard — 2023–2025 with market close baseline

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~00:13 EDT** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| DB | local `data/cfb.sqlite3` after historical backfill for **2023, 2024, 2025** |
| Command | `cli backtest --start-year 2023 --end-year 2025 --promote` |
| Market floor | `historical_cfbd_min_books=1` (baseline-only; CFBD provider counts are thin) |
| Promotion | `config/promotion.json` **status=`shadow`** (gates fail-closed) |

## Historical backfill join rates (A4)

| Season | ML close / FBS | Rate |
|---|---:|---:|
| 2023 | 746 / 796 | **0.937** |
| 2024 | 783 / 803 | **0.975** |
| 2025 | 817 / 809 | **1.010** † |
| A2 leakage | 0 | pass |

† Denominator is the CLI’s FBS filter; a few priced games sit outside that count. Treat as ≥0.85 pass, not literal over-coverage.

## Full walk-forward (headline n; not the promote compare)

| Model | n | Brier | Log loss |
|---|---:|---:|---:|
| Internal ridge | 2204 | 0.1880 | 0.5680 |
| Elo-only | 2125 | 0.1869 | 0.5519 |
| Internal Elo | 2126 | 0.1856 | 0.5469 |
| P_logit | 2204 | 0.2045 | 0.5971 |
| **Ensemble** | **2204** | **0.1801** | **0.5320** |
| Market close (covered subset) | 2103 | 0.1781 | 0.5287 |

Skipped (no close ML consensus at `min_books>=1`): **101**. Always printed; never silent.

## Same-game-set promote evidence (honest overlap)

Promotion candidate = **ensemble**. Metrics below share the **identical** game set
(model prediction ∩ vig-free close P(home)).

| Model | n | Brier | Log loss |
|---|---:|---:|---:|
| **Market close** | **2103** | **0.1781** | **0.5287** |
| **Ensemble** | **2103** | **0.1881** | **0.5548** |
| Ridge | 2103 | 0.1964 | 0.5934 |
| Elo-only | 2067 | 0.1915 | 0.5650 |

| Gate field | Value |
|---|---|
| n_full / n_overlap / skipped | 2204 / **2103** / 101 |
| seasons_with_overlap | 2023, 2024, 2025 |
| `beat_market` (ensemble_ll < market_ll) | **false** (0.5548 vs 0.5287) |
| median CLV (model-vs-close) | **+0.0004** prob-pts (**+3.7 bps**); `median_clv_non_negative=true` |
| calibration_gap (ensemble reliability, n≥100) | **0.0659** (> 0.03) |

### CLV formula (model-vs-close)

Open ML is typically N/A from CFBD, so promote evidence uses **model-vs-close**
probability CLV on the side the model favors:

- if `p_model >= 0.5` (favors home): `clv = p_close - p_model`
- else (favors away): `clv = p_model - p_close`

Units: probability points (0.01 = 1pp); bps = 10000 × points. Positive ⇒ close
assigned more probability to the model’s favored side than the model did
(classic “better number than close”).

## Gate implication (still shadow)

| Promotion need | Status |
|---|---|
| ≥1500 settled overlap games | **pass** (2103) |
| ≥3 seasons with overlap | **pass** (2023–25) |
| OOS logloss beat market (same set) | **fail** |
| Median CLV ≥ 0 | **pass** (+3.7 bps) |
| Calibration gap ≤ 0.03 | **fail** (0.0659) |
| `status` flip | **No** — stays `shadow` (fail-closed) |

## Implementation notes

- `cfb_analytics/backtest/moneyline.py` — same-game-set section + CLV helpers
- `cfb_analytics/backtest/promote.py` — gate eval + evidence JSON write
- `cfb-analytics backtest --promote` — prints evidence; writes `config/promotion.json`
- Headline “ridge vs market” compare on mismatched n is **removed**; use same-game-set

## Follow-up: walk-forward ensemble calibration

See [`moneyline_scorecard_2023_2025_calibration.md`](moneyline_scorecard_2023_2025_calibration.md).
Season-blocked Platt reduces `calibration_gap` **0.0659 → 0.0370** without
beating market on logloss; status remains **shadow**. The 0.03 gate is not
honestly reachable on this window with few-DOF parametric calibrators.

