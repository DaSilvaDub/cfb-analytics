# Moneyline scorecard — ensemble walk-forward calibration (2023–2025)

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~02:23 EDT** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| Command | `cli backtest --start-year 2023 --end-year 2025 --historical-cfbd-min-books 1 --promote` |
| Calibrator | **Platt on logit(p)**, season-blocked expanding window (`platt/season`) |
| Promotion | `config/promotion.json` **status=`shadow`** (fail-closed; not hand-flipped) |

## Diagnosis (raw ensemble gap 0.0659)

Same-game-set reliability (`n≥100`) before calibration — max \|win_rate − bucket mid\|:

| Bin | n | win_rate | mid | gap |
|---|---:|---:|---:|---:|
| 50%–60% | 283 | 0.484 | 0.550 | **0.0659** |
| 60%–70% | 299 | 0.605 | 0.650 | 0.0446 |
| 20%–30% | 153 | 0.209 | 0.250 | 0.0408 |

Pattern: mid-bin **overstatement of P(home)** (slight home favorites under-win). Extremes (80%+) are already close. Temperature (1 DOF) helps less than Platt (intercept can absorb the home bias).

## Method (anti-leakage)

- Fit `p_cal = sigmoid(a · logit(p) + b)` by IRLS on **prior seasons only**.
- Season 2023 → identity (no prior in the promote window).
- Season 2024 → fit on 2023; season 2025 → fit on 2023+2024.
- Never fit on the evaluation season. Isotonic omitted (easy to overfit thin bins).
- `--promote` uses **calibrated** ensemble for `calibration_gap` + `beat_market`; **raw** ensemble still logged (`ensemble_raw`) and used for median CLV.

## Before / after (same-game set, n_overlap=2103)

| Metric | Raw | Calibrated (platt/season) | Market |
|---|---:|---:|---:|
| Log loss | 0.5548 | 0.5560 | **0.5287** |
| Brier | 0.1881 | 0.1884 | **0.1781** |
| calibration_gap | **0.0659** | **0.0370** | — |
| median CLV (raw) | +0.0004 | (unchanged; CLV on raw) | — |
| beat_market | false | false | — |

Temperature week-blocked only reached ~0.057 gap; season-blocked Platt is the best honest parametric result on this window.

## Is 0.03 reachable honestly?

**No** on 2023–2025 market overlap with few-DOF parametric calibrators without fitting on the test slice. Remaining gap (~0.037) is driven largely by the uncalibrated 2023 fold (identity) and residual 10%–20% / 60%–70% structure. Closing to 0.03 would need either more prior seasons in the promote window, a richer model (not a calibrator), or peeking — which we refuse.

Calibration does **not** secretly beat market: calibrated OOS logloss is slightly *worse* than raw (0.5560 vs 0.5548), both still above market 0.5287.

## Gate status (still shadow)

| Gate | Status |
|---|---|
| n_overlap ≥ 1500 | pass (2103) |
| ≥3 seasons overlap | pass |
| OOS logloss beat market | **fail** |
| Median CLV ≥ 0 | pass (+3.7 bps, raw) |
| Calibration gap ≤ 0.03 | **fail** (0.0370) |
