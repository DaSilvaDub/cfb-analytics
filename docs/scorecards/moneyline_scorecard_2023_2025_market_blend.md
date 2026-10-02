# Moneyline scorecard — market blend floor + skill-gap segments (2023–2025)

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~02:40 EDT** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| Command | `cli backtest --start-year 2023 --end-year 2025 --historical-cfbd-min-books 1 --promote` |
| Floor | `settings.blend.market_weight_floor = **0.75**` (now applied in backtest/promote) |
| Promotion | `config/promotion.json` **status=`shadow`** (fail-closed; not hand-flipped) |

## Audit: where was the floor used?

| Path | Before this change | After |
|---|---|---|
| `config/settings.json` → `blend.market_weight_floor` | Documented only | Still the source of truth |
| Live CORE / scanners / mispriced | **Not applied** (`min_blend_prob` / floor unused in emission) | Helper `config.market_weight_floor()` + `models/market_blend.blend_with_market` available; scanners still do not blend (CORE blocked by shadow) |
| Moneyline backtest / `--promote` | **Not applied** (pure ensemble vs market only) | Walk-forward blend evidence on same-game set |

## Track A — blend before / after (same-game set, n_overlap=2103)

`p = w * p_market + (1-w) * p_model`, `w` fit season-blocked on prior seasons only in `[0.75, 1]`.

| Candidate | Log loss | Brier | vs market |
|---|---:|---:|---|
| Pure market (w=1) | **0.5287** | **0.1781** | — |
| Blend walk-forward (w≥0.75) | 0.5292 | 0.1782 | +0.0005 (slightly worse) |
| Blend fixed w=0.75 | 0.5297 | 0.1783 | +0.0011 |
| Pure model w=0 (calibrated ens) | 0.5560 | 0.1884 | +0.0274 |
| Ensemble raw (pre-cal) | 0.5548 | 0.1881 | +0.0262 |

Walk-forward `w` by season: **2023=0.75** (no prior → floor), **2024=1.00**, **2025=1.00**.
Prior seasons always prefer pure market — the model residual has not earned share yet.

### Tension (honest)

Floor discipline **protects** live edges from an overconfident model (model share ≤ 0.25). With `w` near 1, blend logloss ≈ market, so a “beat market” gate on the **blend** is nearly tautological / gameable. Blending toward market improves logloss vs pure model (0.556 → 0.529) but does **not** demonstrate skill.

### Gate rule (design-of-record, encoded)

**`require_oos_logloss_beat_market` applies to the pure calibrated ensemble**, not the floor-constrained blend.

- Lower `market_weight_floor` only after pure model beats market OOS on the same-game set.
- Until then, keep the floor for live scoring; report blend brackets as evidence only.
- Median CLV stays on the **model** side (raw ensemble vs close).
- Do **not** redefine `beat_market` onto the blend.

`blend_gate_interpretation` in evidence / `SameGameSetEvidence`:
`pure_model_must_beat_market_before_lowering_floor`.

### Gate status (still shadow)

| Gate | Status |
|---|---|
| n_overlap ≥ 1500 | pass (2103) |
| ≥3 seasons overlap | pass |
| Pure-model OOS logloss beat market | **fail** (0.5560 ≮ 0.5287) |
| Median CLV ≥ 0 | pass (+3.7 bps) |
| Calibration gap ≤ 0.03 | **fail** (0.0370) |
| Blend non-worse (evidence only) | false (0.5292 ≰ 0.5287; 2023 floor drag) |

## Track B — where ensemble loses most to market

Raw ensemble vs market on overlap (n=2103). Gap = ens_ll − mkt_ll (positive ⇒ ens worse).

### By season

| Season | n | ens | mkt | gap |
|---|---:|---:|---:|---:|
| 2024 | 715 | 0.5722 | 0.5366 | **+0.0356** |
| 2023 | 681 | 0.5381 | 0.5162 | +0.0219 |
| 2025 | 707 | 0.5534 | 0.5327 | +0.0207 |

### By week (largest lever)

| Weeks | n | ens | mkt | gap | ridge | internal Elo |
|---|---:|---:|---:|---:|---:|---:|
| **1–4** | 462 | 0.5404 | 0.4899 | **+0.0505** | 0.6689 | 0.5549 |
| 5–9 | 799 | 0.5682 | 0.5393 | +0.0288 | 0.5873 | 0.5736 |
| 10+ | 842 | 0.5501 | 0.5398 | +0.0103 | 0.5578 | 0.5513 |

Early season is the dominant skill hole: ridge is unusable weeks 1–4; Elo holds up better but still trails market.

### Other segments

| Segment | Worst bucket | gap |
|---|---|---:|
| Market fav strength | lean 55–65% / lock 80%+ | ~+0.032 |
| Favorite side | market **away** favorite | **+0.0350** (home fav +0.0206) |
| Home conference | American Athletic / Big 12 | ~+0.041 |
| Venue | home/away vs neutral | similar |

### Member skill vs market (overlap)

| Member | logloss | gap vs market |
|---|---:|---:|
| Market | 0.5287 | — |
| Ensemble (raw, w≈ elo 0.80 / ridge 0.15 / logit 0.05) | 0.5548 | +0.0262 |
| Internal Elo | 0.5606 | +0.0282 |
| Ridge | 0.5934 | +0.0648 |
| Logit | 0.6125 | +0.0838 |

### Spikes tried (none clearly helped OOS — not shipped)

| Experiment | Same-set ens logloss | vs global 0.5548 |
|---|---:|---|
| Drop logit (renorm) | 0.5558 | worse |
| Season-blocked WF ensemble weights | 0.5565 | worse |
| WF weights + elo-heavy cold start | 0.5555 | worse |
| Drop ridge in weeks 1–4 | 0.5637 | much worse |
| Elo-only in weeks 1–4 | 0.5549 | flat |

No small low-overfit weight tweak closed the gap. Stopped (no feature spray).

## Ranked skill upgrades (top 3)

1. **Early-season prior / delay ridge trust (weeks 1–4)** — gap +0.0505; ridge LL 0.67 vs market 0.49. Cite `backtest/harness.py` (min_games / same-week cutoff), `features/team_ratings.py` + `models/shrinkage.py` (preseason blend), `features/preseason.py`. Spike plan: raise effective `min_games` for ridge member only, or replace early ridge with a stronger preseason margin prior; remeasure weeks 1–4 slice + full overlap.

2. **Away-favorite / HFA calibration** — market away-fav gap +0.0350 vs home-fav +0.0206. Cite `models/elo.py` (`HFA_ELO_POINTS`), `settings.model.elo_home_field` (60). Spike plan: walk-forward fit HFA on prior seasons only; segment OOS by home/away favorite.

3. **Third ensemble member quality (logit / QB)** — logit alone +0.0838 vs market; rest/talent already in `features/ensemble.py`, but T3 QB features (`qb_epa_pp_diff`, `qb_status_penalty`, …) are documented gaps and **QB confirmed is a live CORE blocker not present in backtest**. Cite `features/ensemble.py` docstring, `features/qb.py`. Spike plan: when `player_game_passing` coverage spans ≥3 seasons, add one QB feature under walk-forward; until then do not lower logit weight further (already 0.05; dropping it did not help).

## Files touched

- `cfb_analytics/models/market_blend.py` — pure blend math
- `cfb_analytics/backtest/market_blend_fit.py` — walk-forward `w` fit
- `cfb_analytics/backtest/moneyline.py` — same-game-set blend brackets
- `cfb_analytics/backtest/promote.py` — evidence + gate wording
- `cfb_analytics/config.py` — `market_weight_floor()` helper
- `tests/test_market_blend.py`
