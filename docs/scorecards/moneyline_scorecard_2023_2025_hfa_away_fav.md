# Moneyline scorecard — away-favorite / Elo HFA spike (2023–2025)

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~04:00 ET** |
| Code | `feat/cfbd-historical-lines-2024` (PR #17) |
| Spike script | `scripts/spike_hfa_away_fav.py` (+ score-time follow-ups in `artifacts/spike_hfa_away_fav_scoretime.json`) |
| Same-game set | CFBD historical **close** ML (`min_books=1`) for scoring + fav-side **segmentation**; open prior weeks ≤2 as in production |
| Promotion | still **shadow** — this spike does **not** flip `beat_market` / does **not** change `HFA_ELO_POINTS` |

## Map: how HFA works today

| Layer | Behavior | File pointers |
|---|---|---|
| **Internal Elo (ensemble ~80%)** | Single global constant `HFA_ELO_POINTS = 60` (~2.6 pts per plan). Added to home rating in `expected_score` / `update_ratings` / `EloRatings.probability`. **Neutral site → HFA=0** (already). No conference- or venue-specific HFA (explicitly deferred in `models/elo.py` docstring). | `cfb_analytics/models/elo.py`, `features/elo_internal.py`, `backtest/harness.py` (`elo_hfa=`), `config/settings.json` `model.elo_home_field=60` (**documented only — not wired**; code imports the module constant) |
| **Ridge (~15%)** | `home_field` coefficient **fit from data** each week (`solution[1]`). Neutral → both sides `is_home=False`. Mid-2024 probe ≈ **5.4 pts** (higher than Elo’s 2.6). | `cfb_analytics/models/ridge.py`, `features/team_ratings.py` |
| **Logit (~5%)** | Own home/neutral encoding via `game_features(..., neutral_site=)`. | `features/logistic_features` path in harness |
| **CFBD Elo baseline** | Same plan constant 60; not in promote ensemble. | `backtest/elo_baseline.py` |
| **Futures / reasoning** | Separate ~2.5–2.6 pt HFA knobs; out of moneyline promote path. | `models/futures.py`, `reasoning/models.py` |

**Away / home favorite (segmentation):** market close P(home) — `away_fav` if `p_mkt < 0.5`, `home_fav` if `> 0.5`, `pickem` if `\|p−0.5\| < 0.02`. Close is **not** a same-game feature; open prior remains weeks ≤2 only.

Scoring path for this spike matches production levers already on the branch: early ridge sigma ×1.5 (weeks ≤4), open-market prior replace (weeks ≤2). Ensemble weights = production mix **without logit** (ridge/elo renorm) for speed; Elo dominates (~84% here).

## Root miss (diagnosis)

Baseline `elo_hfa=60`, same-game overlap **n=2103**:

| Segment | n | ens LL | mkt LL | gap | ens bias (p−y) | elo bias |
|---|---:|---:|---:|---:|---:|---:|
| **away_fav** | 780 | 0.5989 | 0.5644 | **+0.0346** | **+0.0617** | **+0.0617** |
| home_fav | 1245 | 0.5154 | 0.4967 | +0.0188 | −0.0199 | −0.0270 |
| pickem | 78 | 0.6754 | 0.6947 | **−0.0193** (beats mkt) | −0.0108 | −0.0131 |
| neutral | 64 | 0.5868 | 0.5673 | +0.0195 | +0.0229 | +0.0218 |
| **full** | 2103 | 0.5519 | 0.5287 | +0.0232 | ~0 | ~0 |

**Interpretation**

1. **Away-fav: overstated home chance** (+6.2 pp bias). When the market favors the road team, Elo/ensemble still put too much mass on the home dog — consistent with **overstated Elo HFA** *on that segment* (or missing road-favorite skill that market has).
2. **Home-fav: understated home chance** (−2.0 pp). Same global HFA cannot fix both; lowering HFA helps away-fav and hurts home-fav.
3. **Ridge’s ~5.4 pt fitted HFA is a red herring for this lever.** Raising Elo toward that scale (100–120) **worsens** away-fav and full gap. Bias data rejects “Elo HFA too low globally.”
4. **Neutral HFA=0 path looks sane** (small n; gap similar to overall). Ridge actually beats market on the 64 neutral games (LL 0.552 < 0.567); Elo is the softer neutral member.
5. Asymmetry (away gap +0.035 vs home +0.019) is therefore **not** a one-knob HFA constant bug — it is a **segment skill** miss that a global HFA trade only partially papered over.

## Spike table (2023–2025 walk-forward)

Constant Elo HFA re-fit (updates + predict). Season-blocked WF pick = argmin prior-season Elo LL from the same grid (2023 → default 60; 2024/2025 → **50**).

| Candidate | Away gap | Home gap | Full ens LL | Full gap | Elo LL | Ship? |
|---|---:|---:|---:|---:|---:|---|
| elo_hfa=30 | **+0.0270** | +0.0238 | 0.5522 | +0.0236 | 0.5626 | N (hurts home/full) |
| elo_hfa=40 | +0.0290 | +0.0217 | 0.5517 | +0.0230 | 0.5612 | N |
| **elo_hfa=50** | +0.0315 | +0.0200 | **0.5515** | **+0.0229** | **0.5606** | **N (marginal)** |
| elo_hfa=60 (baseline) | +0.0346 | +0.0188 | 0.5519 | +0.0232 | 0.5606 | — |
| elo_hfa=70 | +0.0380 | +0.0180 | 0.5527 | +0.0240 | 0.5613 | N |
| elo_hfa=80 | +0.0420 | +0.0176 | 0.5539 | +0.0253 | 0.5628 | N |
| elo_hfa=90 | +0.0465 | +0.0177 | 0.5556 | +0.0270 | 0.5649 | N |
| elo_hfa=100 | +0.0514 | +0.0181 | 0.5578 | +0.0291 | 0.5677 | N |
| elo_hfa=120 | +0.0626 | +0.0203 | 0.5634 | +0.0347 | 0.5753 | N |
| wf_prior_hfa (60/50/50) | +0.0327 | +0.0198 | 0.5518 | +0.0231 | 0.5608 | N (≈ constant 50) |

### Score-time follow-ups (fit HFA=60, adjust predict only)

| Candidate | Away gap | Home gap | Full gap | Ship? |
|---|---:|---:|---:|---|
| pred_hfa=40 (fit=60) | +0.0296 | +0.0212 | +0.0229 | N (tied w/ refit 50; no code win) |
| pred_hfa=50 (fit=60) | +0.0319 | +0.0198 | +0.0229 | N |
| away_elo_pred_hfa=20 | +0.0323 | +0.0219 | +0.0242 | N |
| away_elo_edge ×1.25 | +0.0435 | +0.0198 | +0.0271 | N (bias↓ but LL↑ — overconfident) |
| away_elo_edge ×0.75 | +0.0423 | +0.0178 | +0.0256 | N |

## Ship / don’t ship

**Do not change `HFA_ELO_POINTS` / `settings.model.elo_home_field`.**

Reasons:

- Best constant (50) improves full gap by only **~0.0003** LL vs 60 — noise-scale vs early-sigma’s shipped ~0.0008, and **trades** home-fav (+0.0012 gap) for away-fav (−0.0031).
- Away-fav gap remains **+0.031** even at HFA=50; asymmetry not closed.
- Away-only Elo dampens / edge scales do not beat a mild global cut and do not clear `beat_market`.
- Raising HFA toward ridge’s ~5.4 pts is **strictly harmful** OOS.

Scorecard + spike script are the deliverable (evidence on PR #17).

## Promote gate movement

Unchanged vs post–market-prior branch state:

| Gate | Status |
|---|---|
| Pure-model OOS logloss beat market | **fail** (full gap still ~+0.023) |
| Calibration gap ≤ 0.03 | **fail** (~0.037; not re-litigated here) |
| Median CLV ≥ 0 | pass (prior evidence) |
| `market_weight_floor=0.75` / shadow | unchanged |

## Recommended next action

1. **Stop Elo/HFA constant spikes** — diagnosis says the away-fav hole is not a global HFA knob.
2. **Preferred next lever:** weeks **3–4 soft open blend** (`weight < 1.0`, not full replace) if early residual remains after W1–2 replace — already flagged on the market-prior scorecard.
3. **Promote blockers summary** (optional parallel): freeze skill spikes and write a one-pager of remaining gates (`beat_market` ~0.024, cal gap ~0.037, floor 0.75) so CORE promotion stays fail-closed with a clean checklist.
4. **Not justified yet:** logit drop / QB features (coverage still thin; prior weight spikes failed). Conference-specific HFA left untried — only revisit if a walk-forward venue/conference residual table shows a clean prior-only fit.

## Files

- `scripts/spike_hfa_away_fav.py` (new)
- `docs/scorecards/moneyline_scorecard_2023_2025_hfa_away_fav.md` (this file)
- Raw: `artifacts/spike_hfa_away_fav_2023_2025.json`, `artifacts/spike_hfa_away_fav_scoretime.json` (gitignored)
