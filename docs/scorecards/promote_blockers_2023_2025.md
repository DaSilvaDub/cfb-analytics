# Promote blockers one-pager — 2023–2025

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~04:30 ET** |
| Branch / PR | `feat/cfbd-historical-lines-2024` ([PR #17](https://github.com/DaSilvaDub/cfb-analytics/pull/17)) |
| HEAD (at write) | tip of PR branch after W3–4 open blend ship |
| Promote path | `cfb_analytics/backtest/promote.py` + `config/promotion.json` |
| Status | **`shadow`** (fail-closed). No hand-flip. |

## 1. Current status

Two-key gate (sample floor **and** OOS skill) still fails. Historical CFBD odds remain **local-only** (not on `data` branch).

| Evidence source | Overlap n | Seasons | Ens LL | Mkt LL | Gap | Notes |
|---|---:|---|---:|---:|---:|---|
| Last formal `--promote` write (`promotion.json` evidence) | **2103** | 2023–2025 | **0.5560** (platt/season) | **0.5287** | +0.0274 | Pre early-σ / open-prior ships; includes cal + CLV + blend brackets |
| Latest shipped path (spike scorecard + `artifacts/spike_open_blend_w34_*.json`) | **2270** | 2023–2025 | **0.5398** | **0.5208** | **+0.0189** | W1 unlock expands n; early σ×1.5 + W1–2 open replace + W3–4 soft w=0.75. Spike arms often skip logit for speed — directional for `beat_market`, not a fresh promote JSON rewrite |

**Bottom line:** early open levers cut the full gap ~0.027 → ~0.019; **`beat_market` still false**. Cal gap / median CLV below cite the formal promote evidence (not re-litigated on every spike).

## 2. Gate scoreboard

Encoded in `evaluate_promotion` (`promote.py`) unless noted.

| Gate | Threshold | Current | P/F | What it means |
|---|---|---|---|---|
| `n_overlap` | ≥ `min_settled_games` **1500** | 2103 (promote) / 2270 (post-unlock spike) | **PASS** | Same-game model∩close set is large enough |
| `seasons_with_overlap` | ≥ **3** | 2023, 2024, 2025 | **PASS** | Multi-season OOS window |
| **`beat_market`** | pure **calibrated** ens LL **&lt;** market LL (same-game set) | 0.5398 ≮ 0.5208 (latest); 0.5560 ≮ 0.5287 (promote) | **FAIL** | Skill to lower `market_weight_floor`. Blend is **evidence-only** — high-w blend≈market must not redefine this gate |
| **`calibration_gap`** | ≤ **0.03** (`max_abs_calibration_gap`; buckets n≥100) | **0.0370** (platt/season; was 0.0659 raw) | **FAIL** | Max \|win_rate − bucket mid\| on overlap reliability |
| **`median_clv_non_negative`** | median model-vs-close CLV ≥ 0 | **+0.000374** (**+3.7 bps**); `true` | **PASS** | CLV on **raw** ens side (not open-ML book; CFBD open ML N/A) |
| **`market_weight_floor`** | settings **0.75** (policy / live helper; **not** a P/F row in `evaluate_promotion`) | Wired into promote blend evidence + `config.market_weight_floor()` / `blend_with_market`; WF prefers w→1 | **KEEP** | Floor stays until pure model clears `beat_market` |

Failures currently written in `promotion.json`: `beat_market=False`, `calibration_gap 0.0370 > 0.03`.

## 3. Why each blocker fails

### beat_market (primary)

Root cause is **missing mid/late-season skill**, not an unset knob. Weeks 1–4 were the dominant hole (+0.05 ens–mkt); early σ + open W1–2 replace + W3–4 soft blend largely exhausted that lever (W5+ gap still **+0.0196**, n=1641 — unchanged by open blend). Full gap after ships ≈ **+0.0189**. Open-seeded early rows partly track open≈close, so they shrink headline gap without proving pure-model edge. Cite: `moneyline_scorecard_2023_2025_market_blend.md` (Track B), `…_open_blend_w34.md`, `artifacts/spike_open_blend_w34_2023_2025.json`.

### calibration_gap

Season-blocked Platt on logit(p) (`platt/season`) cut gap **0.0659 → 0.0370** without peeking. Remaining ~0.037 is mostly the **2023 identity fold** (no prior season in window) plus residual mid-bin structure; few-DOF parametric cal cannot honestly hit **0.03** on this window. Calibrated OOS logloss is slightly **worse** than raw (0.5560 vs 0.5548) — cal is not a secret beat-market path. Cite: `moneyline_scorecard_2023_2025_calibration.md`, `backtest/prob_calibrate.py`.

### market_weight_floor / blend discipline

Floor was settings-only; now applied in promote blend brackets. WF fit pins **w=1.0** for 2024/2025 (2023→floor 0.75 with no prior). High-w blend LL ≈ market, so a blend-based `beat_market` would be nearly tautological / gameable — **explicitly rejected** in `promote.py` evidence comment and `blend_gate_interpretation=pure_model_must_beat_market_before_lowering_floor`. Cite: `moneyline_scorecard_2023_2025_market_blend.md`, `models/market_blend.py`.

### median CLV (passing — keep watching)

Model-vs-close median CLV ~**+3.7 bps** on the formal promote set. Open W1–2 rows ≈ 0 CLV vs close by construction; do not treat early open seeding as CLV skill. Gate still real (`require_median_clv_non_negative`).

## 4. What we already tried

| Lever | Outcome |
|---|---|
| Early ridge σ ×1.5 weeks ≤4 | **Shipped** — small W1–4 / full gap cut; ridge LL 0.67→0.57 |
| W1–2 open-spread replace + W1 open unlock | **Shipped** — W1–2 gap ~0.053→0.011 (matched); unlock expands overlap |
| W3–4 soft open blend w=0.75 | **Shipped** — W3–4 gap 0.047→0.018; full gap ~0.023→0.019; W5+ flat |
| Elo HFA constant / WF HFA grid | **Failed / no model change** — away-fav hole is segment skill, not a global HFA knob |
| Shrinkage k↑, early ridge-weight damp, drop-logit / WF ens weights, margin-shrink toward open, bare `prior_only` week-1 | **Failed or ≤ noise** — not shipped (see early_season + market_blend + market_prior scorecards) |

## 5. What would honestly clear promote

Ranked, realistic — skill vs policy vs data ops.

1. **Skill (required for `beat_market`):** close the **weeks 5+** ens–mkt gap (~+0.02). Needs information the early-open stack does not add — e.g. multi-season QB / roster features when coverage exists, or a true as-of open **ML** (not synthetic kickoff−7d spread). Do not “tune harder” on σ / HFA / W3–4 weight.
2. **Calibration (policy or data):** either expand the promote window with more prior seasons so Platt is never identity on the first fold, **or** document a deliberate threshold move (e.g. 0.04) with Pipeline Architect sign-off — **not** peeking/isotonic on the eval slice. 0.03 is not honest on current 2023–2025 setup alone.
3. **Data ops:** publish historical `cfbd_historical` odds (+ rebuild artifacts) onto the **`data` branch** so CI / other agents can reproduce `--promote` without a local-only DB. Separate from skill; currently blocks shared verification.
4. **Floor:** lower `market_weight_floor` **only after** pure cal ens beats market OOS. Until then keep 0.75 for any live blend helper.

## 6. Explicit non-goals

- **No hand-flip** of `promotion.json` `status` → `promoted`.
- **No close-as-prior** (same-game close is the promote baseline — leakage if used as a feature).
- **No gaming `beat_market` via high-w blend** (blend≈market is evidence, not the gate).
- **Stop Elo HFA constant spikes** (diagnosed; scorecard closed).
- **Do not full-replace weeks 3–4** with open (soft blend only; W5+ untouched by design).
- **Do not force-push** the `data` branch; no CORE / live betting / Outlier scrape in this track.

## 7. Recommended next 3 actions

1. **Freeze CFB skill spikes** against promote until a *new* information source lands (true as-of open ML, or ≥3-season QB/roster coverage). Early open + σ levers are largely exhausted; remaining gap is weeks 5+ skill + cal policy.
2. **Data-ops: publish historical odds** (2023–2025 `cfbd_historical` + market rebuild) to the `data` branch under normal publish process — make promote evidence reproducible outside this box. Do not invent prices; do not force-push.
3. **Re-run full `backtest --promote` (with logit)** after (2) or any skill change; refresh `promotion.json` evidence + this scoreboard. Optionally decide cal-gap policy (keep 0.03 vs documented 0.04) **before** the next skill cycle — separately from `beat_market`.

## Pointers

| Artifact | Path |
|---|---|
| Gate code | `cfb_analytics/backtest/promote.py` |
| Live evidence blob | `config/promotion.json` |
| Formal same-game promote scorecard | `docs/scorecards/moneyline_scorecard_2023_2025_with_market.md` |
| Calibration | `docs/scorecards/moneyline_scorecard_2023_2025_calibration.md` |
| Blend / floor discipline | `docs/scorecards/moneyline_scorecard_2023_2025_market_blend.md` |
| Early σ | `docs/scorecards/moneyline_scorecard_2023_2025_early_season.md` |
| Open W1–2 | `docs/scorecards/moneyline_scorecard_2023_2025_market_prior_early.md` |
| Open W3–4 | `docs/scorecards/moneyline_scorecard_2023_2025_open_blend_w34.md` |
| HFA (no ship) | `docs/scorecards/moneyline_scorecard_2023_2025_hfa_away_fav.md` |
