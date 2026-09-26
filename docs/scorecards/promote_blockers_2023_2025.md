# Promote blockers one-pager — 2023–2025

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~04:30 ET** (re-promote §8 at **~04:47 ET**) |
| Branch / PR | `feat/cfbd-historical-lines-2024` ([PR #17](https://github.com/DaSilvaDub/cfb-analytics/pull/17)) |
| HEAD (at write) | tip of PR branch after W3–4 open blend ship |
| Promote path | `cfb_analytics/backtest/promote.py` + `config/promotion.json` |
| Status | **`shadow`** (fail-closed). No hand-flip. |

## 1. Current status

Two-key gate (sample floor **and** OOS skill) still fails. Historical CFBD odds remain **local-only** (not on `data` branch).

| Evidence source | Overlap n | Seasons | Ens LL | Mkt LL | Gap | Notes |
|---|---:|---|---:|---:|---:|---|
| Prior formal `--promote` (pre early-open ships) | **2103** | 2023–2025 | **0.5560** (platt/season) | **0.5287** | +0.0274 | Historical baseline in `promotion.json` before W1–4 levers |
| Spike path (`artifacts/spike_open_blend_w34_*.json`) | **2270** | 2023–2025 | **0.5398** | **0.5208** | **+0.0189** | Directional; often skip_logit |
| **This re-promote** (`promotion.json` @ 2026-09-26T08:47:15Z) | **2270** | 2023–2025 | **0.5404** | **0.5208** | **+0.0196** | Full logit + platt/season; see §8. Historical odds still **not** on `data` branch |

**Bottom line:** early open levers cut the full gap ~0.027 → ~0.019; formal re-promote confirms **`beat_market` still false** (ens 0.5404 ≮ mkt 0.5208). Status **`shadow`**. Data-branch publish of `cfbd_historical` is still blocked (force-push-only path) — §8.1.

## 2. Gate scoreboard

Encoded in `evaluate_promotion` (`promote.py`) unless noted.

| Gate | Threshold | Current | P/F | What it means |
|---|---|---|---|---|
| `n_overlap` | ≥ `min_settled_games` **1500** | **2270** (re-promote §8) | **PASS** | Same-game model∩close set is large enough |
| `seasons_with_overlap` | ≥ **3** | 2023, 2024, 2025 | **PASS** | Multi-season OOS window |
| **`beat_market`** | pure **calibrated** ens LL **&lt;** market LL (same-game set) | **0.5404 ≮ 0.5208** (re-promote §8; gap +0.0196) | **FAIL** | Skill to lower `market_weight_floor`. Blend is **evidence-only** — high-w blend≈market must not redefine this gate |
| **`calibration_gap`** | ≤ **0.03** (`max_abs_calibration_gap`; buckets n≥100) | **0.0640** (re-promote §8 platt/season; prior formal blob had 0.0370) | **FAIL** | Max \|win_rate − bucket mid\| on overlap reliability |
| **`median_clv_non_negative`** | median model-vs-close CLV ≥ 0 | **+0.0199** (**+199 bps**); `true` (re-promote §8) | **PASS** | CLV on **raw** ens side (not open-ML book; CFBD open ML N/A) |
| **`market_weight_floor`** | settings **0.75** (policy / live helper; **not** a P/F row in `evaluate_promotion`) | Wired into promote blend evidence + `config.market_weight_floor()` / `blend_with_market`; WF prefers w→1 | **KEEP** | Floor stays until pure model clears `beat_market` |

Failures currently written in `promotion.json`: `beat_market=False`, `calibration_gap 0.0640 > 0.03` (re-promote §8).

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

---

## 8. Post-publish attempt / re-promote — 2026-09-26 ~04:47 ET

| Field | Value |
|---|---|
| Actor | Pipeline Architect (executor) |
| Feature HEAD basis | `9574cfb` + this commit |
| Historical odds on `data` branch? | **NO** — publish **skipped** (see §8.1) |
| Promote DB | local `data/cfb.sqlite3` (same box DB used for early-open ships; `cfbd_historical` 2023–2025 present) |
| Command | `python -m cfb_analytics.cli backtest --start-year 2023 --end-year 2025 --historical-cfbd-min-books 1 --promote` |
| `promotion.json` status | **`shadow`** (fail-closed; no hand-flip) |
| Evaluated (UTC) | 2026-09-26T08:47:15Z (~04:47 ET) |

### 8.1 Data-branch publish — blocked (force-push is the only path)

**Map:** The only write path for `origin/data` is `.github/workflows/daily-ingest.yml` → step **Publish database**: orphan `git init`, single commit of `cfb.sqlite3.gz` + README, then **`git push --force` to `data:data`**. README and the CFB Data Fixer gzip investigation (`/workspace/cfb-publish-fix/REPORT.md` / PR #14) document the same design. There is **no** fast-forward / subtree / LFS publish. `backfill-lines` is **not** wired into the workflow (only optional `backfill-cfbd` / `backfill-elo`).

**Delta (local vs `origin/data` as of fetch 2026-09-26):**

| | Local box DB | `origin/data` (`cfb.sqlite3.gz`) |
|---|---:|---:|
| Raw / gzip size | ~87 MB / ~31.6 MB gzip | ~63 MB / ~23.4 MB gzip |
| `odds_snapshots` `cfbd` | 79432 | 79432 |
| `odds_snapshots` `cfbd_historical` | **62720** (2023:17928, 2024:21642, 2025:23150) | **0** |
| `market_consensus` seasons | 2023–2026 | **2026 only** |
| Max live `cfbd` `captured_utc` | 2026-09-25T15:42:13Z | same |

Gzip of the local DB (~31.6 MB) is **under** GitHub’s 50 MB soft limit — size is not the blocker. The brief **HARD** rule forbids this agent from force-pushing `data`. Blindly replacing `data` with the local file would also race any newer Daily ingest after this snapshot.

**Ops note for Daniel / CFB Data Fixer (exact safe sequence):**

```bash
# A) Restore live data branch (do not start from a stale local-only copy)
git fetch origin data
mkdir -p data
git show origin/data:cfb.sqlite3.gz | gunzip > data/cfb.sqlite3
python -m cfb_analytics.cli init-db

# B) Add historical odds onto that restored DB (needs CFBD_API_KEY + CFB_HTTP_MODE=live)
python -m cfb_analytics.cli backfill-lines --year 2023 --rebuild-market --historical-cfbd-min-books 1
python -m cfb_analytics.cli backfill-lines --year 2024 --rebuild-market --historical-cfbd-min-books 1
python -m cfb_analytics.cli backfill-lines --year 2025 --rebuild-market --historical-cfbd-min-books 1

# C) Sanity: expect ~62k cfbd_historical rows; market_consensus for 2023–2025 + 2026

# D) Publish via the sanctioned orphan force-push (Data Fixer / CI only).
#    Mirror daily-ingest.yml "Publish database" (WAL checkpoint → gzip -6 → orphan
#    commit → git push --force data:data), OR extend the workflow with a
#    workflow_dispatch input (e.g. backfill_lines_years) and:
#      gh workflow run "Daily ingest" -f backfill_lines_years=2023,2024,2025
#    Do NOT invent a non-force-push path; none exists by design.

# E) Verify
git fetch origin data
git show origin/data:cfb.sqlite3.gz | gunzip > /tmp/cfb_verify.sqlite3
# confirm source=cfbd_historical counts by season; join rates sane for promote
```

Optional hardening: add `backfill-lines` to `daily-ingest.yml` `workflow_dispatch` so publish stays in Actions and does not require a laptop force-push.

### 8.2 Gate scoreboard after full `--promote` re-run

| Gate | Threshold | This run | P/F |
|---|---|---|---|
| `n_overlap` | ≥ 1500 | **2270** (n_full 2432; skip no-mkt 162) | **PASS** |
| `seasons_with_overlap` | ≥ 3 | 2023, 2024, 2025 | **PASS** |
| **`beat_market`** | cal ens LL **&lt;** mkt LL | ens **0.5404** ≮ mkt **0.5208** (gap **+0.0196**) | **FAIL** |
| **`calibration_gap`** | ≤ 0.03 | **0.0640** | **FAIL** |
| **`median_clv_non_negative`** | ≥ 0 | **+0.0199** (+199.0 bps) | **PASS** |
| `market_weight_floor` | 0.75 policy | WF w: 2023=0.75, 2024=0.95, 2025=1.00; blend LL 0.5211 ≈ mkt | **KEEP** |

Failures written: `beat_market=False`, `calibration_gap 0.0640 > 0.03`. Status remains **`shadow`**.

### 8.3 Notes vs prior formal promote evidence

- Overlap **2103 → 2270** matches W1 unlock / early-open path (spike scorecards), not a data-branch change.
- Ens–mkt gap **~+0.0196** aligns with post–W3–4-blend spike (~+0.0189); **`beat_market` still false**.
- Cal gap **0.037 → 0.064** on this full promote (platt/season): worse than the pre-open formal blob — treat the **0.064** figure as current gate truth; do not cherry-pick the older 0.037.
- Median CLV **+3.7 bps → +199 bps**: open-seeded early rows and larger overlap change the CLV distribution; gate still pass-only (non-negative). Do not read as new mid-season skill.
- Skill spikes remain **frozen**; no model knobs changed in this pass.

### 8.4 Next human / ops step

1. **CFB Data Fixer (or Daniel):** run §8.1 sequence (or workflow extension) so `cfbd_historical` lands on `origin/data` without this agent force-pushing.
2. After publish: anyone can restore `data` and reproduce `--promote`; optional CI comment refresh.
3. Skill track stays frozen until a **new information** source (true as-of open ML, multi-season QB/roster, etc.) or a documented cal-gap policy decision.

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
