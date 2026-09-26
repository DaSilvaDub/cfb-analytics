# Moneyline scorecard — residual ens–mkt gap segmentation (2023–2025)

| Field | Value |
|---|---|
| Generated | **2026-09-26 ~06:00 ET** |
| Branch / PR | `feat/cfbd-historical-lines-2024` ([PR #17](https://github.com/DaSilvaDub/cfb-analytics/pull/17)) |
| Script | `scripts/segment_residual_gap.py` |
| Raw | `artifacts/segment_residual_gap_2023_2025.json` (gitignored) |
| DB | `origin/data` tip `c8006e9…` (62720 `cfbd_historical`) |
| Promote | stays **`shadow`** — segmentation only; no model ship |

## Pipeline under test

Shipped production levers only:

- Early ridge σ ×1.5 (weeks ≤4)
- W1–2 open-spread replace (w=1) + W1 unlock
- W3–4 soft open blend w=0.75
- Spike mix skips logit (ridge/elo renorm) for speed — full gap matches promote raw within ~0.001

Close ML = scoring + segmentation baseline only (never a same-game feature).

## Full same-game set

| Metric | Value |
|---|---:|
| n_overlap | **2270** |
| ens LL (raw) | 0.5398 |
| mkt LL | 0.5208 |
| **gap** | **+0.0189** |
| ens bias (p−y) | +0.0035 |

Aligns with formal promote cal ens gap **+0.0196** and open-blend spike **+0.0189**.

## Where the residual +0.02 lives

### By week

| Slice | n | ens LL | mkt LL | gap | gap_mass |
|---|---:|---:|---:|---:|---:|
| weeks_1_2 | 309 | 0.4893 | 0.4729 | +0.0164 | 0.0022 |
| weeks_3_4 | 320 | 0.4889 | 0.4708 | +0.0181 | 0.0025 |
| weeks_1_4 | 629 | 0.4891 | 0.4718 | +0.0172 | 0.0048 |
| **weeks_5_plus** | **1641** | 0.5592 | 0.5396 | **+0.0196** | **0.0142** |
| **weeks_5_8** | **638** | 0.5790 | 0.5471 | **+0.0319** | **0.0090** |
| weeks_9_12 | 657 | 0.5586 | 0.5391 | +0.0195 | 0.0056 |
| weeks_13_plus | 346 | 0.5236 | 0.5267 | **−0.0030** | 0 |
| week_5 | 159 | 0.5431 | 0.4949 | **+0.0482** | 0.0034 |
| week_6 | 147 | 0.5805 | 0.5506 | +0.0300 | 0.0019 |

Early-open cut W1–4; **W5–8 (esp W5)** holds the remaining skill hole. Late season (W13+) actually beats market on this window.

### By favorite side (close ML)

| Slice | n | gap | ens bias | gap_mass |
|---|---:|---:|---:|---:|
| away_fav | 814 | +0.0269 | **+0.0557** | 0.0097 |
| home_fav | 1373 | +0.0165 | −0.0264 | 0.0100 |
| pickem | 83 | −0.0197 | −0.0142 | 0 |
| **w5plus_away_fav** | **624** | **+0.0309** | **+0.0618** | 0.0085 |
| w5plus_home_fav | 950 | +0.0152 | −0.0220 | 0.0064 |

Away-favorite overstatement persists (HFA constant spike already closed — do not re-open).

### By open-spread magnitude / market edge

| Slice | n | gap | gap_mass |
|---|---:|---:|---:|
| spread_abs_0_3 | 359 | +0.0201 | 0.0032 |
| **spread_abs_3_7** | **629** | **+0.0333** | **0.0092** |
| spread_abs_7_14 | 658 | −0.0022 | 0 |
| spread_abs_14_plus | 624 | +0.0260 | 0.0071 |
| edge_5_10pp | 260 | +0.0275 | 0.0032 |
| edge_10_15pp | 233 | +0.0268 | 0.0028 |
| edge_25pp_plus | 1075 | +0.0187 | 0.0089 |

Medium spreads (3–7 pts) are the densest non-week residual. Blowouts (14+) still soft; mid-range (7–14) nearly break-even.

### Conference / neutral

| Slice | n | gap | gap_mass |
|---|---:|---:|---:|
| conf_game | 1624 | +0.0176 | 0.0126 |
| non_conf_game | 646 | +0.0223 | 0.0064 |
| neutral | 79 | +0.0109 | 0.0004 |

Conference flag is mostly a size effect (most games are conf); not a distinct lever.

## Ranked residual (top 3 actionable)

1. **Weeks 5–8 skill** (mass 0.0090; W5 alone +0.048) — binding unlock for `beat_market`.
2. **Away-fav × W5+** (gap +0.031, bias +6.2 pp) — segment skill, not global HFA.
3. **Medium open spreads 3–7** (gap +0.033, mass 0.0092) — toss-up band where market has edge.

## Spike / freeze

**Skill freeze.** No new honest lever is OOS-promising without new data:

- Extending open blend into W5+ = more market borrow (rejected for promote skill).
- HFA / early-open / σ / drop-logit / WF ens weights already tried.
- True open ML unavailable from CFBD; multi-season QB not on historical promote path.

See `promote_blockers_2023_2025.md` §11.
