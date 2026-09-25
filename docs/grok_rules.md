# Grok CFB Reasoning Heuristics & Decision Governance

*Source: Grok College Football Analysis & Late-Slate Audit (transcribed from `C:\Users\dasil\Desktop\Plans\Skip to main content.md`)*

## 1. Core Analytical Philosophy: "No Blind Favorites"

Never bet a team simply because they are a heavy favorite or carry a prestigious program name. Every recommendation must be backed by a full contextual dossier and explicit data edges.

---

## 2. The 6-Dimensional Game Dossier Framework

Before any Moneyline, Spread, Total, or Prop bet is recommended, the system must evaluate and document:

1. **Tape Analysis: Honest vs. Junk**
   - **Honest Tape**: Performance against comparable FBS/Power competition (efficiency, EPA/success rate, yards per play, scoring against top-75 defenses).
   - **Junk / Cupcake Tape**: Blowouts against FCS or bottom-10 FBS opponents (e.g., 52–0 vs MVSU, 49–3 vs Sac State). **Never use cupcake boxes as the baseline identity or offensive ceiling.** Strip out cupcake stats when projecting lines.
   - Identify each team's true offensive/defensive **floor** and **ceiling**.

2. **Position & Roster Talent Matching**
   - Grade and contrast position units:
     - **QB**: True production against real fronts, completion %, YPA, turnover-worthy plays, pressure rate.
     - **RB & OL**: Run game efficiency, rushing yards per carry, offensive line push and run-blocking grades.
     - **Front & Secondary**: Rush defense allowed YPG, defensive front 7 disruption, secondary coverage grades.
     - **Depth**: Reserve talent drop-off (crucial for late-game covers vs. backdoor covers).
   - Factor in 247/On3 Roster Talent Composite, Blue-Chip Ratio, recruiting classes, and transfer portal composite.

3. **Weather & Venue Situational Dynamics**
   - Temperature, sustained wind, wind gusts, and precipitation forecasts.
   - Stadium elevation, dome vs. outdoor, home field value (standard ~2.5 to 3.0 points).
   - Travel distance, time zone shifts (e.g. 2,000-mile cross-country flight, 10:30 PM / 11:00 PM ET kickoff tax on road favorites).

4. **Injuries & Availability**
   - Starting QB availability and continuity (e.g., backup QB in hostile environment).
   - Trench attrition: Starting OL/DL injuries severely distort expected run efficiency and pass protection.
   - Opponent injury advantages.

5. **Program Continuity & Schedule Context**
   - Head coach / coordinator continuity (e.g., Year 1 new staff vs. established system).
   - Lookahead / trap spots (e.g., marquee rivalry next week).
   - Rest disparity (bye week advantage, short turnaround).

6. **Market Devig & Edge Quantification**
   - Consensus market line and vig-free fair probability (Shin/Multiplicative).
   - Divergence between fundamental model projection and market price (spotting cupcake inflation).

---

## 3. Heuristic Decision Rules & Gating Constraints

### Rule A: The Home Power Smash Script
- **Trigger**: Home Power conference team coming off a loss/underperformance, playing a team that either lost to FCS, scored $\le 7$ against a Power team, or is a severely outmatched G5/MAC opponent.
- **Action**:
  - **Bet the OVER** on the game total (wounded Power offense rebounds and scores 40–55 points).
  - **Bet the First Half (1H) Favorite Spread** (e.g. 1H −14 to −17) to capture the smash without fourth-quarter backdoor/kneel-down risk.
  - **Do NOT default to the Under or the fat underdog.**

### Rule B: Wounded Home FBS Team Hunting First Win (The "Montana Rule")
- **Trigger**: An 0–2 or 1–2 FBS team playing at home after getting punched by tough Power opponents, facing an FCS opponent (even a top-3 ranked FCS team like Montana).
- **Action**:
  - Top-ranked FCS is **not** a shield against a desperate, physically superior FBS roster at home.
  - Take the **Home FBS Favorite Spread / 1H / OVER**. Do not take points on the FCS dog under 24 points.

### Rule C: Fat Road Dogs vs. Road Fatigue
- **Trigger**: Favorite laying $-25$ or more on the road at a late kickoff (10:30 PM / 11:00 PM ET), especially following a high-scoring or emotional game the prior week.
- **Action**:
  - **Take the Points with the Fat Underdog ($+20.0$ or higher)**.
  - Road favorites frequently pull starters, drain clock, or play sloppy late in late-night cross-country spots (e.g., Sacramento State +27.5, Tulane +20.5).

### Rule D: Thin Dogs (+1.5 to +6.0) Are Traps
- **Trigger**: A thin underdog priced between $+1.5$ and $+6.0$, especially in conference or rivalry games.
- **Action**:
  - Thin dogs fail at high rates. Do not play thin underdogs on gut feeling or small common-opponent edges.
  - **Pass** unless there is a confirmed starting QB injury, a $>20$-point common-opponent tape differential, or major trench matchup mismatch.

### Rule E: High-Scoring Conference Matchups
- **Trigger**: Two Power or capable G5 teams that both score $30+$ PPG against honest competition (e.g., Purdue vs. UCLA).
- **Action**:
  - **Target the OVER first**, side second. Late-night conference shootouts easily clear mid-50 totals.

### Rule F: First Half (1H) Preferred Over Full-Game Blowout Covers
- **Trigger**: Massive favorite ($-24$ to $-35$).
- **Action**:
  - Late-game substitutions, garbage-time drives, and 4th-quarter kneels kill $-30+$ covers.
  - Target the **First Half spread** (e.g., 1H $-14$ to $-17$) where starters play with maximum tempo and intensity.

---

## 4. Market Misprice Detection

1. **Spread Misprices**: When a team's market spread is inflated by $4+$ points due to a cupcake blowout (e.g., laying 6 on the road based on a 49-3 win over FCS).
2. **Game Totals**: Capitalize on market over-reactions where a low offensive output against an elite defense drags down a game total against a porous defense.
3. **Team Props (Rush/Passing Yards)**:
   - Identify run-heavy identities facing weak run fronts (e.g., team rush OVER).
   - Fade passing totals for game-manager QBs on heavy favorite teams (under passing yards / over rush yards).

---

## 5. Output Card Format

Every candidate play must emit a structured dossier:
```
MATCHUP: [Away] at [Home] | Kickoff: [Time ET] | Network: [TV]
MARKET: [Moneyline / Spread / Total / Team Prop]
RECOMMENDED PLAY: [Side / Line] (Confidence: [X.X/10] - [TIER])

1. Tape Evaluation: Honest vs Junk
2. Position & QB Edge Breakdown
3. Weather, Venue & Travel Factors
4. Injuries & Trench Health
5. Program Structure & Situational Spot
6. Mathematical Edge vs Market Consensus
7. What NOT to Bet (Contra-Indications)
```
