# Opponent Offensive Tendencies

A local dashboard that helps a defense understand how opposing WRs, TEs, and RBs
line up, move, and take on different roles on passing plays — beyond their official
position: where they line up, whether they run routes or block, how they move, and
how those tendencies shift by game situation.

Built on the **2022 NFL Big Data Bowl (NYU regional)** data: 2021 season, **Weeks 1–8**,
122 games, **passing dropbacks only**.

---

## Quick start

```bash
# from the workspace root
python3 -m pip install -r scout/requirements.txt

# 1. audit tracking coverage (optional, ~1 min)
python3 scout/audit_tracking.py

# 2. build the player-play feature table (~40s, reads all tracking)
python3 scout/build_features.py

# 3. train + evaluate the route-vs-block model
python3 scout/model_route_block.py

# 4. launch the dashboard
streamlit run scout/dashboard.py
```

`build_features.py` accepts `--games N` to process only the first N games for a quick
end-to-end test.

Outputs land in `scout/outputs/` (`player_play.parquet/.csv`, `tracking_audit.json`,
`model_results.json`, `route_block_gbm.joblib`).

---

## What each script does

| File | Purpose |
|---|---|
| `audit_tracking.py` | Validates tracking across **all 122 games**: snap coverage, pre-snap window length, event-tag availability. Produces the limitations that shape every downstream choice. |
| `build_features.py` | Core reusable pipeline → **one row per player-play** (WR/TE/RB/FB) with play context, PFF role/alignment labels, and tracking-derived movement features. Validates alignment buckets against tracking geometry. |
| `profiles.py` | Descriptive player profiles: alignment freq, route-vs-block, situational splits, motion, movement, weekly trend — all with sample sizes and Wilson confidence intervals. |
| `model_route_block.py` | Pre-snap **route-vs-block** classifier with temporal split, leakage-safe historical feature, and baseline comparison. |
| `playviz.py` | Plots player tracks for a representative play. |
| `dashboard.py` | Streamlit UI tying it together with team/player filters. |

---

## Method notes (and why they matter)

### Unit of analysis
Everything is counted per **player-play** `(gameId, playId, nflId)`, never per tracking
frame. The 42,698 observations are real opportunities, not inflated by the ~42 frames
each play contains.

### Alignment (wide / slot / inline / backfield)
Taken from PFF's snap-time `pff_positionLinedUp` label, bucketed. **Validated** against
tracking lateral position at the snap frame:

| bucket | mean distance from field center |
|---|---|
| wide | 13.9 yd |
| slot | 7.3 yd |
| inline | 5.3 yd |
| backfield | 3.4 yd |

The ordering is exactly as expected, so we trust the labels. Unmappable labels are kept
as `unknown` rather than guessed.

### Route vs block
From PFF `pff_role` (`Pass Route` / `Pass Block`). Used as ground truth for profiles and
as the **model target** — never as a model input.

### Tracking limitations (from the full-game audit)
- **Snap event** present on **99.7%** of plays → reliable anchor.
- **Pre-snap window is only ~0.5s** (median 5 frames). So "pre-snap motion" is measured
  only over that sliver and reported as a **lower bound** of true motion usage.
- **Pass-forward (release)** present on **88.2%** → used as the end of the post-snap
  movement window.
- **Arrival / catch events on only ~4.4%** of plays → we **do not** impute a fixed catch
  time, and arrival/separation metrics are intentionally omitted from the core tool.
- Nearest defender is **not** treated as the assigned coverage defender.
- `playResult` (net yards) is **not** treated as an individual's receiving yards.

### Model: route vs block on a dropback
- **Inputs (pre-snap only):** official position, alignment bucket, down, yards-to-go,
  offensive formation, offensive personnel, field position, and a **player historical
  route rate computed from training weeks only** (no leakage).
- **Split:** train weeks 1–5, test weeks 6–8. A play belongs to exactly one week, so all
  of a play's observations stay in the same split.
- **Scope:** modeled on **TE/RB/FB** (WRs run routes ~99.8% of the time → trivial).
- **Class balance:** ~84% route / 16% block.
- **Results (test weeks 6–8):** GBM ROC-AUC **0.767** vs best baseline (player-history)
  **0.650** → **+0.117 lift**; well-calibrated predicted probabilities. Accuracy alone is
  misleading given the imbalance, so AUC/calibration/PR are reported together.
- **Scope statement:** this predicts responsibility **conditional on a passing dropback**.
  The dataset contains only dropbacks, so it **cannot** establish run-vs-pass tendencies.

### Separation of claims
The dashboard keeps three things visually distinct:
1. **Observed** — measured facts with sample sizes / CIs.
2. **Model** — pre-snap predictions with performance vs baselines.
3. **Hypotheses** — plain-language defensive considerations, explicitly labeled for
   coaches to assess. The tool never claims a coverage or assignment "will work," and
   warns against acting on small-sample cells.

---

## External data

No external data is required — the tool is fully functional on the provided dataset.
The Big Data Bowl `gameId`/`playId` merge to nflverse's `old_game_id`/`play_id` (per the
dataset README) if you later want real target IDs, air yards, YAC, or EPA to extend the
receiving analysis. That join is **not** needed for anything this tool currently does.

---

## Known limitations (summary)
- 8 weeks of one season — good for method-building, not for definitive season rankings.
- Passing dropbacks only; no run plays, so no run/pass tendency modeling.
- ~0.5s pre-snap window limits motion analysis.
- No targeted-receiver ID in the raw data; per-receiver receiving yardage is not available.
- Tendencies are not opponent-adjusted; strength of opposing coverage is not controlled.
