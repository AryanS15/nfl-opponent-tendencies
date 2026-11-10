# nfl-opponent-tendencies

An NFL player scouting dashboard exploring WR, TE, and RB alignment, movement, and
passing-play responsibilities.

It helps a defense understand how an opposing wide receiver, tight end, or running back
*actually behaves* on passing plays — beyond their roster position: where they line up
(wide / slot / inline / backfield), whether they run a route or pass-block, how they
move after the snap, and how those tendencies shift by down, distance, formation, and
opponent.

> **Data scope:** all findings come from a **sample of passing dropbacks from Weeks 1–8
> of the 2021 NFL season** (the 2022 Big Data Bowl regional dataset). These are tendencies
> *on sampled dropbacks* — **not** all offensive plays, and the data contains no designed
> run plays. Treat everything as early-season 2021 passing-down behavior, not full-season
> truth.

---

## Features

- **Single-player scouting** — alignment mix, route-vs-block rate, situational splits, and
  movement, each with sample sizes and 95% confidence intervals.
- **Opponent filter** — view a player across all defenses faced, or drill into one matchup
  and compare it against all other opponents (with percentage-point differences).
- **League-wide view** — baselines by position, situation, and data-driven "actual role"
  groups (e.g. slot WR, blocking-heavy TE), with a toggle to weight each play equally or
  each player equally.
- **Pre-snap route-vs-block model** — predicts responsibility using only information
  available before the snap, evaluated on held-out weeks against simple baselines.
- **Play diagrams** — recognizable football-field visualizations of real tracked plays.

Observed facts, model predictions, and coaching hypotheses are kept visually distinct
throughout. The tool never claims a coverage or assignment is proven effective.

---

## Project layout

```
.
├── scout/                 # the dashboard + pipeline (main app)
│   ├── dashboard.py       # Streamlit UI
│   ├── profiles.py        # descriptive player + league aggregations
│   ├── build_features.py  # builds the player-play feature table from tracking
│   ├── model_route_block.py  # trains/evaluates the route-vs-block model
│   ├── audit_tracking.py  # validates tracking coverage across all games
│   ├── playviz.py         # play-diagram rendering
│   ├── requirements.txt
│   ├── README.md          # detailed methodology & limitations
│   └── outputs/           # generated artifacts (git-ignored; regenerate locally)
├── analysis/              # exploratory scripts used to validate the data
└── README.md              # you are here
```

The dataset itself is **not** included in this repository (see below).

---

## Live demo (static site)

A backend-free, static version of the dashboard runs on GitHub Pages — no Python server
required. It loads a small `index.json` plus one JSON file per player on demand, so the
whole tracking dataset is never shipped to the browser.

**Live demo:** _(added after first deployment — see the "Static web build" section)_

The Streamlit app (below) remains the full-featured local version; the static site mirrors
its validated calculations and model outputs.

---

## Setup

Requires Python 3.9+.

```bash
python3 -m pip install -r scout/requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`, `pyarrow`,
`joblib`.

### Get the dataset (not redistributed here)

This project is built on the **NFL Big Data Bowl 2022 (NYU regional) dataset**, which is
**not** included in this repo — please download it yourself and do not redistribute it.

1. Obtain the data from its source:
   <https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data>
   (or the corresponding Kaggle Big Data Bowl 2022 competition page).
2. Place it so the CSVs sit at this path relative to the repo root:

   ```
   nfl-big-data-bowl-regional-event-data/
   └── data/
       ├── games.csv
       ├── players.csv
       ├── plays.csv
       ├── pffScoutingData.csv
       └── tracking/
           └── tracking_<gameId>.csv   (122 files, Weeks 1–8)
   ```

This folder is git-ignored, so it will never be committed or pushed.

---

## Run

```bash
# 1. (optional) audit tracking coverage across all games
python3 scout/audit_tracking.py

# 2. build the player-play feature table (~40s; reads all tracking)
python3 scout/build_features.py            # or: --games 8 for a quick subset

# 3. train + evaluate the route-vs-block model
python3 scout/model_route_block.py

# 4. launch the dashboard
python3 -m streamlit run scout/dashboard.py
```

Then open <http://localhost:8501>. Steps 2–3 generate the files the dashboard reads
(`scout/outputs/`); run them once after downloading the data.

---

## Static web build (GitHub Pages)

The `web/` folder is a self-contained HTML/CSS/JS site that works with no backend. It
reads JSON exported from the same `profiles.py` calculations and model artifacts the
Streamlit app uses.

### Export the data and build the site

```bash
# generate scout/outputs/* first (see "Run" above), then:
python3 scout/export_web.py            # writes web/data/{index.json, players/<id>.json}
# quick subset while developing:
python3 scout/export_web.py --limit 15
```

### Preview locally over HTTP

Open it through a local server (not `file://`, so `fetch()` works):

```bash
cd web && python3 -m http.server 8080
# then open http://localhost:8080/
```

### How it's deployed

`.github/workflows/pages.yml` publishes the pre-built `web/` directory to GitHub Pages
on every push to `main` that touches `web/`. The workflow does **not** run Python or read
the dataset — it just uploads the committed static files. All asset paths are relative
(`./styles.css`, `./app.js`, `./data/...`) so the site works under the
`/<repo>/` Pages subpath.

### What the static site ships (data redistribution)

The source NFL/PFF tracking dataset is **not** redistributed. The site contains only:
- **derived aggregates** (route/alignment rates, movement means, counts, model metrics), and
- a **small number of per-player play-coordinate excerpts** used only to draw example
  diagrams (a handful of plays per player, not the tracking dataset).

This mirrors the approach of comparable public Big Data Bowl demos. The raw
`tracking_*.csv` files are never exported or committed.

---

## How it works (short version)

- **Unit of analysis** is one player on one play — tracking frames are never counted as
  independent samples.
- **Alignment** comes from PFF's snap-time label, validated against tracking geometry.
- **Route vs block** comes from PFF's role label; it is the model's target, never an input.
- **Model** uses pre-snap inputs only, trains on Weeks 1–5, and evaluates on Weeks 6–8,
  with a leakage-free historical feature. On held-out data it beats a
  player-history baseline on ROC-AUC (≈0.77 vs 0.65).

Full methodology, the tracking-coverage audit results, and the reasoning behind each
derived field are in [`scout/README.md`](scout/README.md).

---

## Limitations

- **Sampled passing dropbacks, Weeks 1–8 of 2021 only.** No run plays; the data cannot
  establish run-vs-pass tendencies, only route-vs-block *conditional on a dropback*.
- Good for method-building and matchup prep, **not** for definitive season-long rankings.
- Pre-snap tracking window is only ~0.5s, so motion usage is a lower bound.
- Catch/arrival events are tagged on only ~4% of plays, so separation/arrival metrics are
  intentionally omitted (no fixed-offset imputation).
- Each player faced each defense in ~one game, so single-opponent samples are small and
  confounded with game script and teammates — matchup differences are hypotheses, not
  proven effects of the defense.
- No targeted-receiver ID in the raw data, so per-receiver receiving yardage isn't modeled.

---

## Data & attribution

The source tracking and scouting data are provided by the NFL Next Gen Stats team and
Pro Football Focus via the NFL Big Data Bowl. This repository contains **only original
analysis code** and does not include or redistribute any of that data. Please obtain it
from the official source linked above and follow its terms.
