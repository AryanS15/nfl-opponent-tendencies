"""
Export a compact, browser-ready dataset for the static (GitHub Pages) version of
the Opponent Offensive Tendencies dashboard.

It reuses the SAME validated calculations as the Streamlit app (profiles.py) and
the SAME model outputs (route_block_gbm.joblib / hist_rates.joblib / model_results.json),
so the static site never re-derives anything.

What it writes (into web/data/):
  index.json                      small manifest: teams -> players, scope, model summary,
                                   league baselines, role definitions
  players/<nflId>.json            one file per eligible player, loaded ON DEMAND:
                                     - all-opponents profile (default view)
                                     - per-opponent profiles + matchup comparison
                                     - situational tables, movement, takeaways
                                     - held-out prediction (model splits preserved)
                                     - a few example-play coordinate EXCERPTS for diagrams

DATA REDISTRIBUTION NOTE:
  We export only DERIVED aggregates (rates, means, counts) plus a SMALL number of
  per-player play coordinate excerpts needed to draw example diagrams. We do NOT
  export the raw tracking dataset. The source NFL/PFF data is not redistributed.

Run:  python3 scout/export_web.py            # all eligible players
      python3 scout/export_web.py --limit 15 # quick subset for testing
      python3 scout/export_web.py --plays 2  # example plays per player (default 2)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import profiles as P
import playviz as V

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"
WEB_DATA = ROOT / "web" / "data"
MIN_PLAYS = 20


def clean(obj):
    """Make an object JSON-safe: NaN/inf -> None, numpy -> python, round floats."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 3)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [clean(v) for v in obj.tolist()]
    if pd.isna(obj) if np.isscalar(obj) else False:
        return None
    return obj


def df_records(df):
    return clean(df.to_dict("records")) if df is not None and not df.empty else []


def profile_to_json(prof):
    """Serialize a player_profile() dict (contains DataFrames) to plain JSON."""
    if prof is None:
        return None
    return {
        "meta": clean(prof["meta"]),
        "overall_role": clean(prof["overall_role"]),
        "near_constant_route": bool(prof["near_constant_route"]),
        "alignment": df_records(prof["alignment"]),
        "route_by_alignment": df_records(prof["route_by_alignment"]),
        "route_by_down": df_records(prof["route_by_down"]),
        "route_by_dd": df_records(prof["route_by_dd"]),
        "route_by_formation": df_records(prof["route_by_formation"]),
        "redzone": clean(prof["redzone"]),
        "motion": clean(prof["motion"]),
        "movement": clean(prof["movement"]),
        "depth_dist": df_records(prof.get("depth_dist")),
        "weekly_trend": clean(prof["weekly_trend"]),
        "holdout": clean(prof["holdout"]),
    }


def example_play_json(df, nflId, which, max_frames=60):
    """Return a compact coordinate excerpt for one example play (typical/unusual).
    Only the observed snap->endpoint window; downsampled player paths."""
    typ, unu = V.pick_example_plays(df, nflId)
    pick = typ if which == "typical" else unu
    if pick is None:
        return None
    tr = V.load_play_tracking(pick["gameId"], pick["playId"])
    if tr is None or tr.empty:
        return None
    snap_f = V._snap_frame(tr)
    end_f, end_ev = V._endpoint(tr)
    if end_f < snap_f:
        end_f = int(tr.frameId.max())
    win = tr[(tr.frameId >= snap_f) & (tr.frameId <= end_f)].copy()
    play_dir = win.playDirection.dropna().iloc[0] if win.playDirection.notna().any() else "right"
    ball_snap = tr[(tr.frameId == snap_f) & (tr.team == "football")]
    if len(ball_snap):
        los_x = float(ball_snap.x.iloc[0])
    else:
        fp = tr[(tr.frameId == snap_f) & (tr.nflId == nflId)]
        los_x = float(fp.x.iloc[0]) if len(fp) else float(win.x.median())
    off_team = None
    if (tr.nflId == nflId).any():
        off_team = tr.loc[tr.nflId == nflId, "team"].iloc[0]

    # build per-entity tracks (downsample to keep JSON small)
    tracks = []
    frames_sorted = sorted(win.frameId.unique())
    step = max(1, len(frames_sorted) // max_frames)
    keep = set(frames_sorted[::step]) | {frames_sorted[-1]}
    for nid, d in win.groupby("nflId"):
        d = d[d.frameId.isin(keep)].sort_values("frameId")
        is_ball = d.team.iloc[0] == "football"
        kind = "ball" if is_ball else ("focus" if nid == nflId else
               ("offense" if (off_team and d.team.iloc[0] == off_team) else "defense"))
        pts = [[round(float(x), 2), round(float(y), 2)] for x, y in zip(d.x, d.y)]
        if len(pts) >= 2:
            tracks.append({"kind": kind, "pts": pts})
    return {
        "which": which, "gameId": int(pick["gameId"]), "playId": int(pick["playId"]),
        "align_bucket": pick["align_bucket"], "depth": pick["depth"], "why": pick["why"],
        "play_direction": play_dir, "los_x": round(los_x, 1),
        "end_event": end_ev,
        "window_frames": int(end_f - snap_f + 1),
        "window_seconds": round((end_f - snap_f + 1) / 10.0, 1),
        "tracks": tracks,
    }


def matchup_to_json(df, nflId, opponent):
    mc = P.matchup_comparison(df, nflId, opponent)
    if mc is None:
        return None
    return {
        "opponent": mc["opponent"], "n_vs": mc["n_vs"], "n_other": mc["n_other"],
        "n_games_vs": mc["n_games_vs"], "weeks_vs": clean(mc["weeks_vs"]),
        "small_sample": mc["small_sample"], "min_cell": mc["min_cell"],
        "comparisons": df_records(mc["comparisons"]),
    }


def main(limit=None, plays=2):
    df = P.load()
    baselines = P.position_baselines(df)
    elig = P.eligible_players(df, MIN_PLAYS)
    model = joblib.load(OUT / "route_block_gbm.joblib")
    hist = joblib.load(OUT / "hist_rates.joblib")
    model_res = json.loads((OUT / "model_results.json").read_text())
    audit = json.loads((OUT / "tracking_audit.json").read_text())

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    (WEB_DATA / "players").mkdir(exist_ok=True)

    if limit:
        elig = elig.head(limit)

    # ---- index.json ----
    teams = {}
    for _, r in elig.sort_values(["team", "officialPosition", "n"],
                                 ascending=[True, True, False]).iterrows():
        teams.setdefault(r.team, []).append({
            "nflId": int(r.nflId), "name": r.displayName,
            "position": r.officialPosition, "plays": int(r.n),
        })

    # league baselines by position (player-weighted = fair baseline)
    lbp = P.league_by_position(df, MIN_PLAYS)
    league_pos = {}
    for pos, v in lbp.items():
        league_pos[pos] = {
            "route_pct": v["route_rate"]["per_player"],
            "align": {k: round(val, 1) for k, val in v["align_player"].items()},
            "mean_depth_yd": v["depth"]["per_player"],
            "mean_top_speed": v["top_speed"]["per_player"],
        }

    index = {
        "scope": {
            "season": 2021, "weeks": [1, 8], "games": 122,
            "label": "2021 NFL · Weeks 1–8 · Sampled passing dropbacks only",
            "pct_plays_with_arrival_event": audit.get("pct_plays_with_arrival_event"),
        },
        "model": {
            "gbm_auc": model_res["gbm"]["roc_auc"],
            "baseline_player_hist_auc": model_res["baseline_player_hist"]["roc_auc"],
            "train_weeks": hist["train_weeks"], "test_weeks": hist["test_weeks"],
            "class_note": "~84% route / 16% block on modeled TE/RB/FB plays",
        },
        "model_results": clean(model_res),
        "baselines": clean(baselines),
        "league_by_position": clean(league_pos),
        "role_defs": P.ROLE_DEFS,
        "n_players": int(elig.nflId.nunique()),
        "teams": teams,
    }
    (WEB_DATA / "index.json").write_text(json.dumps(clean(index), separators=(",", ":")))
    print("wrote index.json:", len(teams), "teams,", len(elig), "players")

    # ---- per-player files ----
    total_bytes = 0
    for i, (_, r) in enumerate(elig.iterrows(), 1):
        nid = int(r.nflId)
        dp = df[df.nflId == nid]
        prof_all = P.player_profile(dp, nid)
        pvl = P.player_vs_league(df, nid, MIN_PLAYS, player_df=dp)
        takeaways = P.build_takeaways(prof_all, baselines)
        pred = P.holdout_prediction(dp, model, nid,
                                    hist["player_rate"], hist["pos_rate"], hist["global_rate"])
        opps = P.player_opponents(df, nid)

        # per-opponent profiles + matchup comparisons
        per_opp = {}
        for opp in opps.defense:
            dpo = dp[dp.defensiveTeam == opp]
            per_opp[opp] = {
                "profile": profile_to_json(P.player_profile(dpo, nid)),
                "matchup": matchup_to_json(df, nid, opp),
                "holdout": clean(P.holdout_prediction(dpo, model, nid,
                             hist["player_rate"], hist["pos_rate"], hist["global_rate"])),
            }

        payload = {
            "meta": clean(prof_all["meta"]),
            "position": r.officialPosition,
            "team": r.team,
            "opponents": df_records(opps),
            "all": profile_to_json(prof_all),
            "takeaways": clean(takeaways),
            "vs_league": {
                "position": pvl["position"],
                "league_n_players": pvl["league_n_players"],
                "comparisons": df_records(pvl["comparisons"]),
            } if pvl else None,
            "holdout": clean(pred),
            "example_plays": [p for p in (
                example_play_json(dp, nid, "typical"),
                example_play_json(dp, nid, "unusual"),
            )[:plays] if p is not None],
            "per_opponent": per_opp,
        }
        f = WEB_DATA / "players" / f"{nid}.json"
        f.write_text(json.dumps(clean(payload), separators=(",", ":")))
        total_bytes += f.stat().st_size
        if i % 50 == 0 or i == len(elig):
            print(f"  {i}/{len(elig)} players exported")

    print(f"player files total: {total_bytes/1e6:.1f} MB "
          f"(avg {total_bytes/max(len(elig),1)/1024:.0f} KB/player)")
    print("export complete ->", WEB_DATA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plays", type=int, default=2)
    args = ap.parse_args()
    main(args.limit, args.plays)
