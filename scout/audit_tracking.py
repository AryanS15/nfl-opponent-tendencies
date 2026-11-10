"""
Full-coverage tracking audit across ALL games.
Answers the task's data-validation requirements:
  - tracking start/end windows (relative to snap) per play
  - event tag availability across all 122 games
  - which movement/route features are reliably measurable
Writes a summary to scout/outputs/tracking_audit.json and prints a report.

Run: python3 scout/audit_tracking.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "nfl-big-data-bowl-regional-event-data" / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

TRACK_FILES = sorted((DATA / "tracking").glob("tracking_*.csv"))

SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}

def main():
    event_totals = {}
    # per-play: does it have a snap frame? how many frames before/after snap?
    plays_total = 0
    plays_with_snap = 0
    frames_before_snap = []
    frames_after_snap = []
    total_frames_per_play = []
    motion_event_plays = 0           # man_in_motion / shift present
    pass_forward_plays = 0
    arrival_event_plays = 0          # pass_arrived or pass_outcome_caught
    play_action_plays = 0
    games_seen = 0
    frame_rate_hz = 10               # 0.1s frames per NGS spec

    for f in TRACK_FILES:
        games_seen += 1
        # read only needed columns to keep memory sane
        tr = pd.read_csv(f, usecols=["gameId","playId","nflId","frameId","event","team","x","y","s"])
        # event totals
        ev = tr.event.dropna()
        vc = ev.value_counts()
        for k, v in vc.items():
            event_totals[k] = event_totals.get(k, 0) + int(v)

        # per-play analysis using ball rows only would miss nothing for framing;
        # frameId is shared across players, so use one player stream per play.
        # Use groupby on (gameId, playId) over distinct frameIds.
        g = tr.groupby(["gameId","playId"])
        for (gid, pid), d in g:
            plays_total += 1
            frames = d[["frameId","event"]].drop_duplicates("frameId").sort_values("frameId")
            total_frames_per_play.append(frames.frameId.max())
            snap_rows = frames[frames.event.isin(SNAP_EVENTS)]
            if len(snap_rows):
                plays_with_snap += 1
                snap_frame = snap_rows.frameId.min()
                frames_before_snap.append(snap_frame - frames.frameId.min())
                frames_after_snap.append(frames.frameId.max() - snap_frame)
            evset = set(frames.event.dropna())
            if evset & {"man_in_motion","shift"}:
                motion_event_plays += 1
            if "pass_forward" in evset:
                pass_forward_plays += 1
            if evset & {"pass_arrived","pass_outcome_caught"}:
                arrival_event_plays += 1
            if "play_action" in evset:
                play_action_plays += 1

    def pct(n): return round(100*n/plays_total, 1) if plays_total else 0.0
    def secs(frames_list):
        a = np.array(frames_list)
        return {
            "mean_frames": round(float(a.mean()),1),
            "median_frames": float(np.median(a)),
            "p10_frames": float(np.percentile(a,10)),
            "p90_frames": float(np.percentile(a,90)),
            "mean_seconds": round(float(a.mean())/frame_rate_hz,2),
        }

    summary = {
        "games_audited": games_seen,
        "plays_total": plays_total,
        "plays_with_snap_event": plays_with_snap,
        "pct_plays_with_snap": pct(plays_with_snap),
        "frames_before_snap": secs(frames_before_snap) if frames_before_snap else {},
        "frames_after_snap": secs(frames_after_snap) if frames_after_snap else {},
        "total_frames_per_play": secs(total_frames_per_play),
        "pct_plays_with_motion_event": pct(motion_event_plays),
        "pct_plays_with_pass_forward": pct(pass_forward_plays),
        "pct_plays_with_arrival_event": pct(arrival_event_plays),
        "pct_plays_with_play_action_event": pct(play_action_plays),
        "event_totals_top20": dict(sorted(event_totals.items(), key=lambda x:-x[1])[:20]),
    }

    (OUT / "tracking_audit.json").write_text(json.dumps(summary, indent=2))

    print("="*70)
    print("TRACKING AUDIT — ALL GAMES")
    print("="*70)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for kk, vv in v.items():
                print(f"   {kk}: {vv}")
        else:
            print(f"{k}: {v}")

    print("\n--- FEATURE RELIABILITY VERDICT ---")
    print(f"Pre-snap window exists for {summary['pct_plays_with_snap']}% of plays "
          f"(median {summary['frames_before_snap'].get('median_frames','?')} frames "
          f"~ {summary['frames_before_snap'].get('mean_seconds','?')}s before snap).")
    print(f"Pass-forward frame present on {summary['pct_plays_with_pass_forward']}% -> "
          f"RELIABLE anchor for release.")
    print(f"Arrival/catch event present on only {summary['pct_plays_with_arrival_event']}% -> "
          f"DO NOT impute a fixed offset; restrict arrival metrics to tagged plays.")
    print(f"Explicit motion event on {summary['pct_plays_with_motion_event']}% -> "
          f"use geometry-based motion detection as primary, event as corroboration.")

if __name__ == "__main__":
    main()
