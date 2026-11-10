"""
Step 3b: Prove tracking-derived metric is feasible: receiver separation at the
pass-arrival frame. ONE frame per receiver per play (not all frames) -> avoids
treating tracking frames as separate opportunities.
Runs on a sample of games to keep it fast, reports coverage of 'event' tags.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "nfl-big-data-bowl-regional-event-data" / "data"
players = pd.read_csv(DATA / "players.csv")
pff = pd.read_csv(DATA / "pffScoutingData.csv")
pos = players.set_index("nflId")["officialPosition"].to_dict()

SAMPLE = sorted((DATA/"tracking").glob("tracking_*.csv"))[:8]  # 8 games

def hr(t): print("\n" + "="*70 + f"\n{t}\n" + "="*70)

rows = []
event_counts = {}
for f in SAMPLE:
    tr = pd.read_csv(f)
    for e in tr.event.dropna().unique():
        event_counts[e] = event_counts.get(e,0) + int((tr.event==e).sum())
    # frames tagged as pass arrival / caught
    arrival_events = {"pass_arrived","pass_outcome_caught"}
    arr = tr[tr.event.isin(arrival_events)]
    # for each play, pick first arrival frame
    for (gid,pid), g in arr.groupby(["gameId","playId"]):
        frame = g.frameId.min()
        snap = tr[(tr.gameId==gid)&(tr.playId==pid)&(tr.frameId==frame)]
        offense_ball = snap[snap.team=="football"]
        # route runners on this play
        rr = pff[(pff.gameId==gid)&(pff.playId==pid)&(pff.pff_role=="Pass Route")].nflId.tolist()
        cov = pff[(pff.gameId==gid)&(pff.playId==pid)&(pff.pff_role=="Coverage")].nflId.tolist()
        recv = snap[snap.nflId.isin(rr)]
        defs = snap[snap.nflId.isin(cov)]
        if recv.empty or defs.empty: continue
        for _, r in recv.iterrows():
            d = np.hypot(defs.x - r.x, defs.y - r.y)
            if len(d)==0: continue
            rows.append({
                "gameId":gid,"playId":pid,"nflId":r.nflId,
                "position":pos.get(r.nflId),
                "nearest_def_yds": float(d.min()),
            })

sep = pd.DataFrame(rows)
hr("TRACKING 'event' tag availability (8-game sample)")
for k,v in sorted(event_counts.items(), key=lambda x:-x[1])[:15]:
    print(f"  {k}: {v} frames")

hr("SEPARATION AT PASS ARRIVAL by receiver position (8-game sample)")
print("receiver-arrival observations:", len(sep),
      "across", sep[['gameId','playId']].drop_duplicates().shape[0], "plays")
print("(one row per route-runner per play at a single arrival frame)\n")
agg = sep[sep.position.isin(["WR","TE","RB","FB"])].groupby("position").agg(
    n=("nearest_def_yds","size"),
    mean_sep=("nearest_def_yds","mean"),
    median_sep=("nearest_def_yds","median"),
).round(2).sort_values("mean_sep")
print(agg)
print("\nNOTE: this is nearest defender to EACH route runner at the arrival frame,")
print("not the targeted receiver specifically. It demonstrates feasibility of")
print("separation / matchup metrics from tracking for a full-season build.")
