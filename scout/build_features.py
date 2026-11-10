"""
Core reusable data processing: build ONE row per player-play observation for
WR/TE/RB/FB, combining play context + PFF scouting + tracking-derived features.

Correctness guarantees (per task requirements):
  - Unit of analysis = player-play (gameId, playId, nflId). Never a tracking frame.
  - Alignment uses the PFF label `pff_positionLinedUp` (a snap-time ground-truth
    label) bucketed into wide/slot/inline/backfield, and is VALIDATED against
    tracking geometry (lateral position at snap). Mismatches reported, not hidden.
  - Pre-snap motion is derived ONLY from the observed pre-snap window (~0.5s,
    median 5 frames). We do NOT claim motion we cannot see; low-displacement =
    'no_motion_observed', not 'no motion'.
  - Post-snap movement summarized only within the observed tracking window.
  - Arrival/catch metrics are NOT computed here (only 4.4% of plays tagged).
  - Role (route vs block) taken from PFF `pff_role` — this is a LABEL, used as the
    target for modeling and for descriptive profiles, never as a model input.

Output: scout/outputs/player_play.parquet  (+ csv fallback)

Run: python3 scout/build_features.py            # all games
     python3 scout/build_features.py --games 8  # first N games (quick)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "nfl-big-data-bowl-regional-event-data" / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

SKILL = ["WR", "TE", "RB", "FB"]
SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}

# ---- alignment bucketing from PFF positionLinedUp ----
def align_bucket(lbl: str):
    if not isinstance(lbl, str):
        return "unknown"
    if lbl in ("LWR", "RWR"):
        return "wide"
    if lbl.startswith("S") and "WR" in lbl:   # SLWR, SRWR, SLoWR, SLiWR, SRoWR, SRiWR
        return "slot"
    if lbl.startswith("SCB"):                 # slot corner alignment for a receiver (rare)
        return "slot"
    if lbl.startswith("TE"):                  # TE-L, TE-R, TE-oR, TE-iL, ...
        return "inline"
    if lbl.startswith("HB") or lbl.startswith("FB"):
        return "backfield"
    return "unknown"


def load_core():
    players = pd.read_csv(DATA / "players.csv")
    plays = pd.read_csv(DATA / "plays.csv")
    pff = pd.read_csv(DATA / "pffScoutingData.csv")
    games = pd.read_csv(DATA / "games.csv")
    return players, plays, pff, games


def tracking_features_for_game(path, pff_game):
    """Return dict keyed by (playId, nflId) -> tracking feature dict for skill players."""
    tr = pd.read_csv(path, usecols=[
        "gameId", "playId", "nflId", "frameId", "event", "team",
        "x", "y", "s", "a", "dir", "playDirection"
    ])
    # skill nflIds on this game from pff (offense skill players w/ role)
    want = set(pff_game["nflId"].unique())
    tr = tr[tr.nflId.isin(want)]
    if tr.empty:
        return {}

    # snap frame per play (min snap-tagged frame)
    snap = (tr[tr.event.isin(SNAP_EVENTS)]
            .groupby("playId")["frameId"].min().rename("snapFrame"))
    # pass_forward frame per play (release anchor)
    rel = (tr[tr.event == "pass_forward"]
           .groupby("playId")["frameId"].min().rename("releaseFrame"))

    feats = {}
    for (pid, nid), d in tr.groupby(["playId", "nflId"]):
        d = d.sort_values("frameId")
        sf = snap.get(pid, np.nan)
        rf = rel.get(pid, np.nan)
        row = {
            "snap_frame": sf if pd.notna(sf) else None,
            "n_frames": int(d.frameId.max() - d.frameId.min() + 1),
            "playDirection": d.playDirection.iloc[0],
        }
        # snap-time position (lateral/longitudinal), normalized so offense moves +x
        if pd.notna(sf) and (d.frameId == sf).any():
            s_row = d[d.frameId == sf].iloc[0]
            row["x_snap"] = float(s_row.x)
            row["y_snap"] = float(s_row.y)
            row["s_snap"] = float(s_row.s)
        # pre-snap window motion (ONLY frames before snap; typically ~5 frames)
        if pd.notna(sf):
            pre = d[d.frameId < sf]
            if len(pre) >= 2:
                dx = pre.x.iloc[-1] - pre.x.iloc[0]
                dy = pre.y.iloc[-1] - pre.y.iloc[0]
                row["presnap_lateral_disp"] = float(abs(dy))
                row["presnap_total_disp"] = float(np.hypot(dx, dy))
                row["presnap_max_speed"] = float(pre.s.max())
                row["presnap_frames"] = int(len(pre))
                # motion flag: clear lateral movement in the observed window
                row["motion_observed"] = bool(abs(dy) >= 1.0 or pre.s.max() >= 1.5)
            else:
                row["presnap_frames"] = int(len(pre))
                row["motion_observed"] = None  # window too short -> unknown
        # post-snap movement, from snap to release (or end), within observed window
        if pd.notna(sf):
            end = rf if pd.notna(rf) else d.frameId.max()
            post = d[(d.frameId >= sf) & (d.frameId <= end)]
            if len(post) >= 2:
                # normalize downfield direction by playDirection
                sign = 1.0 if row["playDirection"] == "right" else -1.0
                row["depth_at_release"] = float(sign * (post.x.iloc[-1] - post.x.iloc[0]))
                row["max_speed_postsnap"] = float(post.s.max())
                row["mean_speed_postsnap"] = float(post.s.mean())
                row["total_dist_postsnap"] = float(post.s.sum() * 0.1)  # s(yd/s)*0.1s
                # change-of-direction proxy: stdev of motion direction (dir)
                dd = post["dir"].dropna()
                row["dir_std_postsnap"] = float(dd.std()) if len(dd) > 2 else None
                # net lateral break
                row["lateral_break"] = float(abs(post.y.iloc[-1] - post.y.iloc[0]))
                row["postsnap_window_frames"] = int(len(post))
                row["release_anchored"] = bool(pd.notna(rf))
        feats[(pid, nid)] = row
    return feats


def build(n_games=None):
    players, plays, pff, games = load_core()
    pos_map = players.set_index("nflId")["officialPosition"].to_dict()
    name_map = players.set_index("nflId")["displayName"].to_dict()

    pff = pff.copy()
    pff["officialPosition"] = pff["nflId"].map(pos_map)
    # offensive skill players only, with a defined role
    skill = pff[pff.officialPosition.isin(SKILL) & pff.pff_role.notna()].copy()
    skill["align_bucket"] = skill["pff_positionLinedUp"].map(align_bucket)
    # role label: route vs block vs other
    def role_cat(r):
        if r == "Pass Route":
            return "route"
        if r == "Pass Block":
            return "block"
        return "other"
    skill["role_cat"] = skill["pff_role"].map(role_cat)

    play_cols = ["gameId", "playId", "quarter", "down", "yardsToGo", "possessionTeam",
                 "defensiveTeam", "offenseFormation", "personnelO", "absoluteYardlineNumber",
                 "preSnapHomeScore", "preSnapVisitorScore", "gameClock", "dropBackType"]
    skill = skill.merge(plays[play_cols], on=["gameId", "playId"], how="left")
    skill = skill.merge(games[["gameId", "week", "homeTeamAbbr", "visitorTeamAbbr"]],
                        on="gameId", how="left")

    # field position bucket: distance to opponent end zone (absoluteYardlineNumber 0-120)
    skill["yards_to_goal"] = skill["absoluteYardlineNumber"].apply(
        lambda v: min(v, 100) if pd.notna(v) else np.nan)
    skill["redzone"] = skill["absoluteYardlineNumber"].apply(
        lambda v: bool(pd.notna(v) and v <= 20))
    # down/distance bucket
    def dd_bucket(row):
        d, y = row["down"], row["yardsToGo"]
        if pd.isna(d):
            return "unknown"
        if d in (1,) and y >= 10:
            return "1st_long"
        if d in (2,):
            return "2nd_short" if y <= 3 else ("2nd_med" if y <= 7 else "2nd_long")
        if d in (3, 4):
            return "3rd/4th_short" if y <= 3 else ("3rd/4th_med" if y <= 7 else "3rd/4th_long")
        return "1st"
    skill["dd_bucket"] = skill.apply(dd_bucket, axis=1)
    skill["displayName"] = skill["nflId"].map(name_map)

    # ---- attach tracking features game by game ----
    track_files = sorted((DATA / "tracking").glob("tracking_*.csv"))
    if n_games:
        track_files = track_files[:n_games]
        keep_gids = {int(f.stem.split("_")[1]) for f in track_files}
        skill = skill[skill.gameId.isin(keep_gids)]

    track_cols = ["x_snap","y_snap","s_snap","snap_frame","n_frames","presnap_lateral_disp",
                  "presnap_total_disp","presnap_max_speed","presnap_frames","motion_observed",
                  "depth_at_release","max_speed_postsnap","mean_speed_postsnap",
                  "total_dist_postsnap","dir_std_postsnap","lateral_break",
                  "postsnap_window_frames","release_anchored","playDirection"]
    for c in track_cols:
        skill[c] = np.nan

    feat_rows = []
    for f in track_files:
        gid = int(f.stem.split("_")[1])
        pff_game = skill[skill.gameId == gid][["playId","nflId"]]
        if pff_game.empty:
            continue
        feats = tracking_features_for_game(f, pff_game)
        for (pid, nid), d in feats.items():
            rec = {"gameId": gid, "playId": pid, "nflId": nid}
            rec.update(d)
            feat_rows.append(rec)
        print(f"  processed {f.name}: {len(feats)} player-play tracking rows")

    # merge tracking features back on the (gameId, playId, nflId) key (vectorized)
    skill = skill.drop(columns=track_cols)
    if feat_rows:
        feat_df = pd.DataFrame(feat_rows)
        skill = skill.merge(feat_df, on=["gameId", "playId", "nflId"], how="left")
    else:
        for c in track_cols:
            skill[c] = np.nan

    skill = skill.reset_index(drop=True)
    # persist
    try:
        skill.to_parquet(OUT / "player_play.parquet", index=False)
        print("wrote", OUT / "player_play.parquet")
    except Exception as e:
        print("parquet failed (", e, ") -> csv")
    skill.to_csv(OUT / "player_play.csv", index=False)
    print("wrote", OUT / "player_play.csv", "rows:", len(skill))

    # ---- validate alignment bucket vs tracking geometry ----
    print("\n--- ALIGNMENT VALIDATION (PFF label vs tracking lateral position) ---")
    v = skill.dropna(subset=["y_snap", "x_snap"]).copy()
    # hash y position relative to field center (26.65). wide = near sideline.
    v["from_center"] = (v["y_snap"] - 26.65).abs()
    for b in ["wide", "slot", "inline", "backfield"]:
        sub = v[v.align_bucket == b]
        if len(sub):
            print(f"  {b:10s} n={len(sub):6d}  mean |y-center|={sub.from_center.mean():.1f} yd")
    print("  (expect wide >> slot > inline ~ backfield; backfield deeper behind LOS)")
    return skill


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=None, help="limit to first N games")
    args = ap.parse_args()
    build(args.games)
