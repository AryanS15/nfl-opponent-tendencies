"""
Representative play visualization.

Draws the ACTUAL observed tracking window for one play:
  - line of scrimmage (from the ball's snap-frame x)
  - an arrow showing the direction the offense is moving
  - distinct offense / defense colors with a legend
  - the selected player's path highlighted, with snap + endpoint markers
  - the endpoint event is identified from the tracking `event` tags
We only draw frames from snap to the play's last tagged outcome (or last frame),
i.e. the window that actually exists in the data.

Play selection is explained in the dashboard: we show a TYPICAL play (closest to
the player's median route depth) and a clearly-labelled UNUSUAL play (farthest from
it). We never present the unusual one as representative.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "nfl-big-data-bowl-regional-event-data" / "data"

OFF_COLOR = "#1f77b4"      # blue = offense
DEF_COLOR = "#d62728"      # red = defense
FOCUS_COLOR = "#f5c518"    # gold = selected player
BALL_COLOR = "#6b3f1d"

SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
# ordered priority of "endpoint" events we try to find for the window end
END_EVENTS = ["pass_outcome_caught", "pass_outcome_incomplete", "pass_arrived",
              "qb_sack", "pass_outcome_interception", "tackle", "out_of_bounds",
              "pass_forward"]

def load_play_tracking(gameId, playId):
    f = DATA / "tracking" / f"tracking_{gameId}.csv"
    if not f.exists():
        return None
    tr = pd.read_csv(f)
    return tr[tr.playId == playId].copy()

def _snap_frame(tr):
    s = tr[tr.event.isin(SNAP_EVENTS)]
    return int(s.frameId.min()) if len(s) else int(tr.frameId.min())

def _endpoint(tr):
    """Return (frameId, event_label) for the window end, using real event tags."""
    for ev in END_EVENTS:
        hit = tr[tr.event == ev]
        if len(hit):
            return int(hit.frameId.min()), ev
    return int(tr.frameId.max()), "end of tracking"

# map of tracking-x -> familiar yard-number label (mirrored at midfield)
YARD_LABELS = {
    20: "10", 30: "20", 40: "30", 50: "40", 60: "50",
    70: "40", 80: "30", 90: "20", 100: "10",
}

def _draw_field(ax):
    """Draw a recognizable football field in TRACKING coordinates (x 0-120,
    y 0-53.3). Does not transform any player data — presentation only."""
    FIELD_LEN, FIELD_WID = 120, 53.3
    turf, endzone = "#3a7d44", "#21522c"
    # playing field turf
    ax.add_patch(plt.Rectangle((10, 0), 100, FIELD_WID, color=turf, zorder=0))
    # end zones (x 0-10 and 110-120)
    ax.add_patch(plt.Rectangle((0, 0), 10, FIELD_WID, color=endzone, zorder=0))
    ax.add_patch(plt.Rectangle((110, 0), 10, FIELD_WID, color=endzone, zorder=0))
    ax.text(5, FIELD_WID/2, "END ZONE", color="white", fontsize=9, fontweight="bold",
            rotation=90, ha="center", va="center", alpha=0.85, zorder=1)
    ax.text(115, FIELD_WID/2, "END ZONE", color="white", fontsize=9, fontweight="bold",
            rotation=90, ha="center", va="center", alpha=0.85, zorder=1)

    # 5-yard lines between the goal lines (x 15,20,...,105)
    for xv in range(15, 110, 5):
        ax.axvline(xv, color="white", alpha=0.25, lw=0.8, zorder=1)
    # goal lines at x=10 and x=110 (bold, bright)
    for xv in (10, 110):
        ax.axvline(xv, color="white", alpha=0.95, lw=2.0, zorder=1)

    # yard numbers at the mapped tracking-x positions, near both sidelines
    for xv, lbl in YARD_LABELS.items():
        for yv, va in ((6, "center"), (FIELD_WID - 6, "center")):
            ax.text(xv, yv, lbl, color="white", fontsize=9, ha="center", va=va,
                    alpha=0.9, zorder=1)
    # "Goal line" labels at x=10 and x=110
    for xv in (10, 110):
        ax.text(xv, FIELD_WID - 2.0, "Goal line", color="white", fontsize=6.5,
                ha="center", va="center", alpha=0.75, zorder=1)

    # frame + equal scaling so routes aren't distorted
    ax.set_xlim(0, FIELD_LEN); ax.set_ylim(0, FIELD_WID)
    ax.set_aspect("equal", adjustable="box")
    # remove the raw 0-120 x-axis numbering (explained in a technical note instead)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def plot_play(gameId, playId, focus_nflId=None, title=None):
    tr = load_play_tracking(gameId, playId)
    if tr is None or tr.empty:
        return None, {}

    snap_f = _snap_frame(tr)
    end_f, end_ev = _endpoint(tr)
    if end_f < snap_f:
        end_f = int(tr.frameId.max())
    # restrict to the OBSERVED window: snap -> endpoint
    win = tr[(tr.frameId >= snap_f) & (tr.frameId <= end_f)].copy()

    play_dir = win.playDirection.dropna().iloc[0] if win.playDirection.notna().any() else "right"
    # line of scrimmage = ball x at snap (fallback: focus player x at snap)
    ball_snap = tr[(tr.frameId == snap_f) & (tr.team == "football")]
    if len(ball_snap):
        los_x = float(ball_snap.x.iloc[0])
    else:
        fp = tr[(tr.frameId == snap_f) & (tr.nflId == focus_nflId)]
        los_x = float(fp.x.iloc[0]) if len(fp) else float(win.x.median())

    # offense team = team of the focus player (or home-ish fallback)
    off_team = None
    if focus_nflId is not None and (tr.nflId == focus_nflId).any():
        off_team = tr.loc[tr.nflId == focus_nflId, "team"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    _draw_field(ax)

    # line of scrimmage
    ax.axvline(los_x, color="#ffd166", lw=2, ls="--", zorder=3)
    ax.text(los_x, 54.6, "LOS", color="#c99700", fontsize=8, ha="center", fontweight="bold")

    # direction-of-offense arrow (toward the end zone the offense attacks)
    arrow_dx = 8 if play_dir == "right" else -8
    ax.annotate("", xy=(los_x + arrow_dx, 3), xytext=(los_x, 3),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5), zorder=4)
    ax.text(los_x + arrow_dx/2, 4.8, "offense", color="white", fontsize=7.5,
            ha="center", fontweight="bold", zorder=4)

    for nid, d in win.groupby("nflId"):
        d = d.sort_values("frameId")
        if d.team.iloc[0] == "football":
            ax.plot(d.x, d.y, color=BALL_COLOR, lw=1.2, alpha=0.7, zorder=2)
            continue
        if focus_nflId is not None and nid == focus_nflId:
            ax.plot(d.x, d.y, color=FOCUS_COLOR, lw=3, zorder=6)
            ax.scatter(d.x.iloc[0], d.y.iloc[0], color=FOCUS_COLOR, s=90, zorder=7,
                       edgecolor="black", marker="o")           # snap
            ax.scatter(d.x.iloc[-1], d.y.iloc[-1], color=FOCUS_COLOR, s=130, zorder=7,
                       edgecolor="black", marker="*")           # endpoint
        else:
            is_off = (d.team.iloc[0] == off_team) if off_team else True
            ax.plot(d.x, d.y, color=OFF_COLOR if is_off else DEF_COLOR,
                    lw=1, alpha=0.6, zorder=1)

    ax.set_title(title or f"Play {playId} (game {gameId})", fontsize=11)
    legend = [
        Line2D([0],[0], color=FOCUS_COLOR, lw=3, label="selected player"),
        Line2D([0],[0], color=OFF_COLOR, lw=2, label="offense"),
        Line2D([0],[0], color=DEF_COLOR, lw=2, label="defense"),
        Line2D([0],[0], color=BALL_COLOR, lw=2, label="ball"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=FOCUS_COLOR,
               markeredgecolor="black", markersize=9, label="snap", lw=0),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=FOCUS_COLOR,
               markeredgecolor="black", markersize=13, label=f"endpoint ({end_ev})", lw=0),
    ]
    # place the legend OUTSIDE the field (below) so it never covers the play
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=6, fontsize=7.5, framealpha=0.9, columnspacing=1.0, handletextpad=0.4)
    plt.tight_layout()
    meta = {"snap_frame": snap_f, "end_frame": end_f, "end_event": end_ev,
            "window_frames": int(end_f - snap_f + 1),
            "window_seconds": round((end_f - snap_f + 1) / 10.0, 1),
            "play_direction": play_dir, "los_x": round(los_x, 1)}
    return fig, meta


def pick_example_plays(df, nflId):
    """Return (typical, unusual) route plays with explained selection.
    typical  = closest to the player's MEDIAN route depth (a normal rep)
    unusual  = farthest from median depth (clearly labelled as atypical)
    Each is a dict with gameId, playId, align_bucket, depth, and a 'why' string."""
    d = df[(df.nflId == nflId) & (df.role_cat == "route") &
           (df.depth_at_release.notna())].copy()
    if d.empty:
        return None, None
    med = d.depth_at_release.median()
    d["dist_from_med"] = (d.depth_at_release - med).abs()
    typ = d.sort_values("dist_from_med").iloc[0]
    unu = d.sort_values("dist_from_med").iloc[-1]
    def pack(row, why):
        return {"gameId": int(row.gameId), "playId": int(row.playId),
                "align_bucket": row.align_bucket,
                "depth": round(float(row.depth_at_release), 1), "why": why}
    typical = pack(typ, f"closest to his median route depth ({med:.1f} yd) — a normal rep")
    unusual = pack(unu, f"farthest from his median depth — an outlier, NOT representative")
    return typical, unusual
