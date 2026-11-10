"""
Step 3: Initial insights, with sample sizes. Plays are the unit (not frames).
Builds a best-effort targeted-receiver POSITION by mapping parsed names to the
offensive team's roster for that game, then analyzes defense-vs-position and
coverage effectiveness. Flags ambiguity rates.
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "nfl-big-data-bowl-regional-event-data" / "data"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

players = pd.read_csv(DATA / "players.csv")
games = pd.read_csv(DATA / "games.csv")
plays = pd.read_csv(DATA / "plays.csv")
pff = pd.read_csv(DATA / "pffScoutingData.csv")

def hr(t): print("\n" + "="*70 + f"\n{t}\n" + "="*70)

# ---------- Build name->position map, scoped to the offensive team per play ----------
# Roster per team-game is not given directly; approximate team roster from who
# lined up on offense in pff for that team across games. We map "F.Last" tokens.
pos = players.set_index("nflId")["officialPosition"].to_dict()
name = players.set_index("nflId")["displayName"].to_dict()

def short(n):
    # "Tom Brady" -> "T.Brady" ; handle suffixes crudely
    parts = n.replace(".", "").split()
    if len(parts) < 2: return None
    first = parts[0]
    last = parts[-1]
    if last.lower() in {"jr","sr","ii","iii","iv","v"} and len(parts) >= 3:
        last = parts[-2]
    return f"{first[0]}.{last}"

players["short"] = players["displayName"].map(short)

# which team did each player play offense for? use pff route/pass rows joined to team via tracking? 
# Simpler: use playDescription possessionTeam + global name uniqueness fallback.
# Build short-name -> set of (nflId, position). Resolve per offense using route runners on that play.
short_to_ids = players.groupby("short")["nflId"].apply(list).to_dict()

# route runners per play (candidate targets) with positions
routes = pff[pff.pff_role.isin(["Pass Route"])][["gameId","playId","nflId"]].copy()
routes["position"] = routes.nflId.map(pos)
routes["short"] = routes.nflId.map(lambda i: short(name.get(i,"")) if name.get(i) else None)

# parse target name
pat = re.compile(r"pass (?:incomplete )?(?:short|deep)? ?(?:left|right|middle)? ?to ([A-Z]\.[A-Za-z'\-]+)")
plays["targetShort"] = plays.playDescription.str.extract(pat)[0]

# For each play, match targetShort against that play's route runners (unambiguous, on-field)
route_lookup = routes.groupby(["gameId","playId"]).apply(
    lambda d: dict(zip(d.short, d.position)), include_groups=False
).to_dict()

def target_pos(row):
    d = route_lookup.get((row.gameId, row.playId))
    if not d or pd.isna(row.targetShort): return None
    return d.get(row.targetShort)

plays["targetPos"] = plays.apply(target_pos, axis=1)

pass_plays = plays[plays.passResult.isin(["C","I","IN"])].copy()  # actual pass attempts
hr("TARGET POSITION RESOLUTION (matched against on-field route runners)")
print("pass attempts (C/I/IN):", len(pass_plays))
print("target name parsed:", pass_plays.targetShort.notna().sum())
print("target POSITION resolved via on-field routes:",
      pass_plays.targetPos.notna().sum(),
      f"({pass_plays.targetPos.notna().mean()*100:.1f}% of attempts)")
print("\nunresolved reason = checkdown/RB/scramble/name-mismatch; treat as lower bound")

# ---------- INSIGHT A: Target share & completion rate by receiver position ----------
hr("INSIGHT A: Target volume & completion rate by TARGETED position")
res = pass_plays.dropna(subset=["targetPos"])
res = res[res.targetPos.isin(["WR","TE","RB","FB"])]
tab = res.groupby("targetPos").agg(
    attempts=("playId","size"),
    completions=("passResult", lambda s: (s=="C").sum()),
    comp_pct=("passResult", lambda s: (s=="C").mean()*100),
    mean_yds=("playResult","mean"),
    median_yds=("playResult","median"),
).sort_values("attempts", ascending=False)
tab["comp_pct"] = tab["comp_pct"].round(1)
tab["mean_yds"] = tab["mean_yds"].round(2)
print(tab)

# ---------- INSIGHT B: Coverage scheme effectiveness (defense-level, per play) ----------
hr("INSIGHT B: Completion % & yards allowed by coverage scheme (pass attempts)")
cov = plays[plays.passResult.isin(["C","I","IN","S"])].copy()
att = plays[plays.passResult.isin(["C","I","IN"])].copy()
covtab = att.groupby("pff_passCoverage").agg(
    attempts=("playId","size"),
    comp_pct=("passResult", lambda s:(s=="C").mean()*100),
    mean_yds=("playResult","mean"),
    int_rate=("passResult", lambda s:(s=="IN").mean()*100),
).query("attempts >= 50").sort_values("mean_yds")
covtab = covtab.round(2)
print(covtab)

hr("INSIGHT B2: Man vs Zone (attempts only)")
mz = att.groupby("pff_passCoverageType").agg(
    attempts=("playId","size"),
    comp_pct=("passResult", lambda s:(s=="C").mean()*100),
    mean_yds=("playResult","mean"),
    int_rate=("passResult", lambda s:(s=="IN").mean()*100),
).round(2)
print(mz)

# ---------- INSIGHT C: Sack/pressure by defensive team (pass rush quality) ----------
hr("INSIGHT C: Pressure generated by defense (sack+hit+hurry rate), min 150 dropbacks")
# a play is 'pressured' if any defender recorded hit/hurry/sack
pdef = pff.groupby(["gameId","playId"]).agg(
    hit=("pff_hit","max"), hurry=("pff_hurry","max"), sack=("pff_sack","max")
).reset_index()
pdef = pdef.merge(plays[["gameId","playId","defensiveTeam"]], on=["gameId","playId"])
pdef["pressure"] = pdef[["hit","hurry","sack"]].fillna(0).max(axis=1)
dteam = pdef.groupby("defensiveTeam").agg(
    dropbacks=("playId","size"),
    pressure_rate=("pressure", lambda s: s.mean()*100),
    sack_rate=("sack", lambda s: s.fillna(0).mean()*100),
).query("dropbacks >= 150").sort_values("pressure_rate", ascending=False)
dteam = dteam.round(2)
print(dteam.head(12))
print("...\n", dteam.tail(5))

# ---------- INSIGHT D: Does play-action help? (situation control) ----------
hr("INSIGHT D: Play-action vs not (completion & yards, attempts)")
pa = att.groupby("pff_playAction").agg(
    attempts=("playId","size"),
    comp_pct=("passResult", lambda s:(s=="C").mean()*100),
    mean_yds=("playResult","mean"),
).round(2)
pa.index = pa.index.map({0:"no play-action",1:"play-action"})
print(pa)

# ========================= CHARTS =========================
# Chart 1: target share by position
fig, ax = plt.subplots(1,2, figsize=(11,4))
tab["attempts"].plot(kind="bar", ax=ax[0], color="#4C72B0")
ax[0].set_title("Pass attempts by targeted position\n(resolved via on-field routes)")
ax[0].set_ylabel("attempts"); ax[0].set_xlabel("")
for i,v in enumerate(tab["attempts"]): ax[0].text(i, v+20, str(int(v)), ha="center")
tab["comp_pct"].plot(kind="bar", ax=ax[1], color="#55A868")
ax[1].set_title("Completion % by targeted position"); ax[1].set_ylabel("comp %"); ax[1].set_xlabel("")
ax[1].set_ylim(0,100)
for i,v in enumerate(tab["comp_pct"]): ax[1].text(i, v+2, f"{v:.0f}%", ha="center")
plt.tight_layout(); plt.savefig(OUT/"fig1_target_position.png", dpi=120); plt.close()

# Chart 2: coverage effectiveness
fig, ax = plt.subplots(figsize=(9,5))
covtab_sorted = covtab.sort_values("mean_yds")
x = np.arange(len(covtab_sorted))
ax.barh(x, covtab_sorted["mean_yds"], color="#C44E52")
ax.set_yticks(x); ax.set_yticklabels(covtab_sorted.index)
ax.set_xlabel("mean yards allowed per pass attempt")
ax.set_title("Coverage scheme: yards allowed per attempt (n>=50)")
for i,(y,n) in enumerate(zip(covtab_sorted["mean_yds"], covtab_sorted["attempts"])):
    ax.text(y+0.1, i, f"{y:.1f}  (n={int(n)})", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(OUT/"fig2_coverage.png", dpi=120); plt.close()

# Chart 3: pressure rate by defense
fig, ax = plt.subplots(figsize=(8,8))
d = dteam.sort_values("pressure_rate")
ax.barh(np.arange(len(d)), d["pressure_rate"], color="#8172B3")
ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d.index, fontsize=8)
ax.set_xlabel("pressure rate (% of dropbacks)")
ax.set_title("Pressure rate by defense (Weeks 1-8 2021, min 150 dropbacks)")
plt.tight_layout(); plt.savefig(OUT/"fig3_pressure.png", dpi=120); plt.close()

print("\nSaved charts to", OUT)
