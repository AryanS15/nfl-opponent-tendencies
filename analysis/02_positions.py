"""
Step 2: What can we attribute to WR/TE/RB, and how does play outcome connect
to a receiver position? Checks route participation, position-lined-up granularity,
and whether the targeted receiver is recoverable (it is only via text parsing).
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "nfl-big-data-bowl-regional-event-data" / "data"
players = pd.read_csv(DATA / "players.csv")
plays = pd.read_csv(DATA / "plays.csv")
pff = pd.read_csv(DATA / "pffScoutingData.csv")

pos = players.set_index("nflId")["officialPosition"].to_dict()
pff["officialPosition"] = pff["nflId"].map(pos)

def hr(t): print("\n" + "="*70 + f"\n{t}\n" + "="*70)

# ---- Route runners by official position ----
hr("ROUTE RUNNERS (pff_role == 'Pass Route') by officialPosition")
routes = pff[pff.pff_role == "Pass Route"]
print(routes.officialPosition.value_counts(dropna=False))
print("\ntotal route-runner rows:", len(routes))
print("plays with >=1 route runner:", routes[['gameId','playId']].drop_duplicates().shape[0])

hr("ROUTES PER PLAY by position (avg routes run per dropback)")
nplays = plays.shape[0]
for p in ["WR","TE","RB","FB"]:
    c = (routes.officialPosition == p).sum()
    print(f"{p}: {c} routes over {nplays} plays = {c/nplays:.2f} per play")

# ---- Coverage defenders by position ----
hr("COVERAGE DEFENDERS (pff_role=='Coverage') by officialPosition")
cov = pff[pff.pff_role == "Coverage"]
print(cov.officialPosition.value_counts(dropna=False).head(15))

# ---- Can we identify the targeted receiver? Parse playDescription. ----
hr("TARGETED RECEIVER: recoverable only from playDescription text")
# Pattern: "pass {depth} {dir} to X.Lastname"
pat = re.compile(r"pass (?:incomplete )?(?:short|deep)? ?(?:left|right|middle)? ?to ([A-Z]\.[A-Za-z'\-]+)")
plays["targetName"] = plays.playDescription.str.extract(pat)[0]
comp = plays[plays.passResult.isin(["C","I"])]
print("completions+incompletions:", len(comp))
print("of those, target name parsed:", comp.targetName.notna().sum(),
      f"({comp.targetName.notna().mean()*100:.1f}%)")
print("\nNOTE: name is 'first-initial.lastname' -> ambiguous to map to nflId/position reliably.")
print("Sample parsed targets:\n", plays.loc[plays.targetName.notna(),
      ["playDescription","targetName"]].head(5).to_string())

# ---- air yards / yards after catch available? ----
hr("RECEIVING DETAIL COLUMNS present in plays.csv?")
for col in ["targetNflId","receiverNflId","passLength","airYards","yardsAfterCatch","epa"]:
    print(f"  {col}: {'PRESENT' if col in plays.columns else 'ABSENT'}")

# ---- outcome fields we DO have, per play ----
hr("WHAT OUTCOME WE HAVE: playResult is per-PLAY net yards, not per receiver")
print(plays.groupby("passResult")["playResult"].agg(["count","mean","median"]))
