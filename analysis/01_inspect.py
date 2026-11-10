"""
Step 1: Inspect the NFL Big Data Bowl (2022 edition) data.
Verifies coverage, structure, data quality. No tracking files loaded here (heavy).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "nfl-big-data-bowl-regional-event-data" / "data"

games = pd.read_csv(DATA / "games.csv")
players = pd.read_csv(DATA / "players.csv")
plays = pd.read_csv(DATA / "plays.csv")
pff = pd.read_csv(DATA / "pffScoutingData.csv")

def hr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

hr("GAMES")
print("rows:", len(games))
print("seasons:", sorted(games.season.unique()))
print("weeks:", sorted(games.week.unique()))
print("date range:", games.gameDate.min(), "->", games.gameDate.max())
print("teams (home+visitor):", sorted(set(games.homeTeamAbbr) | set(games.visitorTeamAbbr)))
print("# distinct teams:", len(set(games.homeTeamAbbr) | set(games.visitorTeamAbbr)))
print("games per week:\n", games.week.value_counts().sort_index())

hr("PLAYS")
print("rows:", len(plays))
print("distinct (gameId,playId):", plays[["gameId","playId"]].drop_duplicates().shape[0])
print("games represented in plays:", plays.gameId.nunique())
print("\npassResult value counts:\n", plays.passResult.value_counts(dropna=False))
print("\ndropBackType value counts:\n", plays.dropBackType.value_counts(dropna=False))
print("\noffenseFormation:\n", plays.offenseFormation.value_counts(dropna=False))
print("\npff_passCoverage:\n", plays.pff_passCoverage.value_counts(dropna=False))
print("\npff_passCoverageType:\n", plays.pff_passCoverageType.value_counts(dropna=False))
print("\ndown:\n", plays.down.value_counts(dropna=False).sort_index())
print("\nquarter:\n", plays.quarter.value_counts(dropna=False).sort_index())
print("\nplayResult describe:\n", plays.playResult.describe())

hr("PLAYS: missing values (%)")
miss = (plays.isna().mean() * 100).round(1).sort_values(ascending=False)
print(miss[miss > 0])

hr("PLAYS: duplicate check")
dups = plays.duplicated(subset=["gameId","playId"]).sum()
print("duplicate (gameId,playId) rows:", dups)

hr("PLAYERS")
print("rows:", len(players))
print("distinct nflId:", players.nflId.nunique())
print("\nofficialPosition:\n", players.officialPosition.value_counts(dropna=False))
print("\nmissing (%):\n", (players.isna().mean()*100).round(2))

# weight distribution by skill position
hr("PLAYERS: weight by skill position")
for pos in ["WR","TE","RB","FB"]:
    w = players.loc[players.officialPosition==pos, "weight"]
    if len(w):
        print(f"{pos}: n={len(w)}, mean wt={w.mean():.0f}, min={w.min()}, max={w.max()}")

hr("PFF SCOUTING")
print("rows:", len(pff))
print("distinct (gameId,playId):", pff[['gameId','playId']].drop_duplicates().shape[0])
print("\npff_role:\n", pff.pff_role.value_counts(dropna=False))
print("\npff_positionLinedUp (top 25):\n", pff.pff_positionLinedUp.value_counts(dropna=False).head(25))
print("\nrows per play (describe):\n", pff.groupby(['gameId','playId']).size().describe())

hr("JOIN INTEGRITY")
play_keys = set(map(tuple, plays[['gameId','playId']].values))
pff_keys = set(map(tuple, pff[['gameId','playId']].drop_duplicates().values))
print("plays present in pff:", len(play_keys & pff_keys), "/", len(play_keys))
print("pff plays not in plays.csv:", len(pff_keys - play_keys))
print("pff nflIds not in players.csv:", pff.loc[~pff.nflId.isin(players.nflId) & pff.nflId.notna()].nflId.nunique())
print("games in plays not in games.csv:", set(plays.gameId.unique()) - set(games.gameId.unique()))

# tracking file coverage
hr("TRACKING FILE COVERAGE")
tfiles = sorted((DATA / "tracking").glob("tracking_*.csv"))
tgids = {int(f.stem.split("_")[1]) for f in tfiles}
print("# tracking files:", len(tfiles))
print("games.csv gameIds without a tracking file:", set(games.gameId) - tgids)
print("tracking files without a games.csv entry:", tgids - set(games.gameId))
