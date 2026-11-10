"""
Player profile computation from the player-play table.
Everything here is DESCRIPTIVE (observed), reported with sample sizes and
Wilson confidence intervals for rates. No modeling, no coaching claims.

Reusable API:
  load() -> DataFrame
  eligible_players(df, min_plays=20) -> DataFrame[nflId, name, pos, team, n]
  player_profile(df, nflId) -> dict of tables
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"

def load():
    p = OUT / "player_play.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(OUT / "player_play.csv")

def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, hi) in %."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2/n
    center = (phat + z*z/(2*n)) / denom
    half = (z*np.sqrt(phat*(1-phat)/n + z*z/(4*n*n))) / denom
    return (round(100*(center-half),1), round(100*(center+half),1))

TEST_WEEKS = {6, 7, 8}
TRAIN_WEEKS = {1, 2, 3, 4, 5}

# plain-language names for coded buckets, reused by the dashboard
DD_LABELS = {
    "1st_long": "1st & 10+",
    "1st": "1st down",
    "2nd_short": "2nd & short (≤3)",
    "2nd_med": "2nd & medium (4-7)",
    "2nd_long": "2nd & long (8+)",
    "3rd/4th_short": "3rd/4th & short (≤3)",
    "3rd/4th_med": "3rd/4th & medium (4-7)",
    "3rd/4th_long": "3rd/4th & long (8+)",
    "unknown": "unknown",
}
ALIGN_LABELS = {
    "wide": "Wide (outside)",
    "slot": "Slot",
    "inline": "Inline (tight)",
    "backfield": "Backfield",
    "unknown": "Unknown",
}

def eligible_players(df, min_plays=20):
    g = (df.groupby(["nflId","displayName","officialPosition","possessionTeam"])
           .size().rename("n").reset_index())
    g = g[g.n >= min_plays].sort_values("n", ascending=False)
    g = g.rename(columns={"possessionTeam":"team"})
    return g

def position_baselines(df):
    """League route rate and mean movement by official position, over route/block
    plays. Used to judge whether a player's tendency adds info beyond his position."""
    rb = df[df.role_cat.isin(["route", "block"])]
    out = {}
    for pos, d in rb.groupby("officialPosition"):
        rt = d[d.role_cat == "route"]
        out[pos] = {
            "route_pct": round(100 * (d.role_cat == "route").mean(), 1),
            "n": int(len(d)),
            "mean_depth_yd": round(float(rt.depth_at_release.mean()), 2) if len(rt) else None,
            "mean_top_speed": round(float(rt.max_speed_postsnap.mean()), 2) if len(rt) else None,
            "mean_lateral_break": round(float(rt.lateral_break.mean()), 2) if len(rt) else None,
        }
    return out

def _rate_table(df, by, flag):
    """Rate of `flag`(bool) grouped by `by`, with n and Wilson CI."""
    rows = []
    for key, d in df.groupby(by):
        n = len(d)
        k = int(d[flag].sum())
        lo, hi = wilson(k, n)
        rows.append({by: key, "n": n, "rate_pct": round(100*k/n,1),
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values("n", ascending=False)

def player_profile(df, nflId, min_cell=5):
    d = df[df.nflId == nflId].copy()
    if d.empty:
        return None
    meta = {
        "nflId": int(nflId),
        "name": d.displayName.iloc[0],
        "position": d.officialPosition.iloc[0],
        "team": d.possessionTeam.iloc[0],
        "n_plays": int(len(d)),
        "n_games": int(d.gameId.nunique()),
        "weeks": sorted(d.week.dropna().unique().tolist()),
    }
    d["is_route"] = d.role_cat == "route"
    d["is_block"] = d.role_cat == "block"

    # alignment frequencies
    align = (d.align_bucket.value_counts(normalize=True)*100).round(1)
    align_n = d.align_bucket.value_counts()
    alignment = pd.DataFrame({"pct": align, "n": align_n}).reset_index(names="alignment")

    # route vs block overall + by alignment
    overall = {
        "route_pct": round(100*d.is_route.mean(),1),
        "block_pct": round(100*d.is_block.mean(),1),
        "route_ci": wilson(int(d.is_route.sum()), len(d)),
        "n": len(d),
    }
    route_by_align = _rate_table(d, "align_bucket", "is_route")
    route_by_align = route_by_align[route_by_align.n >= min_cell]

    # situational route tendencies (only cells with enough n)
    route_by_down = _rate_table(d.dropna(subset=["down"]), "down", "is_route")
    route_by_down = route_by_down[route_by_down.n >= min_cell]
    route_by_dd = _rate_table(d, "dd_bucket", "is_route")
    route_by_dd = route_by_dd[route_by_dd.n >= min_cell]
    route_by_form = _rate_table(d.dropna(subset=["offenseFormation"]), "offenseFormation", "is_route")
    route_by_form = route_by_form[route_by_form.n >= min_cell]
    rz = d[d.redzone == True]
    redzone = {"n": len(rz), "route_pct": round(100*rz.is_route.mean(),1) if len(rz) else None}

    # pre-snap motion (only where observable)
    mo = d[d.motion_observed.notna()]
    motion = {
        "observable_n": int(len(mo)),
        "motion_rate_pct": round(100*mo.motion_observed.mean(),1) if len(mo) else None,
        "note": "pre-snap window ~0.5s only; rate is lower bound of true motion usage",
    }

    # movement tendencies on route plays (within observed window)
    rt = d[d.is_route & d.depth_at_release.notna()].copy()
    # categorize each route's downfield depth into shallow/intermediate/deep
    def depth_band(v):
        if pd.isna(v): return None
        if v < 5: return "short (<5)"
        if v < 12: return "intermediate (5-12)"
        return "deep (12+)"
    # lateral break direction: which way (from snap y) the player ended up
    movement = {
        "n_route_tracked": int(len(rt)),
        "mean_depth_at_release_yd": round(float(rt.depth_at_release.mean()),2) if len(rt) else None,
        "median_depth_yd": round(float(rt.depth_at_release.median()),2) if len(rt) else None,
        "mean_max_speed_yd_s": round(float(rt.max_speed_postsnap.mean()),2) if len(rt) else None,
        "mean_lateral_break_yd": round(float(rt.lateral_break.mean()),2) if len(rt) else None,
        "note": "measured snap->release (median release ~ pass_forward); window-limited",
    }
    depth_dist = None
    if len(rt):
        bands = rt.depth_at_release.map(depth_band).value_counts(normalize=True) * 100
        bands_n = rt.depth_at_release.map(depth_band).value_counts()
        depth_dist = pd.DataFrame({"band": bands.index, "pct": bands.round(1).values,
                                   "n": bands_n.reindex(bands.index).values})

    # per-game trend (route rate with n)
    trend = []
    for wk, g in d.groupby("week"):
        trend.append({"week": int(wk), "n": len(g),
                      "route_pct": round(100*g.is_route.mean(),1)})
    trend = sorted(trend, key=lambda r: r["week"])

    # held-out availability (weeks 6-8 are the model's test period)
    weeks_set = set(int(w) for w in meta["weeks"])
    holdout = {
        "has_test_weeks": bool(weeks_set & TEST_WEEKS),
        "test_weeks_present": sorted(weeks_set & TEST_WEEKS),
        "train_weeks_present": sorted(weeks_set & TRAIN_WEEKS),
        "n_test": int(len(d[d.week.isin(TEST_WEEKS)])),
        "n_train": int(len(d[d.week.isin(TRAIN_WEEKS)])),
    }

    # near-constant route rate? (makes the route-rate charts uninformative)
    near_constant = overall["route_pct"] >= 97.0 or overall["route_pct"] <= 3.0

    return {
        "meta": meta,
        "alignment": alignment,
        "overall_role": overall,
        "route_by_alignment": route_by_align,
        "route_by_down": route_by_down,
        "route_by_dd": route_by_dd,
        "route_by_formation": route_by_form,
        "redzone": redzone,
        "motion": motion,
        "movement": movement,
        "depth_dist": depth_dist,
        "weekly_trend": trend,
        "holdout": holdout,
        "near_constant_route": bool(near_constant),
    }


def build_takeaways(prof, baselines):
    """Produce 3-4 specific, evidence-based scouting takeaways + one hypothesis,
    all generated from the player's actual numbers. Returns a list of dicts:
      {kind: observed|hypothesis, text: str}
    Flags insufficient samples explicitly."""
    meta = prof["meta"]; o = prof["overall_role"]; al = prof["alignment"]
    pos = meta["position"]
    base = baselines.get(pos, {})
    items = []

    # 1) alignment
    top = al.iloc[0]
    lbl = ALIGN_LABELS.get(top["alignment"], top["alignment"])
    sec = al.iloc[1] if len(al) > 1 else None
    atxt = f"Lines up **{lbl}** on **{top['pct']:.0f}%** of {meta['n_plays']} plays (n={int(top['n'])})"
    if sec is not None and sec["pct"] >= 20:
        atxt += f"; also {ALIGN_LABELS.get(sec['alignment'], sec['alignment'])} {sec['pct']:.0f}% (n={int(sec['n'])})"
    atxt += "."
    items.append({"kind": "observed", "text": atxt})

    # 2) route vs block, compared to positional baseline
    base_r = base.get("route_pct")
    cmp = ""
    if base_r is not None:
        diff = o["route_pct"] - base_r
        if abs(diff) >= 5:
            cmp = f" — {abs(diff):.0f} pts {'above' if diff>0 else 'below'} the {pos} average of {base_r:.0f}%"
        else:
            cmp = f" — about the {pos} average ({base_r:.0f}%)"
    if pos == "WR":
        items.append({"kind": "observed",
                      "text": f"Runs a route on **{o['route_pct']:.0f}%** of dropbacks{cmp}. "
                              f"Effectively always a receiver; blocking is negligible."})
    else:
        items.append({"kind": "observed",
                      "text": f"Route on **{o['route_pct']:.0f}%** / blocks on **{o['block_pct']:.0f}%** "
                              f"of dropbacks (95% CI {o['route_ci'][0]:.0f}–{o['route_ci'][1]:.0f}%, n={o['n']}){cmp}."})

    # 3) strongest meaningful situational OR movement tendency beyond the overall rate
    strong = _strongest_tendency(prof, o["route_pct"])
    if strong:
        items.append({"kind": "observed", "text": strong})
    else:
        mv = prof["movement"]
        if mv["n_route_tracked"] >= 15 and mv["mean_depth_at_release_yd"] is not None:
            base_d = base.get("mean_depth_yd")
            dcmp = ""
            if base_d is not None:
                dd = mv["mean_depth_at_release_yd"] - base_d
                if abs(dd) >= 1.0:
                    dcmp = f" vs {base_d:.1f} for {pos}s ({'deeper' if dd>0 else 'shallower'})"
            items.append({"kind": "observed",
                          "text": f"On routes, averages **{mv['mean_depth_at_release_yd']:.1f} yd** downfield "
                                  f"by the throw{dcmp} (n={mv['n_route_tracked']}, snap→release window only)."})
        else:
            items.append({"kind": "observed",
                          "text": "No situational split has a large enough sample to call a reliable tendency."})

    # 4) one coaching hypothesis, tied to the evidence above
    hyp = _hypothesis(prof)
    items.append({"kind": "hypothesis", "text": hyp})
    return items


def _strongest_tendency(prof, overall_route):
    """Find the situational cell whose route rate deviates most from the player's
    overall rate, with enough sample AND a confidence interval that excludes the
    overall rate. Returns a sentence or None."""
    best = None
    for key, tbl, labelmap in [
        ("alignment", prof["route_by_alignment"], ALIGN_LABELS),
        ("situation", prof["route_by_dd"], DD_LABELS),
        ("formation", prof["route_by_formation"], None),
    ]:
        if tbl is None or tbl.empty:
            continue
        col = tbl.columns[0]
        for _, r in tbl.iterrows():
            if r["n"] < 15:
                continue
            # only "meaningful" if CI excludes the overall rate and gap is sizable
            gap = r["rate_pct"] - overall_route
            excludes = (r["ci_hi"] < overall_route) or (r["ci_lo"] > overall_route)
            if abs(gap) >= 10 and excludes:
                if best is None or abs(gap) > abs(best["gap"]):
                    name = labelmap.get(r[col], r[col]) if labelmap else r[col]
                    best = {"gap": gap, "name": name, "rate": r["rate_pct"],
                            "n": int(r["n"]), "dim": key,
                            "ci": (r["ci_lo"], r["ci_hi"])}
    if not best:
        return None
    direction = "more likely" if best["gap"] > 0 else "less likely"
    return (f"From **{best['name']}** ({best['dim']}), route rate is **{best['rate']:.0f}%** "
            f"(CI {best['ci'][0]:.0f}–{best['ci'][1]:.0f}%, n={best['n']}) — "
            f"{direction} to run a route than his overall {overall_route:.0f}%.")


def _hypothesis(prof):
    meta = prof["meta"]; o = prof["overall_role"]; al = prof["alignment"]
    top = al.iloc[0]; lbl = ALIGN_LABELS.get(top["alignment"], top["alignment"])
    pos = meta["position"]
    if pos in ("TE", "RB", "FB") and o["route_pct"] >= 90:
        return (f"Because he releases into a route on ~{o['route_pct']:.0f}% of dropbacks and most often "
                f"from {lbl}, a coach might account for him as a receiver there rather than expecting help "
                f"in protection — to be validated against film and matchup.")
    if pos in ("TE", "RB", "FB") and 20 <= o["route_pct"] <= 80:
        strong = _strongest_tendency(prof, o["route_pct"])
        tell = " His alignment/situation splits above are the first place to look for a pre-snap tell." 
        return (f"His role is genuinely mixed (route {o['route_pct']:.0f}% / block {o['block_pct']:.0f}%), so a "
                f"pre-snap read may be worth building.{tell} Hypothesis only — confirm on film.")
    return (f"Primary alignment is {lbl} ({top['pct']:.0f}%); a coach might set the first coverage answer to that "
            f"spot. This is a hypothesis from observed tendencies, not a proven assignment.")


def holdout_prediction(df, model, nflId, train_player_rate, pos_rate, global_rate):
    """Apply the trained model to a player's HELD-OUT (weeks 6-8) observations only.
    Historical feature uses TRAIN-period rate (past info only). Returns dict or None
    if the player has no test-week observations."""
    d = df[(df.nflId == nflId) & (df.role_cat.isin(["route", "block"])) &
           (df.week.isin(TEST_WEEKS))].copy()
    if d.empty:
        return None
    # hist feature: player's TRAIN rate (fallback to position/global) -> no leakage
    if nflId in train_player_rate.index:
        hr = float(train_player_rate.loc[nflId])
        hr_src = "player's weeks 1-5 rate"
    elif d.officialPosition.iloc[0] in pos_rate.index:
        hr = float(pos_rate.loc[d.officialPosition.iloc[0]])
        hr_src = "position prior (player unseen in weeks 1-5)"
    else:
        hr = float(global_rate); hr_src = "global prior"
    d["hist_route_rate"] = hr
    cat = ["officialPosition","align_bucket","offenseFormation","dd_bucket","personnelO"]
    num = ["down","yardsToGo","yards_to_goal","hist_route_rate"]
    for c in cat: d[c] = d[c].fillna("NA").astype(str)
    for c in num: d[c] = d[c].fillna(d[c].median())
    prob = model.predict_proba(d[cat + num])[:, 1]
    actual = (d.role_cat == "route").astype(int).values
    return {
        "n_test": int(len(d)),
        "test_weeks": sorted(d.week.dropna().astype(int).unique().tolist()),
        "mean_pred_route_prob": round(float(prob.mean()) * 100, 1),
        "actual_route_pct": round(float(actual.mean()) * 100, 1),
        "hist_feature_source": hr_src,
        "hist_feature_value_pct": round(hr * 100, 1),
    }

# =====================================================================
# LEAGUE-WIDE AGGREGATION ("All teams" view)
# =====================================================================
# Two weighting modes everywhere:
#   "play"   -> pool all observations; each PLAY counts equally (volume players
#               influence the aggregate more).
#   "player" -> compute each eligible player's own rate first, then average those
#               rates equally so heavily-used players don't dominate.

def _rate_two_ways(df_group):
    """Given a dataframe subset with an `is_route` bool column, return route rate
    both ways plus sample sizes. Expects `nflId` present for the player-weighted mode."""
    n_plays = len(df_group)
    if n_plays == 0:
        return {"per_play": None, "per_player": None, "n_plays": 0, "n_players": 0}
    per_play = 100 * df_group["is_route"].mean()
    per_player_series = df_group.groupby("nflId")["is_route"].mean()
    per_player = 100 * per_player_series.mean()
    return {"per_play": round(per_play, 1), "per_player": round(per_player, 1),
            "n_plays": int(n_plays), "n_players": int(per_player_series.size)}


def _mean_two_ways(df_group, col):
    """Mean of a numeric column both ways (play-weighted vs player-weighted)."""
    d = df_group.dropna(subset=[col])
    if d.empty:
        return {"per_play": None, "per_player": None, "n_plays": 0, "n_players": 0}
    per_play = d[col].mean()
    per_player = d.groupby("nflId")[col].mean().mean()
    return {"per_play": round(float(per_play), 2), "per_player": round(float(per_player), 2),
            "n_plays": int(len(d)), "n_players": int(d.nflId.nunique())}


def assign_actual_role(df, min_plays=20):
    """Classify each ELIGIBLE player into an 'actual role' group from observed
    alignment + route/block behavior (not just official position). Returns a
    Series indexed by nflId. Thresholds are documented and data-driven."""
    elig = eligible_players(df, min_plays)
    d = df[df.nflId.isin(elig.nflId)]
    roles = {}
    for nid, g in d.groupby("nflId"):
        pos = g.officialPosition.iloc[0]
        am = g.align_bucket.value_counts(normalize=True)
        rb = g[g.role_cat.isin(["route", "block"])]
        route = (rb.role_cat == "route").mean() if len(rb) else np.nan
        wide, slot = am.get("wide", 0), am.get("slot", 0)
        if pos == "WR":
            roles[nid] = "Outside WR" if wide >= slot else "Slot WR"
        elif pos == "TE":
            if route >= 0.90:
                roles[nid] = "Receiving TE"
            elif route < 0.80:
                roles[nid] = "Blocking-heavy TE"
            else:
                roles[nid] = "Balanced TE"
        elif pos == "RB":
            roles[nid] = "Pass-catching RB" if route >= 0.85 else "Blocking-heavy RB"
        elif pos == "FB":
            roles[nid] = "Fullback"
        else:
            roles[nid] = "Other"
    return pd.Series(roles, name="actual_role")


ROLE_ORDER = ["Outside WR", "Slot WR", "Receiving TE", "Balanced TE",
              "Blocking-heavy TE", "Pass-catching RB", "Blocking-heavy RB", "Fullback"]

ROLE_DEFS = {
    "Outside WR": "WR aligned wide at least as often as in the slot",
    "Slot WR": "WR aligned in the slot more often than wide",
    "Receiving TE": "TE who runs a route on ≥90% of dropbacks",
    "Balanced TE": "TE route rate 80–90%",
    "Blocking-heavy TE": "TE who runs a route on <80% of dropbacks",
    "Pass-catching RB": "RB who runs a route on ≥85% of dropbacks",
    "Blocking-heavy RB": "RB who runs a route on <85% of dropbacks",
    "Fullback": "Official FB",
}


def league_by_position(df, min_plays=20):
    """Alignment mix, route/block rate, and movement by official position,
    computed both play-weighted and player-weighted. Eligible players only."""
    elig = eligible_players(df, min_plays)
    d = df[df.nflId.isin(elig.nflId)].copy()
    d["is_route"] = d.role_cat == "route"
    out = {}
    for pos in ["WR", "TE", "RB", "FB"]:
        g = d[d.officialPosition == pos]
        if g.empty:
            continue
        # alignment mix both ways
        align_play = (g.align_bucket.value_counts(normalize=True) * 100).round(1)
        per_player_mix = (g.groupby("nflId").align_bucket
                          .value_counts(normalize=True).rename("p").reset_index())
        align_player = (per_player_mix.groupby("align_bucket")["p"].mean() * 100).round(1)
        rt = g[g.is_route]
        out[pos] = {
            "n_players": int(g.nflId.nunique()),
            "n_plays": int(len(g)),
            "align_play": align_play.to_dict(),
            "align_player": align_player.to_dict(),
            "route_rate": _rate_two_ways(g[g.role_cat.isin(["route", "block"])].assign(
                is_route=lambda x: x.role_cat == "route")),
            "depth": _mean_two_ways(rt, "depth_at_release"),
            "top_speed": _mean_two_ways(rt, "max_speed_postsnap"),
            "lateral_break": _mean_two_ways(rt, "lateral_break"),
        }
    return out


def league_by_situation(df, dimension, min_plays=20):
    """Route rate by a situational dimension ('down','dd_bucket','offenseFormation'),
    optionally per position, both weighting modes. Returns a tidy DataFrame."""
    elig = eligible_players(df, min_plays)
    d = df[df.nflId.isin(elig.nflId) & df.role_cat.isin(["route", "block"])].copy()
    d["is_route"] = d.role_cat == "route"
    d = d.dropna(subset=[dimension])
    rows = []
    for (pos, key), g in d.groupby(["officialPosition", dimension]):
        if pos not in ("WR", "TE", "RB"):
            continue
        r = _rate_two_ways(g)
        if r["n_plays"] < 30:          # league cell needs a reasonable sample
            continue
        rows.append({"position": pos, dimension: key,
                     "route_pct_per_play": r["per_play"],
                     "route_pct_per_player": r["per_player"],
                     "n_plays": r["n_plays"], "n_players": r["n_players"]})
    return pd.DataFrame(rows)


def league_by_role(df, min_plays=20):
    """Aggregate alignment + route/block + movement for each ACTUAL-role group,
    both weighting modes."""
    roles = assign_actual_role(df, min_plays)
    d = df[df.nflId.isin(roles.index)].copy()
    d["actual_role"] = d.nflId.map(roles)
    d["is_route"] = d.role_cat == "route"
    rows = []
    for role, g in d.groupby("actual_role"):
        rt = g[g.is_route]
        rr = _rate_two_ways(g[g.role_cat.isin(["route", "block"])].assign(
            is_route=lambda x: x.role_cat == "route"))
        top_align_play = g.align_bucket.value_counts(normalize=True)
        rows.append({
            "actual_role": role,
            "n_players": int(g.nflId.nunique()),
            "n_plays": int(len(g)),
            "route_pct_per_play": rr["per_play"],
            "route_pct_per_player": rr["per_player"],
            "top_alignment": ALIGN_LABELS.get(top_align_play.index[0], top_align_play.index[0]),
            "top_alignment_pct": round(100 * top_align_play.iloc[0], 1),
            "mean_depth_yd": _mean_two_ways(rt, "depth_at_release")["per_player"],
            "mean_top_speed": _mean_two_ways(rt, "max_speed_postsnap")["per_player"],
        })
    out = pd.DataFrame(rows)
    out["__order"] = out.actual_role.map(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 99)
    return out.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def player_vs_league(df, nflId, min_plays=20, player_df=None):
    """Build explicit 'player vs position-average' comparisons using actual values,
    for the key alignment buckets and route rate. Player-weighted league averages
    (so the baseline isn't dominated by a few high-volume players).

    `df` is always the FULL league frame (for the baseline). `player_df`, if given,
    scopes the PLAYER side (e.g. to one opponent); the league baseline stays full."""
    pf = player_df if player_df is not None else df
    prof = player_profile(pf, nflId)
    if prof is None:
        return None
    pos = prof["meta"]["position"]
    lp = league_by_position(df, min_plays).get(pos)
    if lp is None:
        return None
    pl_align = {r["alignment"]: r["pct"] for _, r in prof["alignment"].iterrows()}
    comps = []
    for bucket in ["wide", "slot", "inline", "backfield"]:
        pv = pl_align.get(bucket, 0.0)
        lv = lp["align_player"].get(bucket, 0.0)
        if pv < 1 and lv < 1:
            continue
        comps.append({"metric": f"Aligns {ALIGN_LABELS.get(bucket, bucket)}",
                      "player_pct": round(pv, 1), "league_pct": round(lv, 1),
                      "diff": round(pv - lv, 1)})
    # route rate
    comps.append({"metric": "Runs a route",
                  "player_pct": prof["overall_role"]["route_pct"],
                  "league_pct": lp["route_rate"]["per_player"],
                  "diff": round(prof["overall_role"]["route_pct"] - lp["route_rate"]["per_player"], 1)})
    return {"position": pos, "comparisons": pd.DataFrame(comps),
            "league_n_players": lp["n_players"]}


def player_opponents(df, nflId):
    """Defenses this player actually faced, with play + game counts (most plays first)."""
    d = df[df.nflId == nflId]
    if d.empty:
        return pd.DataFrame(columns=["defense", "plays", "games"])
    g = (d.groupby("defensiveTeam")
           .agg(plays=("playId", "size"), games=("gameId", "nunique"))
           .reset_index().rename(columns={"defensiveTeam": "defense"})
           .sort_values("plays", ascending=False))
    return g


def matchup_comparison(df, nflId, opponent, min_cell=10):
    """Compare the player's tendencies vs ONE defense against vs ALL OTHER defenses.
    Returns percentage-point differences with sample sizes. Describes observed
    differences only — NOT causal effects of the defense. Flags small samples."""
    d = df[df.nflId == nflId].copy()
    d["is_route"] = d.role_cat == "route"
    vs = d[d.defensiveTeam == opponent].copy()
    rest = d[d.defensiveTeam != opponent].copy()
    if vs.empty:
        return None

    def rate(sub, mask=None):
        s = sub if mask is None else sub[mask(sub)]
        return (100 * s["is_route"].mean(), len(s)) if len(s) else (None, 0)

    rows = []
    # overall route rate
    r_vs, n_vs = rate(vs)
    r_rest, n_rest = rate(rest)
    rows.append({"tendency": "Runs a route", "vs_pct": _rnd(r_vs), "other_pct": _rnd(r_rest),
                 "diff_pp": _diff(r_vs, r_rest), "n_vs": n_vs, "n_other": n_rest})
    # alignment shares
    for b in ["wide", "slot", "inline", "backfield"]:
        pv = 100 * (vs.align_bucket == b).mean()
        pr = 100 * (rest.align_bucket == b).mean()
        if pv < 1 and pr < 1:
            continue
        rows.append({"tendency": f"Aligns {ALIGN_LABELS.get(b, b)}",
                     "vs_pct": _rnd(pv), "other_pct": _rnd(pr),
                     "diff_pp": _diff(pv, pr), "n_vs": len(vs), "n_other": len(rest)})
    comp = pd.DataFrame(rows)
    small = n_vs < min_cell
    return {"opponent": opponent, "n_vs": int(n_vs), "n_other": int(n_rest),
            "n_games_vs": int(vs.gameId.nunique()),
            "weeks_vs": sorted(vs.week.dropna().astype(int).unique().tolist()),
            "comparisons": comp, "small_sample": bool(small),
            "min_cell": min_cell}


def _rnd(v):
    return round(v, 1) if v is not None else None

def _diff(a, b):
    return round(a - b, 1) if (a is not None and b is not None) else None


if __name__ == "__main__":
    df = load()
    print("player-play rows:", len(df))
    print("unique players:", df.nflId.nunique())
    elig = eligible_players(df, 20)
    print("eligible players (>=20 plays):", len(elig))
    print("\nTop 10 by volume:")
    print(elig.head(10).to_string(index=False))
    # demo profile for the highest-volume player
    pid = int(elig.iloc[0].nflId)
    prof = player_profile(df, pid)
    print("\n=== DEMO PROFILE:", prof["meta"]["name"], "===")
    print("meta:", prof["meta"])
    print("\nalignment:\n", prof["alignment"].to_string(index=False))
    print("\noverall role:", prof["overall_role"])
    print("\nroute by alignment:\n", prof["route_by_alignment"].to_string(index=False))
    print("\nmotion:", prof["motion"])
    print("\nmovement:", prof["movement"])
