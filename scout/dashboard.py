"""
Opponent Offensive Tendencies (Streamlit).

Goal: a defensive coach selects an opposing WR/TE/RB and, within a few seconds,
reads what matters — primary alignment, route-vs-block role, the strongest
situational/movement tendency, and a clearly-labelled coaching hypothesis.

Sections: takeaways first, then observed detail (adapted per player), example
plays, a corrected held-out prediction panel, and consolidated methodology.

Run: streamlit run scout/dashboard.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

sys.path.append(str(Path(__file__).resolve().parent))
import profiles as P
import playviz as V

OUT = Path(__file__).resolve().parent / "outputs"

st.set_page_config(page_title="Opponent Offensive Tendencies", layout="wide")

# ----------------------- cached loaders -----------------------
@st.cache_data
def get_data():
    return P.load()

@st.cache_data
def get_audit():
    f = OUT / "tracking_audit.json"
    return json.loads(f.read_text()) if f.exists() else {}

@st.cache_data
def get_model_results():
    f = OUT / "model_results.json"
    return json.loads(f.read_text()) if f.exists() else {}

@st.cache_resource
def get_model():
    mp = OUT / "route_block_gbm.joblib"
    hp = OUT / "hist_rates.joblib"
    if mp.exists() and hp.exists():
        return joblib.load(mp), joblib.load(hp)
    return None, None

@st.cache_data
def get_baselines():
    return P.position_baselines(get_data())

df = get_data()
audit = get_audit()
model_res = get_model_results()
model, hist = get_model()
baselines = get_baselines()

# ----------------------- small render helpers -----------------------
def _bar_or_table(t, keyname):
    """Horizontal route-% bar chart with n labels; falls back to a message."""
    if t is None or t.empty:
        st.info(f"Not enough data to break down by {keyname} (need ≥5 plays per cell).")
        return
    fig, ax = plt.subplots(figsize=(7.5, max(2.2, 0.5 * len(t) + 1)))
    cats = t[keyname].astype(str).tolist()[::-1]
    vals = t["route %"].tolist()[::-1]
    ns = t["plays (n)"].tolist()[::-1]
    los = t["CI low %"].tolist()[::-1]
    his = t["CI high %"].tolist()[::-1]
    y = np.arange(len(cats))
    ax.barh(y, vals, color="#1f77b4")
    # CI whiskers
    for i, (v, lo, hi) in enumerate(zip(vals, los, his)):
        if pd.notna(lo) and pd.notna(hi):
            ax.plot([lo, hi], [i, i], color="black", lw=1.2)
            ax.plot([lo, lo], [i-0.1, i+0.1], color="black", lw=1.2)
            ax.plot([hi, hi], [i-0.1, i+0.1], color="black", lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(cats)
    ax.set_xlim(0, 105); ax.set_xlabel("route rate % (bar) · black line = 95% CI")
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(min(v + 2, 99), i, f"{v:.0f}%  (n={int(n)})", va="center", fontsize=8)
    st.pyplot(fig); plt.close(fig)


def _render_movement(prof, baselines):
    """Movement summary on route plays, compared to the positional baseline."""
    mv = prof["movement"]
    pos = prof["meta"]["position"]
    if mv["n_route_tracked"] < 5 or mv["mean_depth_at_release_yd"] is None:
        st.info("Not enough tracked route plays to summarize movement reliably.")
        return
    base = baselines.get(pos, {})
    def cmp_str(val, base_val, unit, more="deeper/faster"):
        if base_val is None or val is None:
            return "—"
        d = val - base_val
        if abs(d) < 0.3:
            return f"≈ {pos} avg ({base_val}{unit})"
        return f"{'+' if d>0 else ''}{d:.1f}{unit} vs {pos} avg ({base_val}{unit})"
    rows = [
        {"metric": "Mean route depth at release (yd downfield)",
         "value": mv["mean_depth_at_release_yd"],
         "vs baseline": cmp_str(mv["mean_depth_at_release_yd"], base.get("mean_depth_yd"), " yd")},
        {"metric": "Median route depth (yd)", "value": mv["median_depth_yd"], "vs baseline": "—"},
        {"metric": "Mean top speed (yd/s)", "value": mv["mean_max_speed_yd_s"],
         "vs baseline": cmp_str(mv["mean_max_speed_yd_s"], base.get("mean_top_speed"), " yd/s")},
        {"metric": "Mean lateral break (yd)", "value": mv["mean_lateral_break_yd"],
         "vs baseline": cmp_str(mv["mean_lateral_break_yd"], base.get("mean_lateral_break"), " yd")},
        {"metric": "Route plays tracked (n)", "value": mv["n_route_tracked"], "vs baseline": "—"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    # depth distribution chart
    dd = prof.get("depth_dist")
    if dd is not None and not dd.empty:
        order = ["short (<5)", "intermediate (5-12)", "deep (12+)"]
        dd = dd.set_index("band").reindex(order).dropna().reset_index()
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.bar(dd["band"], dd["pct"], color="#9467bd")
        for i, (p, n) in enumerate(zip(dd["pct"], dd["n"])):
            ax.text(i, p + 1, f"{p:.0f}%\n(n={int(n)})", ha="center", fontsize=8)
        ax.set_ylabel("% of routes"); ax.set_ylim(0, max(dd["pct"]) * 1.3)
        ax.set_title("Route depth distribution (observed window)")
        st.pyplot(fig); plt.close(fig)
    st.caption(mv["note"] + "  Comparison is to the league average for this official position.")


# ----------------------- ALL-TEAMS view renderer -----------------------
ALIGN_ORDER = ["wide", "slot", "inline", "backfield"]

def _weight_key(weight_mode):
    """Map the UI toggle to the dict/column suffix used by profiles aggregations."""
    return "per_player" if weight_mode.startswith("Each player") else "per_play"

def render_league_view(df):
    st.title("Opponent Offensive Tendencies — League-wide")
    st.markdown("How WRs, TEs, and RBs across the league line up, move, and split "
                "between running routes and blocking.")
    st.caption("**2021 NFL · Weeks 1–8 · Sampled passing dropbacks only.** "
               "These are league-wide patterns **within the sampled dropbacks**, not all "
               "NFL offensive plays. Use them as a comparison baseline for individual players.")

    c1, c2 = st.columns([2, 1])
    with c1:
        weight_mode = st.radio(
            "Weighting", ["Each play equally", "Each player equally"], horizontal=True,
            help="‘Each play equally’ pools all observations, so high-volume players "
                 "influence the average more. ‘Each player equally’ averages each "
                 "eligible player’s own rate, so no single player dominates.")
    with c2:
        min_plays = st.slider("Min plays to be eligible", 10, 100, 20, 5, key="league_min")
    wk = _weight_key(weight_mode)
    elig = P.eligible_players(df, min_plays)
    st.caption(f"Based on **{len(elig)} eligible players** (≥{min_plays} observed plays each). "
               f"Currently weighting **{weight_mode.lower()}**.")

    tabs = st.tabs(["By position", "By situation", "By actual role"])

    # ---------- BY POSITION ----------
    with tabs[0]:
        bp = P.league_by_position(df, min_plays)
        st.markdown("**Alignment mix by position**")
        rows = []
        for pos in ["WR", "TE", "RB", "FB"]:
            if pos not in bp:
                continue
            mix = bp[pos]["align_player" if wk == "per_player" else "align_play"]
            row = {"position": pos, "players": bp[pos]["n_players"], "plays": bp[pos]["n_plays"]}
            for a in ALIGN_ORDER:
                row[P.ALIGN_LABELS[a]] = round(mix.get(a, 0.0), 1)
            rows.append(row)
        mixdf = pd.DataFrame(rows)
        # stacked bar
        fig, ax = plt.subplots(figsize=(8, 3.2))
        bottom = np.zeros(len(mixdf))
        colors = {"Wide (outside)": "#1f77b4", "Slot": "#ff7f0e",
                  "Inline (tight)": "#2ca02c", "Backfield": "#9467bd"}
        for a in ALIGN_ORDER:
            lab = P.ALIGN_LABELS[a]
            ax.bar(mixdf.position, mixdf[lab], bottom=bottom, label=lab, color=colors[lab])
            bottom += mixdf[lab].values
        ax.set_ylabel("% of plays"); ax.set_ylim(0, 105)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
        st.pyplot(fig); plt.close(fig)
        st.dataframe(mixdf, hide_index=True, use_container_width=True)

        st.markdown("**Route vs block, and movement, by position**")
        rows2 = []
        for pos in ["WR", "TE", "RB", "FB"]:
            if pos not in bp:
                continue
            v = bp[pos]
            rows2.append({
                "position": pos,
                "route %": v["route_rate"][wk],
                "mean route depth (yd)": v["depth"][wk],
                "mean top speed (yd/s)": v["top_speed"][wk],
                "mean lateral break (yd)": v["lateral_break"][wk],
                "players": v["n_players"], "plays": v["n_plays"],
            })
        st.dataframe(pd.DataFrame(rows2), hide_index=True, use_container_width=True)
        st.caption("Depth/speed/break measured on route plays within the observed "
                   "snap→release window only.")

    # ---------- BY SITUATION ----------
    with tabs[1]:
        dim = st.selectbox("Situational dimension",
                           ["down", "dd_bucket", "offenseFormation"],
                           format_func=lambda d: {"down": "Down",
                               "dd_bucket": "Down & distance",
                               "offenseFormation": "Formation"}[d])
        sit = P.league_by_situation(df, dim, min_plays)
        if sit.empty:
            st.info("No situational cells meet the minimum sample.")
        else:
            col = f"route_pct_{wk}"
            sit2 = sit.copy()
            if dim == "dd_bucket":
                sit2["label"] = sit2["dd_bucket"].map(lambda v: P.DD_LABELS.get(v, v))
            elif dim == "down":
                sit2["label"] = sit2["down"].map(lambda v: f"{int(v)}" if pd.notna(v) else v)
            else:
                sit2["label"] = sit2[dim].astype(str)
            fig, ax = plt.subplots(figsize=(8.5, 3.6))
            for pos, col_c in [("WR", "#1f77b4"), ("TE", "#2ca02c"), ("RB", "#9467bd")]:
                s = sit2[sit2.position == pos]
                if s.empty:
                    continue
                ax.plot(s["label"], s[col], marker="o", label=pos, color=col_c)
            ax.set_ylabel("route rate %"); ax.set_ylim(0, 105)
            ax.set_xlabel(""); ax.legend(title="position", fontsize=8)
            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig); plt.close(fig)
            show = sit2[["position", "label", col, "n_plays", "n_players"]].rename(
                columns={"label": dim, col: "route %", "n_plays": "plays (n)",
                         "n_players": "players"})
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption("Each cell needs ≥30 plays league-wide. Route rate only (WR/TE/RB).")

    # ---------- BY ACTUAL ROLE ----------
    with tabs[2]:
        st.markdown("Groups defined from **observed** alignment + route/block behavior, "
                    "not just official position.")
        with st.expander("How roles are defined"):
            for r in P.ROLE_ORDER:
                if r in P.ROLE_DEFS:
                    st.markdown(f"- **{r}** — {P.ROLE_DEFS[r]}")
        lr = P.league_by_role(df, min_plays)
        route_col = "route_pct_per_player" if wk == "per_player" else "route_pct_per_play"
        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.barh(lr.actual_role[::-1], lr[route_col][::-1], color="#1f77b4")
        for i, (v, n) in enumerate(zip(lr[route_col][::-1], lr.n_players[::-1])):
            ax.text(min(v + 1, 97), i, f"{v:.0f}%  ({int(n)} players)", va="center", fontsize=8)
        ax.set_xlim(0, 105); ax.set_xlabel("route rate %")
        st.pyplot(fig); plt.close(fig)
        show = lr.rename(columns={
            "actual_role": "role", route_col: "route %",
            "top_alignment": "top alignment", "top_alignment_pct": "top align %",
            "mean_depth_yd": "mean depth (yd)", "mean_top_speed": "top speed (yd/s)",
            "n_players": "players", "n_plays": "plays"})
        st.dataframe(show[["role", "players", "plays", "route %", "top alignment",
                           "top align %", "mean depth (yd)", "top speed (yd/s)"]],
                     hide_index=True, use_container_width=True)

    with st.expander("Scope & method"):
        st.markdown(
            "- League-wide patterns are computed **only over sampled 2021 Weeks 1–8 passing "
            "dropbacks** for eligible players — not all NFL plays, and not run plays.\n"
            "- **Each play equally** pools observations; **each player equally** averages "
            "per-player rates so heavy-usage players don't dominate.\n"
            "- Alignment from PFF snap labels; route/block from PFF role; movement from NGS "
            "tracking within the snap→release window.")


# ----------------------- header + scope -----------------------
st.title("Opponent Offensive Tendencies")
st.markdown("Understand how opposing WRs, TEs, and RBs line up, move, and take on "
            "different roles on passing plays.")
st.caption("**2021 NFL · Weeks 1–8 · Sampled passing dropbacks only.** "
           "All numbers describe behavior **on sampled dropbacks**, not all offensive plays.")

# ----------------------- sidebar: view switch -----------------------
view = st.sidebar.radio(
    "View", ["Single player", "All teams (league-wide)"],
    help="Single player scouts one opponent. All teams shows league-wide patterns "
         "across the sampled dropbacks, useful as a comparison baseline.")
st.sidebar.divider()

# ===================================================================
# ALL-TEAMS (LEAGUE-WIDE) VIEW
# ===================================================================
if view == "All teams (league-wide)":
    render_league_view(df)
    st.stop()

# ----------------------- sidebar filters (single player) -----------------------
st.sidebar.header("Select a player")
teams = sorted(P.eligible_players(df, 10).team.dropna().unique().tolist())
team = st.sidebar.selectbox("Opposing offense", teams,
                            help="The team whose player you are scouting.")
pos_filter = st.sidebar.multiselect("Position", ["WR", "TE", "RB", "FB"],
                                     default=["WR", "TE", "RB"])
min_plays = st.sidebar.slider("Minimum observed plays", 10, 100, 20, 5,
    help="Hide players with fewer observed dropback plays than this. Higher = more "
         "reliable but fewer players.")

elig = P.eligible_players(df, min_plays)
team_players = elig[(elig.team == team) & (elig.officialPosition.isin(pos_filter))]

if team_players.empty:
    st.warning(f"No {', '.join(pos_filter) or 'eligible'} players for {team} with "
               f"≥{min_plays} plays. Lower the minimum or change position/team.")
    st.stop()

label = team_players.apply(
    lambda r: f"{r.displayName} · {r.officialPosition} · {int(r.n)} plays", axis=1)
choice = st.sidebar.selectbox("Player", options=list(team_players.nflId),
                              format_func=lambda i: label[team_players.nflId == i].iloc[0])

st.sidebar.caption(f"{len(elig)} players meet the ≥{min_plays}-play threshold league-wide.")

# ----------------------- opposing-defense filter -----------------------
opp_tbl = P.player_opponents(df, int(choice))
opp_options = ["All opponents"] + [
    f"{r.defense}  ({int(r.plays)} plays)" for _, r in opp_tbl.iterrows()]
opp_choice = st.sidebar.selectbox(
    "Opposing defense", opp_options, index=0,
    help="Defaults to all defenses this player faced. Pick one to see how his "
         "tendencies looked against that defense vs. everyone else.")
opponent = None if opp_choice == "All opponents" else opp_choice.split("  (")[0]

# scope the player's rows to the chosen opponent (All opponents = full profile)
dplayer_full = df[df.nflId == int(choice)]
dscope = dplayer_full if opponent is None else dplayer_full[dplayer_full.defensiveTeam == opponent]

prof = P.player_profile(dscope, int(choice))
meta = prof["meta"]
o = prof["overall_role"]
al = prof["alignment"]

# ----------------------- scope banner (prominent) -----------------------
scope_label = "All opponents" if opponent is None else f"vs {opponent} only"
c1, c2, c3, c4, c5 = st.columns([1.8, 1.3, 1, 1, 1.2])
c1.metric("Player", meta["name"])
c2.metric("Opponent scope", scope_label,
          help="‘All opponents’ aggregates every observed play. A single defense "
               "shows only that matchup.")
c3.metric("Official position", meta["position"],
          help="Roster position. May differ from where he aligns and what he does.")
c4.metric("Plays (dropbacks)", meta["n_plays"])
c5.metric("Games · weeks",
          f"{meta['n_games']} · wk {min(meta['weeks'])}–{max(meta['weeks'])}")
st.caption(f"Scope: **{meta['name']}** · **{scope_label}** · "
           f"{meta['n_plays']} passing-dropback plays across {meta['n_games']} game(s), "
           f"weeks {min(meta['weeks'])}–{max(meta['weeks'])}. "
           f"(He faced {len(opp_tbl)} defense(s) total in the sample.)")
if opponent is not None and meta["n_plays"] < 15:
    st.warning(f"⚠️ Only {meta['n_plays']} plays vs {opponent} — a very small sample. "
               f"Treat everything below as highly uncertain.")

# ============================================================
# MATCHUP COMPARISON (only when a specific defense is selected)
# ============================================================
if opponent is not None:
    st.markdown(f"### {meta['name']} vs {opponent} — did his behavior differ?")
    mc = P.matchup_comparison(df, int(choice), opponent)
    if mc is None:
        st.info(f"No observed plays vs {opponent}.")
    elif mc["n_other"] == 0:
        st.info(f"{meta['name']} only faced {opponent} in the sample "
                f"(n={mc['n_vs']}), so there's no 'other opponents' group to compare against. "
                f"The profile below is his full observed profile.")
    else:
        comp = mc["comparisons"].copy()
        disp = comp.rename(columns={
            "tendency": "tendency", "vs_pct": f"vs {opponent} %",
            "other_pct": "vs others %", "diff_pp": "diff (pp)",
            "n_vs": f"n (vs {opponent})", "n_other": "n (others)"})
        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.caption(f"Plays vs {opponent}: **{mc['n_vs']}** "
                   f"(weeks {mc['weeks_vs']}, {mc['n_games_vs']} game) · "
                   f"vs all other opponents: **{mc['n_other']}**. "
                   f"‘diff (pp)’ = percentage-point difference (vs {opponent} minus vs others).")
        if mc["small_sample"]:
            st.warning(f"⚠️ Small sample vs {opponent} (n={mc['n_vs']} < {mc['min_cell']}). "
                       f"Differences are easily driven by chance.")
        st.info("🧠 These are **observed differences within one or two games**, not proven "
                "effects of that defense. With one game per matchup, game script, teammates, "
                "and play-calling are confounded with the opponent. Treat as a hypothesis.")
    st.divider()

# ============================================================
# 1) TAKEAWAYS FIRST — "What the defense should know"
# ============================================================
hdr = "### 🏈 What the defense should know" + (
    f"  ·  {scope_label}" if opponent is not None else "")
st.markdown(hdr)
takeaways = P.build_takeaways(prof, baselines)
obs = [t for t in takeaways if t["kind"] == "observed"]
hyp = [t for t in takeaways if t["kind"] == "hypothesis"]

tcol1, tcol2 = st.columns([3, 2])
with tcol1:
    for t in obs:
        st.markdown(f"✅ {t['text']}")
    st.caption("✅ = observed in the data (with sample sizes).")
with tcol2:
    for t in hyp:
        st.info(f"🧠 **Coaching hypothesis:** {t['text']}")

st.divider()

# ============================================================
# 2) OBSERVED DETAIL — adapted to the player
# ============================================================
st.markdown("### Observed detail")
st.caption("Directly measured from PFF snap labels + NGS tracking. Not predictions.")

# ---- vs league baseline (player vs same-position average) ----
pvl = P.player_vs_league(df, int(choice), player_df=dscope)
if pvl is not None:
    with st.expander(f"How he compares to all {meta['position']}s "
                     f"({pvl['league_n_players']} eligible players)", expanded=True):
        comp = pvl["comparisons"].copy()
        # one plain-language headline sentence for the biggest alignment gap
        align_rows = comp[comp.metric.str.startswith("Aligns")]
        if not align_rows.empty:
            big = align_rows.iloc[align_rows["diff"].abs().argmax()]
            spot = big["metric"].replace("Aligns ", "")
            updown = "more" if big["diff"] > 0 else "less"
            st.markdown(
                f"**{meta['name']} lines up {spot.lower()} {big['player_pct']:.0f}% of the time "
                f"versus {big['league_pct']:.0f}% for {meta['position']}s overall** — "
                f"{abs(big['diff']):.0f} pts {updown} than average.")
        disp = comp.rename(columns={"metric": "tendency", "player_pct": f"{meta['name']} %",
                                    "league_pct": f"{meta['position']} avg %", "diff": "diff (pts)"})
        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.caption("League average weights **each player equally** (so high-volume players "
                   "don't dominate the baseline). Positive diff = more than the positional average.")

left, right = st.columns(2)

# ---- alignment (always informative) ----
with left:
    st.markdown("**Where he lines up**")
    al2 = al.copy()
    al2["alignment"] = al2["alignment"].map(lambda a: P.ALIGN_LABELS.get(a, a))
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(al2["alignment"][::-1], al2["pct"][::-1], color="#1f77b4")
    for i, (p, n) in enumerate(zip(al2["pct"][::-1], al2["n"][::-1])):
        ax.text(p + 1, i, f"{p:.0f}%  (n={int(n)})", va="center", fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("% of plays")
    st.pyplot(fig); plt.close(fig)

# ---- route vs block: ADAPT to near-constant ----
with right:
    st.markdown("**Route vs. block role**")
    base_r = baselines.get(meta["position"], {}).get("route_pct")
    if prof["near_constant_route"]:
        verb = "runs a route" if o["route_pct"] >= 50 else "pass-blocks"
        st.success(f"Role is essentially fixed: **{verb} on {max(o['route_pct'],o['block_pct']):.0f}%** "
                   f"of dropbacks (n={o['n']}).")
        st.caption("Route-rate split charts are omitted here because they'd all show ~the same value "
                   "and add no information. Focus on alignment, depth, and movement instead.")
        if base_r is not None:
            st.caption(f"Positional baseline: {meta['position']}s run routes {base_r:.0f}% of the time.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Runs a route", f"{o['route_pct']:.0f}%",
                  help=f"Share of this player's dropbacks where PFF tagged him as running a "
                       f"route (vs pass blocking). 95% confidence interval "
                       f"{o['route_ci'][0]:.0f}–{o['route_ci'][1]:.0f}% over n={o['n']} plays. "
                       f"The interval is the plausible range for his true rate given the sample.")
        m2.metric("Pass blocks", f"{o['block_pct']:.0f}%")
        if base_r is not None:
            diff = o["route_pct"] - base_r
            st.caption(f"Positional baseline: {meta['position']}s average {base_r:.0f}% "
                       f"({'+' if diff>=0 else ''}{diff:.0f} pts vs this player).")

# ---- situational tendencies: TABS (readable, full width) ----
if prof["near_constant_route"]:
    st.markdown("**Movement on routes** (the meaningful variation for this player)")
    _render_movement(prof, baselines)
else:
    st.markdown("**Situational route tendencies**")
    st.caption("Each cell shows route rate with sample size and a 95% confidence interval "
               "(the plausible range for the true rate). Cells with n<5 are hidden.")
    tabs = st.tabs(["By alignment", "By down", "By down & distance", "By formation", "Movement"])

    def nice(tbl, keycol, label_map=None, keyname="category"):
        if tbl is None or tbl.empty:
            return None
        t = tbl.copy()
        if label_map:
            t[keycol] = t[keycol].map(lambda v: label_map.get(v, v))
        t = t.rename(columns={keycol: keyname, "rate_pct": "route %",
                              "n": "plays (n)", "ci_lo": "CI low %", "ci_hi": "CI high %"})
        return t[[keyname, "route %", "plays (n)", "CI low %", "CI high %"]]

    with tabs[0]:
        t = nice(prof["route_by_alignment"], "align_bucket", P.ALIGN_LABELS, "alignment")
        _bar_or_table(t, "alignment")
    with tabs[1]:
        t = nice(prof["route_by_down"], "down", None, "down")
        _bar_or_table(t, "down")
    with tabs[2]:
        t = nice(prof["route_by_dd"], "dd_bucket", P.DD_LABELS, "situation")
        _bar_or_table(t, "situation")
    with tabs[3]:
        t = nice(prof["route_by_formation"], "offenseFormation", None, "formation")
        _bar_or_table(t, "formation")
    with tabs[4]:
        _render_movement(prof, baselines)

# ---- pre-snap motion + weekly trend ----
with st.expander("Pre-snap motion & week-to-week change"):
    mo = prof["motion"]
    if mo["motion_rate_pct"] is not None:
        st.metric("Motion observed in last ~0.5s before snap", f"{mo['motion_rate_pct']:.0f}%",
                  help=f"n={mo['observable_n']}. " + mo["note"])
    st.caption("⚠️ Lower bound only — tracking starts ~0.5s before the snap, so motion that began "
               "earlier is invisible. Do not read this as full motion usage.")
    tr = pd.DataFrame(prof["weekly_trend"])
    if not tr.empty and not prof["near_constant_route"]:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot(tr.week, tr.route_pct, marker="o", color="#2ca02c")
        for _, r in tr.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r.week, r.route_pct), fontsize=7,
                        textcoords="offset points", xytext=(0, 6), ha="center")
        ax.set_xlabel("week"); ax.set_ylabel("route %"); ax.set_ylim(0, 105)
        st.pyplot(fig); plt.close(fig)
        st.caption("Per-week samples are small — treat swings cautiously.")
    elif prof["near_constant_route"]:
        st.caption("Weekly route-rate trend omitted (near-constant rate).")

st.divider()

# ============================================================
# 3) EXAMPLE PLAYS
# ============================================================
st.markdown("### Example plays")
st.caption("We show one **typical** play (closest to his median route depth) and one clearly "
           "labelled **unusual** play (an outlier). Only the observed snap→endpoint window is drawn.")
typical, unusual = V.pick_example_plays(dscope, int(choice))
if typical is None:
    st.info("No tracked route plays available to visualize for this player.")
else:
    ptabs = st.tabs(["Typical play", "Unusual play (outlier)"])
    for tab, pick, tag in [(ptabs[0], typical, "TYPICAL"), (ptabs[1], unusual, "UNUSUAL")]:
        with tab:
            fig, pm = V.plot_play(pick["gameId"], pick["playId"], focus_nflId=int(choice),
                title=f"{meta['name']} — {P.ALIGN_LABELS.get(pick['align_bucket'], pick['align_bucket'])} "
                      f"· {pick['depth']:.0f} yd route")
            if fig:
                st.pyplot(fig); plt.close(fig)
                st.caption(f"**{tag}** · {pick['why']}. "
                           f"Window drawn: {pm['window_frames']} frames (~{pm['window_seconds']}s), "
                           f"snap → **{pm['end_event']}**. gameId {pick['gameId']}, playId {pick['playId']}.")
                if tag == "UNUSUAL":
                    st.warning("This is an outlier shown for contrast — not representative of his typical usage.")
    with st.expander("Technical note: field coordinates"):
        st.markdown(
            "Diagrams use the NGS tracking coordinate system: the field is 120 yards long "
            "(x) by 53⅓ yards wide (y). **x=0–10** and **x=110–120** are the two end zones, "
            "with goal lines at **x=10** and **x=110**. The on-field yard numbers are the "
            "familiar football labels (midfield = 50), so tracking x=20 is shown as the "
            "“10”, x=60 as the “50”, and so on. Player paths are plotted directly from the "
            "raw coordinates — nothing is shifted or mirrored, so a play can move left or "
            "right depending on which way that offense faced.")

st.divider()

# ============================================================
# 4) PREDICTION (corrected: held-out by default)
# ============================================================
st.markdown("### Pre-snap route-vs-block prediction")
st.caption("A model that, using only pre-snap information, estimates the chance this player runs a "
           "route (vs pass-blocks) on a dropback. Conditional on a passing dropback.")

if meta["position"] == "WR":
    st.info("WRs run routes ~99.8% of the time on dropbacks, so a model adds nothing here. "
            "Prediction is shown only for TE/RB/FB, whose role genuinely varies.")
elif model is None:
    st.warning("Model artifacts not found. Run `python3 scout/model_route_block.py` first.")
else:
    pred = P.holdout_prediction(dscope, model, int(choice),
                                hist["player_rate"], hist["pos_rate"], hist["global_rate"])
    if pred is None:
        st.warning(f"**No held-out data for {meta['name']}.** He appears only in weeks "
                   f"{prof['holdout']['train_weeks_present']} (the model's training period), with "
                   f"**0 observations** in the held-out test weeks {sorted(P.TEST_WEEKS)}. "
                   f"An in-sample estimate would overstate accuracy, so no performance number is shown "
                   f"for him. Treat the overall held-out metrics below as the quality guide.")
    else:
        pc1, pc2 = st.columns(2)
        pc1.metric("Avg predicted route probability",
                   f"{pred['mean_pred_route_prob']:.0f}%",
                   help="Average of the model's per-play route probability across this player's "
                        "HELD-OUT plays (weeks 6–8), which were NOT used for training.")
        pc2.metric("Actual route rate (same plays)",
                   f"{pred['actual_route_pct']:.0f}%",
                   help="What actually happened on those same held-out plays — the honest check.")
        st.success(f"**Held-out result** over weeks {pred['test_weeks']} "
                   f"(**n={pred['n_test']}** plays). The predicted and actual rates above are "
                   f"directly comparable because these plays were not in training.")
        st.caption(f"The model's historical-tendency input used {pred['hist_feature_source']} "
                   f"(={pred['hist_feature_value_pct']:.0f}%), i.e. only information available before "
                   f"these games — no leakage.")
        st.caption("Note: this is the **average over past situations**, not a prediction for one "
                   "specific upcoming play. For a specific play you'd supply that play's pre-snap inputs.")

    # plain-language model quality + expandable detail
    if model_res:
        gbm = model_res.get("gbm", {})
        base = model_res.get("baseline_player_hist", {})
        if gbm and base:
            lift = gbm["roc_auc"] - base["roc_auc"]
            st.markdown(
                f"**How good is the model?** On held-out weeks 6–8 it ranks route-vs-block plays with "
                f"ROC-AUC **{gbm['roc_auc']:.2f}**, better than just using each player's past route rate "
                f"(**{base['roc_auc']:.2f}**) — a **+{lift:.2f}** gain. Because ~84% of these plays are "
                f"routes, raw accuracy is misleading, so we lead with AUC and calibration.")
        with st.expander("Detailed evaluation metrics (held-out weeks 6–8)"):
            friendly = {"baseline_majority": "Baseline: always predict route",
                        "baseline_position_prior": "Baseline: position average",
                        "baseline_player_hist": "Baseline: player's past route rate",
                        "logreg": "Logistic regression",
                        "gbm": "Gradient boosting (main model)"}
            rows = []
            for k in ["baseline_majority","baseline_position_prior","baseline_player_hist","logreg","gbm"]:
                if k in model_res:
                    r = model_res[k]
                    rows.append({"model": friendly[k], "accuracy": r["accuracy"],
                                 "precision": r["precision"], "recall": r["recall"],
                                 "F1": r["f1"], "ROC-AUC": r["roc_auc"], "Brier": r["brier"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.markdown("- **ROC-AUC**: ranking quality (0.5 = coin flip, 1.0 = perfect). "
                        "- **Brier**: probability error, lower is better. "
                        "- **Precision/Recall**: at a 0.5 cutoff on the 'route' class.")
            cal = model_res.get("gbm_calibration")
            if cal:
                figc, axc = plt.subplots(figsize=(4.5, 3.2))
                axc.plot([0,1],[0,1], "--", color="gray", label="perfect")
                axc.plot(cal["mean_predicted"], cal["fraction_positive"], marker="o",
                         color="#1f77b4", label="model")
                axc.set_xlabel("predicted route prob"); axc.set_ylabel("actual route rate")
                axc.set_title("Calibration (held-out)"); axc.legend(fontsize=8)
                st.pyplot(figc); plt.close(figc)
    st.caption("Scope: predicts route-vs-block **conditional on a passing dropback**. The dataset "
               "contains only dropbacks, so it cannot say anything about run-vs-pass tendencies.")

st.divider()

# ============================================================
# methodology (consolidated, expandable)
# ============================================================
with st.expander("Methodology, data scope & limitations"):
    st.markdown(f"""
**Data**: 2022 NFL Big Data Bowl (NYU regional). 2021 season, **weeks 1–8**, 122 games,
**{len(df):,} player-play observations** for WR/TE/RB/FB. Every play is a passing dropback.

**Unit of analysis**: one player on one play (`gameId, playId, nflId`). Tracking frames are
**not** counted as independent samples.

**Alignment** (wide/slot/inline/backfield): from PFF's snap-time `pff_positionLinedUp` label,
validated against tracking lateral position at the snap (wide ≈13.9 yd from center, slot ≈7.3,
inline ≈5.3, backfield ≈3.4 — as expected).

**Route vs block**: from PFF `pff_role`. Used as ground truth for profiles and as the model
**target** — never as a model input.

**Tracking window**: snap is tagged on 99.7% of plays; the pre-snap window is only ~0.5s; the
pass-forward (release) frame exists on ~88%. Arrival/catch events exist on only
~{audit.get('pct_plays_with_arrival_event','?')}% of plays, so we never impute a catch time and
omit separation/arrival metrics. Play visuals draw only the observed snap→endpoint window and
label the actual endpoint event.

**Model**: pre-snap inputs only (position, alignment, down, distance, formation, offensive
personnel, field position, and a historical route rate from **training weeks only**). Trained on
weeks 1–5, evaluated on weeks 6–8, keeping each play wholly within one split.

**What we do NOT claim**: nearest defender is not the assigned coverage defender; `playResult`
is not individual receiving yards; and no suggested coverage/assignment is claimed to be proven
effective. Coaching notes are hypotheses.
""")

st.caption("Observed facts ✅, model predictions, and coaching hypotheses 🧠 are kept visually "
           "distinct throughout. Flag small-sample cells before acting on them.")
