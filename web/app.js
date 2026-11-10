"use strict";
/* Opponent Offensive Tendencies — static front-end.
   Loads web/data/index.json once, then each player's file on demand.
   All calculations come from the Python export (profiles.py); JS only renders. */

const DATA = "./data";
const ALIGN_LABELS = { wide: "Wide (outside)", slot: "Slot", inline: "Inline (tight)",
  backfield: "Backfield", unknown: "Unknown" };
const DD_LABELS = { "1st_long": "1st & 10+", "1st": "1st down", "2nd_short": "2nd & short (≤3)",
  "2nd_med": "2nd & medium (4-7)", "2nd_long": "2nd & long (8+)",
  "3rd/4th_short": "3rd/4th & short (≤3)", "3rd/4th_med": "3rd/4th & medium (4-7)",
  "3rd/4th_long": "3rd/4th & long (8+)", unknown: "unknown" };
const ALIGN_ORDER = ["wide", "slot", "inline", "backfield"];

let INDEX = null;       // index.json
let CURRENT = null;     // loaded player payload
let CURRENT_OPP = null; // selected opponent or null = All
const CACHE = new Map();

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag);
  if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const fmtPct = (v) => (v == null ? "—" : `${Math.round(v)}%`);
const esc = (s) => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
// markdown-ish: **bold** -> <strong>
const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

/* ---------------- boot ---------------- */
async function boot() {
  try {
    INDEX = await fetch(`${DATA}/index.json`).then(r => {
      if (!r.ok) throw new Error(`index.json ${r.status}`); return r.json();
    });
  } catch (e) {
    $("scope-banner").hidden = false;
    $("scope-banner").innerHTML =
      `<div class="scope-note">Could not load data (<code>${esc(e.message)}</code>). ` +
      `If running locally, serve the folder over HTTP (see README) rather than opening the file directly.</div>`;
    return;
  }
  if (INDEX.scope && INDEX.scope.label) $("scope-line").textContent = INDEX.scope.label;
  populateTeams();
  wireFilters();
  // default selection
  onTeamChange();
}

function populateTeams() {
  const sel = $("team-select");
  sel.innerHTML = "";
  Object.keys(INDEX.teams).sort().forEach(t => sel.append(new Option(t, t)));
}

function wireFilters() {
  $("team-select").addEventListener("change", onTeamChange);
  $("pos-select").addEventListener("change", onTeamChange);
  $("player-select").addEventListener("change", onPlayerChange);
  $("opp-select").addEventListener("change", onOppChange);
}

function onTeamChange() {
  const team = $("team-select").value;
  const pos = $("pos-select").value;
  let players = INDEX.teams[team] || [];
  if (pos !== "ALL") players = players.filter(p => p.position === pos);
  const sel = $("player-select");
  sel.innerHTML = "";
  if (!players.length) {
    sel.append(new Option("— no players —", ""));
    clearPlayerUI("No eligible players for this team/position.");
    return;
  }
  players.sort((a, b) => b.plays - a.plays)
    .forEach(p => sel.append(new Option(`${p.name} · ${p.position} · ${p.plays} plays`, p.nflId)));
  onPlayerChange();
}

async function onPlayerChange() {
  const nflId = $("player-select").value;
  if (!nflId) return;
  CURRENT_OPP = null;
  await loadPlayer(nflId);
  populateOpponents();
  render();
}

function onOppChange() {
  CURRENT_OPP = $("opp-select").value || null;
  render();
}

async function loadPlayer(nflId) {
  $("loading").hidden = false;
  if (CACHE.has(nflId)) { CURRENT = CACHE.get(nflId); $("loading").hidden = true; return; }
  try {
    CURRENT = await fetch(`${DATA}/players/${nflId}.json`).then(r => {
      if (!r.ok) throw new Error(`player ${r.status}`); return r.json();
    });
    CACHE.set(nflId, CURRENT);
  } catch (e) {
    CURRENT = null;
    clearPlayerUI(`Could not load this player (${esc(e.message)}).`);
  }
  $("loading").hidden = true;
}

function populateOpponents() {
  const sel = $("opp-select");
  sel.innerHTML = "";
  sel.append(new Option("All opponents", ""));
  if (CURRENT && CURRENT.opponents) {
    CURRENT.opponents.forEach(o => sel.append(new Option(`${o.defense} (${o.plays} plays)`, o.defense)));
  }
  sel.value = "";
}

function clearPlayerUI(msg) {
  ["matchup", "takeaways", "vsleague", "observed", "plays", "prediction"].forEach(id => $(id).hidden = true);
  const b = $("scope-banner"); b.hidden = false;
  b.innerHTML = `<div class="scope-note">${esc(msg)}</div>`;
}

/* ---------------- scope + profile selection ---------------- */
function activeProfile() {
  // All opponents -> CURRENT.all ; specific -> per_opponent[opp].profile
  if (!CURRENT_OPP) return CURRENT.all;
  const po = CURRENT.per_opponent[CURRENT_OPP];
  return po ? po.profile : CURRENT.all;
}
function activeHoldout() {
  if (!CURRENT_OPP) return CURRENT.holdout;
  const po = CURRENT.per_opponent[CURRENT_OPP];
  return po ? po.holdout : null;
}

/* ---------------- master render ---------------- */
function render() {
  if (!CURRENT) return;
  const prof = activeProfile();
  const meta = prof.meta;
  const scopeLabel = CURRENT_OPP ? `vs ${CURRENT_OPP} only` : "All opponents";

  renderScope(meta, scopeLabel);
  renderMatchup();
  renderTakeaways(scopeLabel);
  renderVsLeague();
  renderObserved(prof);
  renderPlays();
  renderPrediction();
}

function renderScope(meta, scopeLabel) {
  const b = $("scope-banner"); b.hidden = false;
  const wkmin = Math.min(...meta.weeks), wkmax = Math.max(...meta.weeks);
  const smallWarn = (CURRENT_OPP && meta.n_plays < 15)
    ? `<div class="scope-note"><span class="badge small-sample">⚠ small sample</span> Only
       ${meta.n_plays} plays vs ${CURRENT_OPP} — treat everything as highly uncertain.</div>` : "";
  b.innerHTML = `
    <div class="b-item"><span class="k">Player</span><span class="v">${esc(meta.name)}</span></div>
    <div class="b-item"><span class="k">Opponent scope</span><span class="v">${esc(scopeLabel)}</span></div>
    <div class="b-item"><span class="k">Position</span><span class="v">${esc(meta.position)}</span></div>
    <div class="b-item"><span class="k">Plays (dropbacks)</span><span class="v">${meta.n_plays}</span></div>
    <div class="b-item"><span class="k">Games · weeks</span><span class="v">${meta.n_games} · wk ${wkmin}–${wkmax}</span></div>
    <div class="scope-note">Scope: <strong>${esc(meta.name)}</strong> · <strong>${esc(scopeLabel)}</strong> ·
      ${meta.n_plays} passing-dropback plays across ${meta.n_games} game(s), weeks ${wkmin}–${wkmax}.
      (Faced ${CURRENT.opponents.length} defense(s) total.)</div>
    ${smallWarn}`;
}

/* ---------------- matchup (opponent selected) ---------------- */
function renderMatchup() {
  const sec = $("matchup");
  if (!CURRENT_OPP) { sec.hidden = true; return; }
  const po = CURRENT.per_opponent[CURRENT_OPP];
  const mc = po ? po.matchup : null;
  sec.hidden = false;
  if (!mc) { sec.innerHTML = `<h2>vs ${esc(CURRENT_OPP)}</h2><p class="muted">No comparison available.</p>`; return; }
  if (mc.n_other === 0) {
    sec.innerHTML = `<h2>${esc(CURRENT.meta.name)} vs ${esc(CURRENT_OPP)}</h2>
      <div class="note info">Only faced ${esc(CURRENT_OPP)} in the sample (n=${mc.n_vs}),
      so there's no "other opponents" group to compare against.</div>`;
    return;
  }
  const rows = mc.comparisons.map(c => {
    const d = c.diff_pp, cls = d > 0 ? "pos-diff" : (d < 0 ? "neg-diff" : "");
    return `<tr><td>${esc(c.tendency)}</td>
      <td class="num">${fmtPct(c.vs_pct)}</td><td class="num">${fmtPct(c.other_pct)}</td>
      <td class="num ${cls}">${d == null ? "—" : (d > 0 ? "+" : "") + d.toFixed(1)}</td>
      <td class="num">${c.n_vs}</td><td class="num">${c.n_other}</td></tr>`;
  }).join("");
  sec.innerHTML = `<h2>${esc(CURRENT.meta.name)} vs ${esc(CURRENT_OPP)} — did his behavior differ?</h2>
    <div class="table-scroll"><table>
      <tr><th>tendency</th><th class="num">vs ${esc(CURRENT_OPP)} %</th><th class="num">vs others %</th>
      <th class="num">diff (pp)</th><th class="num">n (vs)</th><th class="num">n (others)</th></tr>
      ${rows}</table></div>
    <p class="muted small">Plays vs ${esc(CURRENT_OPP)}: <strong>${mc.n_vs}</strong>
      (weeks ${mc.weeks_vs.join(", ")}, ${mc.n_games_vs} game) · vs all others: <strong>${mc.n_other}</strong>.</p>
    ${mc.small_sample ? `<div class="note warn"><span class="badge small-sample">⚠ small sample</span>
      n=${mc.n_vs} &lt; ${mc.min_cell}. Differences are easily driven by chance.</div>` : ""}
    <div class="note hyp">🧠 These are observed differences within one or two games, not proven
      effects of that defense. Game script, teammates, and play-calling are confounded. Hypothesis only.</div>`;
}

/* ---------------- takeaways ---------------- */
function renderTakeaways(scopeLabel) {
  const sec = $("takeaways"); sec.hidden = false;
  $("takeaways-scope").textContent = CURRENT_OPP ? `· ${scopeLabel}` : "";
  const items = CURRENT_OPP ? recomputeTakeawaysForOpp() : CURRENT.takeaways;
  const obs = items.filter(t => t.kind === "observed");
  const hyp = items.filter(t => t.kind === "hypothesis");
  $("takeaways-observed").innerHTML =
    obs.map(t => `<div class="take obs">✅ ${md(t.text)}</div>`).join("") +
    `<p class="muted small">✅ = observed in the data (with sample sizes).</p>`;
  $("takeaways-hyp").innerHTML =
    hyp.map(t => `<div class="take hyp">🧠 <strong>Coaching hypothesis:</strong> ${md(t.text)}</div>`).join("");
}
// For opponent scope we don't have server-side takeaways; derive a concise observed line.
function recomputeTakeawaysForOpp() {
  const prof = activeProfile(), meta = prof.meta, o = prof.overall_role;
  const al = prof.alignment || [];
  const top = al[0];
  const out = [];
  if (top) out.push({ kind: "observed",
    text: `Vs ${CURRENT_OPP}: lines up **${ALIGN_LABELS[top.alignment] || top.alignment}** on `
        + `**${Math.round(top.pct)}%** of ${meta.n_plays} plays (n=${top.n}).` });
  if (meta.position === "WR")
    out.push({ kind: "observed", text: `Ran a route on **${Math.round(o.route_pct)}%** of these dropbacks.` });
  else
    out.push({ kind: "observed", text: `Route **${Math.round(o.route_pct)}%** / block **${Math.round(o.block_pct)}%** `
        + `over n=${o.n} plays vs ${CURRENT_OPP}.` });
  out.push({ kind: "hypothesis", text: `Small single-matchup sample — see the vs-others comparison above before keying on any of this.` });
  return out;
}

/* ---------------- vs league ---------------- */
function renderVsLeague() {
  const sec = $("vsleague");
  const vl = CURRENT.vs_league;
  if (!vl || CURRENT_OPP) { sec.hidden = true; return; }  // league compare is for the full profile
  sec.hidden = false;
  $("vsleague-n").textContent = `(${vl.league_n_players} eligible ${vl.position}s)`;
  const aligns = vl.comparisons.filter(c => c.metric.startsWith("Aligns"));
  if (aligns.length) {
    const big = aligns.reduce((a, b) => Math.abs(b.diff) > Math.abs(a.diff) ? b : a);
    const spot = big.metric.replace("Aligns ", "").toLowerCase();
    const ud = big.diff > 0 ? "more" : "less";
    $("vsleague-headline").innerHTML =
      `<strong>${esc(CURRENT.meta.name)} lines up ${esc(spot)} ${Math.round(big.player_pct)}% of the time `
      + `versus ${Math.round(big.league_pct)}% for ${esc(vl.position)}s overall</strong> — `
      + `${Math.abs(big.diff).toFixed(0)} pts ${ud} than average.`;
  } else $("vsleague-headline").textContent = "";
  const rows = vl.comparisons.map(c => {
    const cls = c.diff > 0 ? "pos-diff" : (c.diff < 0 ? "neg-diff" : "");
    return `<tr><td>${esc(c.metric)}</td><td class="num">${fmtPct(c.player_pct)}</td>
      <td class="num">${fmtPct(c.league_pct)}</td>
      <td class="num ${cls}">${c.diff == null ? "—" : (c.diff > 0 ? "+" : "") + c.diff.toFixed(1)}</td></tr>`;
  }).join("");
  $("vsleague-table").innerHTML =
    `<tr><th>tendency</th><th class="num">${esc(CURRENT.meta.name)} %</th>
     <th class="num">${esc(vl.position)} avg %</th><th class="num">diff (pts)</th></tr>${rows}`;
}

/* ---------------- observed detail ---------------- */
function renderObserved(prof) {
  const sec = $("observed"); sec.hidden = false;
  // alignment bars
  const al = (prof.alignment || []).slice().sort((a, b) => b.pct - a.pct);
  const maxPct = Math.max(100, ...al.map(a => a.pct));
  $("align-chart").innerHTML = al.map(a => `
    <div class="bar-row"><span class="bl">${ALIGN_LABELS[a.alignment] || a.alignment}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(a.pct / maxPct * 100).toFixed(1)}%"></div>
      <span class="bar-val">${a.pct.toFixed(0)}% (n=${a.n})</span></div></div>`).join("");

  // role
  const o = prof.overall_role;
  const base = (INDEX.baselines[prof.meta.position] || {}).route_pct;
  const roleEl = $("role-block");
  if (prof.near_constant_route) {
    const verb = o.route_pct >= 50 ? "runs a route" : "pass-blocks";
    roleEl.innerHTML = `<div class="note ok">Role is essentially fixed: <strong>${verb} on
      ${Math.round(Math.max(o.route_pct, o.block_pct))}%</strong> of dropbacks (n=${o.n}).</div>
      <p class="muted small">Route-rate split charts are omitted (they'd all show ~the same value).
      ${base != null ? `Positional baseline: ${prof.meta.position}s run routes ${Math.round(base)}%.` : ""}</p>`;
  } else {
    const diff = base != null ? (o.route_pct - base) : null;
    roleEl.innerHTML = `
      <div class="metric-row">
        <div class="metric"><div class="m-label">Runs a route</div><div class="m-value">${Math.round(o.route_pct)}%</div>
          <div class="m-sub">95% CI ${o.route_ci[0]}–${o.route_ci[1]}% · n=${o.n}</div></div>
        <div class="metric"><div class="m-label">Pass blocks</div><div class="m-value">${Math.round(o.block_pct)}%</div></div>
      </div>
      ${base != null ? `<p class="muted small">Positional baseline: ${prof.meta.position}s average
        ${Math.round(base)}% (${diff >= 0 ? "+" : ""}${diff.toFixed(0)} pts vs this player).</p>` : ""}`;
  }

  buildSitTabs(prof);
}

function buildSitTabs(prof) {
  const body = $("sit-body");
  const tabs = $("sit-tabs");
  const render = (tab) => {
    if (tab === "align") body.innerHTML = rateTable(prof.route_by_alignment, "align_bucket", ALIGN_LABELS, "alignment");
    else if (tab === "down") body.innerHTML = rateTable(prof.route_by_down, "down", null, "down");
    else if (tab === "dd") body.innerHTML = rateTable(prof.route_by_dd, "dd_bucket", DD_LABELS, "situation");
    else if (tab === "form") body.innerHTML = rateTable(prof.route_by_formation, "offenseFormation", null, "formation");
    else if (tab === "move") body.innerHTML = movementBlock(prof);
  };
  tabs.querySelectorAll(".tab").forEach(b => {
    b.onclick = () => { tabs.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      b.classList.add("active"); render(b.dataset.tab); };
  });
  tabs.querySelector(".tab").classList.add("active");
  render("align");
}

function rateTable(rows, key, labelMap, keyName) {
  if (!rows || !rows.length)
    return `<p class="muted small">Not enough data to break down by ${keyName} (need ≥5 plays per cell).</p>`;
  const bars = rows.map(r => {
    const name = labelMap ? (labelMap[r[key]] || r[key]) : r[key];
    const v = r.rate_pct;
    return `<div class="bar-row"><span class="bl">${esc(String(name))}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${v}%"></div>
      <span class="bar-val">${v.toFixed(0)}% · n=${r.n}</span></div></div>`;
  }).join("");
  return `<div>${bars}</div><p class="muted small">Bar = route rate. Each cell shows sample size;
    95% CI available in the full data. Cells with n&lt;5 hidden.</p>`;
}

function movementBlock(prof) {
  const mv = prof.movement, pos = prof.meta.position;
  if (!mv || mv.n_route_tracked < 5 || mv.mean_depth_at_release_yd == null)
    return `<p class="muted small">Not enough tracked route plays to summarize movement reliably.</p>`;
  const base = INDEX.baselines[pos] || {};
  const cmp = (v, b, u) => (b == null || v == null) ? "—"
    : (Math.abs(v - b) < 0.3 ? `≈ ${pos} avg (${b}${u})`
      : `${v - b > 0 ? "+" : ""}${(v - b).toFixed(1)}${u} vs ${pos} avg (${b}${u})`);
  const rows = [
    ["Mean route depth at release (yd downfield)", mv.mean_depth_at_release_yd, cmp(mv.mean_depth_at_release_yd, base.mean_depth_yd, " yd")],
    ["Median route depth (yd)", mv.median_depth_yd, "—"],
    ["Mean top speed (yd/s)", mv.mean_max_speed_yd_s, cmp(mv.mean_max_speed_yd_s, base.mean_top_speed, " yd/s")],
    ["Mean lateral break (yd)", mv.mean_lateral_break_yd, cmp(mv.mean_lateral_break_yd, base.mean_lateral_break, " yd")],
    ["Route plays tracked (n)", mv.n_route_tracked, "—"],
  ];
  const depth = (prof.depth_dist || []);
  const order = ["short (<5)", "intermediate (5-12)", "deep (12+)"];
  const ddrows = order.map(b => depth.find(d => d.band === b)).filter(Boolean)
    .map(d => `<div class="bar-row"><span class="bl">${d.band}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${d.pct}%;background:#9467bd"></div>
      <span class="bar-val">${d.pct.toFixed(0)}% · n=${d.n}</span></div></div>`).join("");
  return `<div class="table-scroll"><table><tr><th>metric</th><th class="num">value</th><th>vs baseline</th></tr>
    ${rows.map(r => `<tr><td>${r[0]}</td><td class="num">${r[1] == null ? "—" : r[1]}</td><td>${r[2]}</td></tr>`).join("")}
    </table></div>
    ${ddrows ? `<h3>Route depth distribution</h3>${ddrows}` : ""}
    <p class="muted small">${esc(mv.note)} Comparison is to the league average for this position.</p>`;
}

/* ---------------- example plays ---------------- */
function renderPlays() {
  const sec = $("plays");
  // Example plays are always the player's overall representative plays. We note this
  // when an opponent is selected rather than exporting per-opponent play excerpts.
  const plays = CURRENT.example_plays || [];
  if (!plays.length) { sec.hidden = true; return; }
  sec.hidden = false;
  const tabs = $("play-tabs"); const body = $("play-body");
  tabs.innerHTML = plays.map((p, i) =>
    `<button class="tab${i === 0 ? " active" : ""}" data-i="${i}">${p.which === "typical" ? "Typical play" : "Unusual play (outlier)"}</button>`).join("");
  const draw = (i) => {
    const p = plays[i];
    body.innerHTML = `<div class="play-wrap">${fieldSVG(p, CURRENT.meta.name)}</div>
      ${playLegend(p)}
      <p class="muted small"><strong>${p.which.toUpperCase()}</strong> · ${esc(p.why)}.
      Window drawn: ${p.window_frames} frames (~${p.window_seconds}s), snap → <strong>${esc(p.end_event)}</strong>.
      gameId ${p.gameId}, playId ${p.playId}.</p>
      ${p.which === "unusual" ? `<div class="note warn">This is an outlier shown for contrast — not representative.</div>` : ""}`;
  };
  tabs.querySelectorAll(".tab").forEach(b => b.onclick = () => {
    tabs.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); draw(+b.dataset.i);
  });
  // note when scoped to an opponent (plays shown are the overall representatives)
  const noteId = "plays-opp-note";
  const existing = document.getElementById(noteId);
  if (existing) existing.remove();
  if (CURRENT_OPP) {
    const n = el("p", "muted small");
    n.id = noteId;
    n.textContent = `Example plays shown are ${CURRENT.meta.name}'s overall representative plays, not filtered to ${CURRENT_OPP}.`;
    sec.insertBefore(n, tabs);
  }
  draw(0);
}

/* ---- football field SVG (tracking coords 0-120 x 0-53.3) ---- */
function fieldSVG(p, name) {
  const W = 120, H = 53.3, scale = 9; // px per yard (viewBox uses yards directly)
  const yard = {20:"10",30:"20",40:"30",50:"40",60:"50",70:"40",80:"30",90:"20",100:"10"};
  const colors = { focus: "#f5c518", offense: "#4C9BE8", defense: "#e05a4f", ball: "#6b3f1d" };
  let s = `<svg class="field-svg" viewBox="0 0 ${W} ${H + 6}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Play diagram for ${esc(name)}">`;
  // turf + end zones
  s += `<rect x="0" y="0" width="${W}" height="${H}" fill="#3a7d44"/>`;
  s += `<rect x="0" y="0" width="10" height="${H}" fill="#21522c"/>`;
  s += `<rect x="110" y="0" width="10" height="${H}" fill="#21522c"/>`;
  s += `<text x="5" y="${H/2}" fill="#fff" font-size="3" text-anchor="middle" transform="rotate(-90 5 ${H/2})" opacity="0.85" font-weight="bold">END ZONE</text>`;
  s += `<text x="115" y="${H/2}" fill="#fff" font-size="3" text-anchor="middle" transform="rotate(-90 115 ${H/2})" opacity="0.85" font-weight="bold">END ZONE</text>`;
  // 5-yard lines
  for (let x = 15; x < 110; x += 5) s += `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="#fff" stroke-opacity="0.22" stroke-width="0.2"/>`;
  // goal lines
  [10, 110].forEach(x => s += `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="#fff" stroke-opacity="0.9" stroke-width="0.5"/>`);
  // yard numbers both sidelines
  Object.entries(yard).forEach(([x, lbl]) => {
    s += `<text x="${x}" y="7" fill="#fff" font-size="3" text-anchor="middle" opacity="0.85">${lbl}</text>`;
    s += `<text x="${x}" y="${H - 4}" fill="#fff" font-size="3" text-anchor="middle" opacity="0.85">${lbl}</text>`;
  });
  // LOS
  s += `<line x1="${p.los_x}" y1="0" x2="${p.los_x}" y2="${H}" stroke="#ffd166" stroke-width="0.5" stroke-dasharray="1.2 0.8"/>`;
  s += `<text x="${p.los_x}" y="-1.2" fill="#ffd166" font-size="3" text-anchor="middle" font-weight="bold">LOS</text>`;
  // direction arrow
  const dx = p.play_direction === "right" ? 8 : -8;
  s += `<line x1="${p.los_x}" y1="${H - 2.5}" x2="${p.los_x + dx}" y2="${H - 2.5}" stroke="#fff" stroke-width="0.5" marker-end="url(#arrow)"/>`;
  s += `<defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="2" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 z" fill="#fff"/></marker></defs>`;
  s += `<text x="${p.los_x + dx / 2}" y="${H - 3.5}" fill="#fff" font-size="2.6" text-anchor="middle" font-weight="bold">offense</text>`;
  // tracks: draw non-focus first, focus on top
  const order = { defense: 1, offense: 1, ball: 2, focus: 3 };
  p.tracks.slice().sort((a, b) => order[a.kind] - order[b.kind]).forEach(t => {
    const pts = t.pts.map(pt => `${pt[0]},${pt[1]}`).join(" ");
    if (t.kind === "focus") {
      s += `<polyline points="${pts}" fill="none" stroke="${colors.focus}" stroke-width="0.9"/>`;
      const a = t.pts[0], z = t.pts[t.pts.length - 1];
      s += `<circle cx="${a[0]}" cy="${a[1]}" r="1.1" fill="${colors.focus}" stroke="#000" stroke-width="0.2"/>`;
      s += `<path d="${star(z[0], z[1], 1.4)}" fill="${colors.focus}" stroke="#000" stroke-width="0.2"/>`;
    } else {
      const w = t.kind === "ball" ? 0.4 : 0.3;
      s += `<polyline points="${pts}" fill="none" stroke="${colors[t.kind]}" stroke-width="${w}" opacity="0.7"/>`;
    }
  });
  s += `</svg>`;
  return s;
}
function star(cx, cy, r) {
  let d = ""; for (let i = 0; i < 10; i++) {
    const ang = Math.PI / 5 * i - Math.PI / 2, rr = i % 2 ? r * 0.45 : r;
    d += (i ? "L" : "M") + (cx + rr * Math.cos(ang)).toFixed(2) + "," + (cy + rr * Math.sin(ang)).toFixed(2);
  } return d + "Z";
}
function playLegend(p) {
  return `<div class="play-legend">
    <span class="lg"><span class="sw" style="background:#f5c518"></span>selected player</span>
    <span class="lg"><span class="sw" style="background:#4C9BE8"></span>offense</span>
    <span class="lg"><span class="sw" style="background:#e05a4f"></span>defense</span>
    <span class="lg"><span class="sw" style="background:#6b3f1d"></span>ball</span>
    <span class="lg">● snap &nbsp; ★ endpoint (${esc(p.end_event)})</span></div>`;
}

/* ---------------- prediction ---------------- */
function renderPrediction() {
  const sec = $("prediction"); sec.hidden = false;
  const meta = CURRENT.meta;
  const body = $("pred-body");
  if (meta.position === "WR") {
    body.innerHTML = `<div class="note info">WRs run routes ~99.8% of the time on dropbacks, so a
      model adds nothing here. Shown only for TE/RB/FB, whose role genuinely varies.</div>`;
    $("pred-detail").hidden = true; return;
  }
  const pred = activeHoldout();
  const scopeTxt = CURRENT_OPP ? ` vs ${CURRENT_OPP}` : "";
  if (!pred) {
    body.innerHTML = `<div class="note warn"><strong>No held-out data${esc(scopeTxt)}.</strong>
      These plays fall in the model's training period (weeks 1–5) or there are none in the held-out
      weeks 6–8, so no honest performance number is shown here. See overall metrics below.</div>`;
  } else {
    body.innerHTML = `
      <div class="metric-row">
        <div class="metric"><div class="m-label">Avg predicted route prob</div>
          <div class="m-value">${Math.round(pred.mean_pred_route_prob)}%</div>
          <div class="m-sub">held-out weeks ${pred.test_weeks.join(", ")}</div></div>
        <div class="metric"><div class="m-label">Actual route rate (same plays)</div>
          <div class="m-value">${Math.round(pred.actual_route_pct)}%</div>
          <div class="m-sub">the honest check</div></div>
      </div>
      <div class="note ok"><strong>Held-out result</strong> over weeks ${pred.test_weeks.join(", ")}
        (<strong>n=${pred.n_test}</strong> plays). Predicted vs actual are comparable because these
        plays were not used in training.</div>
      <p class="muted small">Historical-tendency input used ${esc(pred.hist_feature_source)}
        (=${Math.round(pred.hist_feature_value_pct)}%) — only pre-game info, no leakage.
        This is an average over past situations, not a prediction for one specific upcoming play.</p>`;
  }
  // overall model quality
  const mr = INDEX.model_results, friendly = {
    baseline_majority: "Baseline: always predict route",
    baseline_position_prior: "Baseline: position average",
    baseline_player_hist: "Baseline: player's past route rate",
    logreg: "Logistic regression", gbm: "Gradient boosting (main model)" };
  const order = ["baseline_majority", "baseline_position_prior", "baseline_player_hist", "logreg", "gbm"];
  $("pred-table").innerHTML =
    `<tr><th>model</th><th class="num">accuracy</th><th class="num">precision</th><th class="num">recall</th>
     <th class="num">F1</th><th class="num">ROC-AUC</th><th class="num">Brier</th></tr>` +
    order.filter(k => mr[k]).map(k => { const r = mr[k];
      return `<tr><td>${friendly[k]}</td><td class="num">${r.accuracy}</td><td class="num">${r.precision}</td>
      <td class="num">${r.recall}</td><td class="num">${r.f1}</td><td class="num">${r.roc_auc}</td>
      <td class="num">${r.brier}</td></tr>`; }).join("");
  $("pred-detail").hidden = false;
}

boot();
