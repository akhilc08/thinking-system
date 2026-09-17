import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "runs/run1_export.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/dashboard.html"

with open(SRC) as f:
    raw = f.read()

# Embedded verbatim as the text content of a <script type="application/json">
# block (a raw-text element, so no JS-string escaping applies) -- the only
# hazard is the literal end-tag sequence, which the HTML parser would treat
# as closing the script early regardless of JSON/JS context.
embedded = raw.replace("</script", "<\\/script")

TEMPLATE = r"""<title>Context Compaction Swarm</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {
  --bg: #f5f6fa;
  --surface: #ffffff;
  --surface-2: #ececf3;
  --border: #dcdce6;
  --text: #1b1d27;
  --text-dim: #5b5d6e;
  --accent: #4b3fd6;
  --accent-soft: #e8e5fb;
  --good: #158a5f;
  --good-soft: #dff3ea;
  --warn: #b6780f;
  --warn-soft: #f7ecd7;
  --bad: #c0392b;
  --bad-soft: #fbe4e1;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1016; --surface: #171923; --surface-2: #1f2230; --border: #2b2e3d;
    --text: #e7e8f0; --text-dim: #9a9db3; --accent: #8b7bff; --accent-soft: #262244;
    --good: #34caa0; --good-soft: #123529; --warn: #e0a83f; --warn-soft: #3a2c11;
    --bad: #ff6b5e; --bad-soft: #3a1613;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1016; --surface: #171923; --surface-2: #1f2230; --border: #2b2e3d;
  --text: #e7e8f0; --text-dim: #9a9db3; --accent: #8b7bff; --accent-soft: #262244;
  --good: #34caa0; --good-soft: #123529; --warn: #e0a83f; --warn-soft: #3a2c11;
  --bad: #ff6b5e; --bad-soft: #3a1613;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  padding: 0 16px 64px; margin: 0;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1, h2, h3, .disp {
  font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif;
  font-weight: 700; text-wrap: balance; letter-spacing: -0.01em;
}
.mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
a { color: var(--accent); }

/* ---- Header ---- */
header.hero { padding: 40px 0 24px; border-bottom: 1px solid var(--border); }
header.hero h1 { font-size: 2.1rem; margin: 0 0 8px; }
header.hero .topic { color: var(--text-dim); font-size: 0.98rem; max-width: 72ch; line-height: 1.5; margin: 0 0 18px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px; border-radius: 999px;
  background: var(--surface-2); border: 1px solid var(--border); font-size: 0.82rem; color: var(--text-dim);
}
.chip b { color: var(--text); font-family: "IBM Plex Mono", monospace; font-weight: 600; }
.chip.status-completed { background: var(--good-soft); color: var(--good); border-color: transparent; }

section { padding: 36px 0; border-bottom: 1px solid var(--border); }
section:last-child { border-bottom: none; }
section > h2 { font-size: 1.3rem; margin: 0 0 4px; }
section > .sub { color: var(--text-dim); font-size: 0.88rem; margin: 0 0 20px; }

/* ---- Badges ---- */
.badge { display: inline-block; padding: 2px 8px; border-radius: 5px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; white-space: nowrap; }
.badge.strat { background: var(--accent-soft); color: var(--accent); }
.badge.st-selected_final { background: var(--good-soft); color: var(--good); }
.badge.st-not_selected_final { background: var(--surface-2); color: var(--text-dim); }
.badge.st-pruned_pool_trim { background: var(--warn-soft); color: var(--warn); }
.badge.st-pruned_reflection { background: var(--bad-soft); color: var(--bad); }
.badge.st-active { background: var(--accent-soft); color: var(--accent); }
.elo { font-family: "IBM Plex Mono", monospace; font-weight: 600; }

/* ---- Winners ---- */
.winners { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.winner-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px;
  border-top: 3px solid var(--good); display: flex; flex-direction: column; gap: 10px;
}
.winner-card .rank-row { display: flex; align-items: baseline; justify-content: space-between; }
.winner-card .rank { font-family: "IBM Plex Sans Condensed"; font-weight: 700; font-size: 1.5rem; color: var(--good); }
.winner-card .elo { font-size: 1.05rem; }
.winner-card p { font-size: 0.9rem; line-height: 1.55; margin: 0; }
.winner-card summary { cursor: pointer; color: var(--accent); font-size: 0.82rem; margin-top: 4px; }
.winner-card details p { margin-top: 8px; }

/* ---- Chart ---- */
.chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; overflow-x: auto; }
.chart-card svg { display: block; min-width: 560px; width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px; font-size: 0.78rem; color: var(--text-dim); }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.legend i { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }

/* ---- Rounds ---- */
.round-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 18px; overflow: hidden; }
.round-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; padding: 16px 20px; background: var(--surface-2); }
.round-head .num { font-family: "IBM Plex Sans Condensed"; font-weight: 700; font-size: 1.1rem; }
.round-body { padding: 18px 20px; display: flex; flex-direction: column; gap: 18px; }
blockquote { margin: 0; padding: 10px 14px; border-left: 3px solid var(--accent); background: var(--accent-soft); border-radius: 0 6px 6px 0; font-size: 0.85rem; color: var(--text); line-height: 1.5; }
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.8rem; color: var(--text-dim); }
.stat-row b { color: var(--text); font-family: "IBM Plex Mono", monospace; }

.debate-list { display: flex; flex-direction: column; gap: 8px; }
.debate { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.debate summary { cursor: pointer; font-size: 0.85rem; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; list-style: none; }
.debate summary::-webkit-details-marker { display: none; }
.debate .side { padding: 2px 6px; border-radius: 4px; background: var(--surface-2); }
.debate .side.winner { background: var(--good-soft); color: var(--good); font-weight: 600; }
.debate .delta { font-family: "IBM Plex Mono", monospace; font-size: 0.75rem; color: var(--text-dim); }
.debate .reasoning { margin-top: 8px; font-size: 0.82rem; color: var(--text-dim); line-height: 1.5; }

.mini-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px; }
.mini-card { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-size: 0.82rem; }
.mini-card.pruned { border-left: 3px solid var(--warn); }
.mini-card .head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; gap: 6px; }
.mini-card p { margin: 0; color: var(--text-dim); line-height: 1.45; }
.mini-card .reason { margin-top: 6px; font-size: 0.75rem; color: var(--warn); }
.mini-card summary { cursor: pointer; color: var(--accent); font-size: 0.75rem; }

/* ---- Structured idea fields ---- */
.fields { display: flex; flex-direction: column; gap: 10px; }
.fields .field b {
  display: block; font-family: "IBM Plex Sans Condensed"; font-weight: 600; font-size: 0.68rem;
  text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-dim); margin-bottom: 2px;
}
.fields .field span { font-size: 0.86rem; line-height: 1.5; }
.hypothesis-line { font-size: 0.92rem; line-height: 1.5; margin: 0; }

/* ---- Browser ---- */
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.filters button {
  font: inherit; font-size: 0.8rem; padding: 6px 12px; border-radius: 999px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text-dim); cursor: pointer;
}
.filters button.active { background: var(--accent); color: white; border-color: var(--accent); }
.browser-list { display: flex; flex-direction: column; gap: 8px; }
.row { border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; background: var(--surface); }
.row[hidden] { display: none; }
.row summary { list-style: none; cursor: pointer; display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.row summary::-webkit-details-marker { display: none; }
.row .snippet { color: var(--text-dim); font-size: 0.85rem; flex: 1 1 260px; min-width: 0; }
.row .full { margin-top: 10px; font-size: 0.88rem; line-height: 1.6; color: var(--text); }
.row .reason { margin-top: 8px; font-size: 0.78rem; color: var(--warn); }
.row .meta { font-size: 0.72rem; color: var(--text-dim); margin-top: 8px; }

footer { padding: 24px 0 0; color: var(--text-dim); font-size: 0.78rem; }
</style>

<div class="wrap">
  <header class="hero">
    <h1>Context Compaction Swarm</h1>
    <p class="topic" id="topic-text"></p>
    <div class="chips" id="meta-chips"></div>
  </header>

  <section>
    <h2>Final ranking</h2>
    <p class="sub">Diversity-aware top-<span id="k-final"></span>, selected from the surviving pool by Elo and pairwise dissimilarity.</p>
    <div class="winners" id="winners"></div>
  </section>

  <section>
    <h2>Elo over rounds</h2>
    <p class="sub">Every line is one idea. A line that stops before the last round was cut — hover the end for why.</p>
    <div class="chart-card">
      <svg id="chart" viewBox="0 0 900 360" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div class="legend" id="legend"></div>
  </section>

  <section>
    <h2>Round by round</h2>
    <p class="sub">Everything generated, debated, evolved, and cut — in order.</p>
    <div id="rounds"></div>
  </section>

  <section>
    <h2>Every idea (<span id="total-count"></span>)</h2>
    <p class="sub">The full pool this run ever held, including what got pruned and why.</p>
    <div class="filters" id="filters"></div>
    <div class="browser-list" id="browser"></div>
  </section>

  <footer>Exported from RunStore — candidates, critiques, debates, Elo snapshots and final selection are all persisted in SQLite; this page is a static read of that export.</footer>
</div>

<script id="run-data" type="application/json">__RUN_DATA__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById('run-data').textContent);
  const $ = (sel, el) => (el || document).querySelector(sel);
  const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
  const esc = (s) => (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const truncate = (s, n) => (s && s.length > n) ? s.slice(0, n).trim() + '\u2026' : (s || '');

  const byId = {};
  data.candidates.forEach(c => byId[c.candidate_id] = c);

  const STATUS_LABEL = {
    selected_final: 'Selected', not_selected_final: 'Not selected', pruned_pool_trim: 'Cut (pool cap)',
    pruned_reflection: 'Cut (reflection)', active: 'Active',
  };
  const STATUS_COLOR_VAR = {
    selected_final: '--good', not_selected_final: '--text-dim', pruned_pool_trim: '--warn',
    pruned_reflection: '--bad', active: '--accent',
  };

  const FIELD_LABELS = [
    ['problem', 'Problem/gap'], ['mechanism', 'Mechanism'], ['method', 'Method'],
    ['prediction', 'Prediction'], ['falsification', 'Falsification'], ['novelty_delta', 'Related-work delta'],
  ];
  const hasStructure = (c) => c.structured && c.structured.hypothesis;
  const titleFor = (c) => hasStructure(c) && c.structured.title ? c.structured.title : null;
  const snippetFor = (c, n) => hasStructure(c) ? truncate(c.structured.hypothesis, n) : truncate(c.text, n);
  const fieldsHtml = (c) => {
    if (!hasStructure(c)) return `<p class="hypothesis-line">${esc(c.text)}</p>`;
    const rows = FIELD_LABELS
      .filter(([key]) => c.structured[key])
      .map(([key, label]) => `<div class="field"><b>${esc(label)}</b><span>${esc(c.structured[key])}</span></div>`)
      .join('');
    return `<p class="hypothesis-line">${esc(c.structured.hypothesis)}</p><div class="fields">${rows}</div>`;
  };

  // ---- Header ----
  $('#topic-text').textContent = data.run.topic;
  const chips = $('#meta-chips');
  const chip = (label, value, extraClass) => {
    const c = el('span', 'chip' + (extraClass ? ' ' + extraClass : ''));
    c.innerHTML = `${esc(label)} <b>${esc(String(value))}</b>`;
    chips.appendChild(c);
  };
  chip('status', data.run.status, 'status-' + data.run.status);
  chip('domain', data.run.domain);
  chip('subfield', data.run.subfield);
  chip('rounds', data.rounds.length);
  chip('ideas produced', data.candidates.length);
  chip('model calls', data.run.total_model_calls);
  chip('vs AgentIdeaBench Active', data.run.compute_multiplier.toFixed(1) + 'x');

  // ---- Winners ----
  $('#k-final').textContent = data.final_selection.length;
  const winners = $('#winners');
  data.final_selection.forEach(f => {
    const c = byId[f.candidate_id];
    if (!c) return;
    const card = el('div', 'winner-card');
    const title = titleFor(c);
    card.innerHTML = `
      <div class="rank-row"><span class="rank">#${f.rank}</span><span class="elo">${Math.round(c.elo)} elo</span></div>
      ${title ? `<h3 style="margin:0;font-size:1.02rem;">${esc(title)}</h3>` : ''}
      <div><span class="badge strat">${esc(c.source_strategy)}</span> <span class="badge st-${c.status}">${esc(STATUS_LABEL[c.status] || c.status)}</span></div>
      ${fieldsHtml(c)}
    `;
    winners.appendChild(card);
  });

  // ---- Elo chart ----
  const snapsByCandidate = {};
  data.elo_snapshots.forEach(s => {
    (snapsByCandidate[s.candidate_id] = snapsByCandidate[s.candidate_id] || []).push(s);
  });
  Object.values(snapsByCandidate).forEach(arr => arr.sort((a, b) => a.round_num - b.round_num));

  const roundsPresent = [...new Set(data.elo_snapshots.map(s => s.round_num))].sort((a, b) => a - b);
  const allElo = data.elo_snapshots.map(s => s.elo);
  const eloMin = Math.min(...allElo, 1150) - 20, eloMax = Math.max(...allElo, 1250) + 20;
  const W = 900, H = 360, padL = 46, padR = 16, padT = 16, padB = 34;
  const xFor = (round) => roundsPresent.length > 1
    ? padL + (round - roundsPresent[0]) / (roundsPresent[roundsPresent.length - 1] - roundsPresent[0]) * (W - padL - padR)
    : (padL + W - padR) / 2;
  const yFor = (elo) => padT + (1 - (elo - eloMin) / (eloMax - eloMin)) * (H - padT - padB);

  const svg = $('#chart');
  let svgHtml = '';
  // gridlines + axis
  roundsPresent.forEach(r => {
    const x = xFor(r);
    svgHtml += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB}" stroke="var(--border)" stroke-width="1"/>`;
    svgHtml += `<text x="${x}" y="${H - padB + 20}" text-anchor="middle" font-size="12" fill="var(--text-dim)" font-family="IBM Plex Mono">round ${r}</text>`;
  });
  for (let g = 0; g <= 4; g++) {
    const val = eloMin + (g / 4) * (eloMax - eloMin);
    const y = yFor(val);
    svgHtml += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--border)" stroke-width="1" stroke-dasharray="2,3"/>`;
    svgHtml += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="var(--text-dim)" font-family="IBM Plex Mono">${Math.round(val)}</text>`;
  }

  Object.entries(snapsByCandidate).forEach(([cid, points]) => {
    const cand = byId[cid];
    if (!cand) return;
    const colorVar = `var(${STATUS_COLOR_VAR[cand.status] || '--accent'})`;
    const pts = points.map(p => `${xFor(p.round_num)},${yFor(p.elo)}`).join(' ');
    const isSelected = cand.status === 'selected_final';
    svgHtml += `<polyline points="${pts}" fill="none" stroke="${colorVar}" stroke-width="${isSelected ? 2.6 : 1.4}" opacity="${isSelected ? 1 : 0.55}"><title>${esc(snippetFor(cand, 140))} \u2014 ${esc(STATUS_LABEL[cand.status])}</title></polyline>`;
    const last = points[points.length - 1];
    const cut = cand.status.startsWith('pruned');
    svgHtml += `<circle cx="${xFor(last.round_num)}" cy="${yFor(last.elo)}" r="${cut ? 4 : 3}" fill="${colorVar}"><title>${esc(snippetFor(cand, 140))} \u2014 ${esc(STATUS_LABEL[cand.status])}${cand.pruned_reason ? ' (' + esc(cand.pruned_reason) + ')' : ''}</title></circle>`;
    if (cut) {
      svgHtml += `<text x="${xFor(last.round_num) + 6}" y="${yFor(last.elo) - 6}" font-size="11" fill="${colorVar}">\u2715</text>`;
    }
  });
  svg.innerHTML = svgHtml;

  const legend = $('#legend');
  [['selected_final', 'Selected (final top-k)'], ['not_selected_final', 'Survived, not selected'],
   ['pruned_pool_trim', 'Cut by pool cap'], ['pruned_reflection', 'Cut by reflection']].forEach(([status, label]) => {
    if (!data.candidates.some(c => c.status === status)) return;
    const s = el('span');
    s.innerHTML = `<i style="background:var(${STATUS_COLOR_VAR[status]})"></i>${esc(label)}`;
    legend.appendChild(s);
  });

  // ---- Rounds ----
  const roundsDiv = $('#rounds');
  const debatesByRound = {};
  data.debates.forEach(d => (debatesByRound[d.round_num] = debatesByRound[d.round_num] || []).push(d));
  const candidatesByRound = {};
  data.candidates.forEach(c => (candidatesByRound[c.round_num] = candidatesByRound[c.round_num] || []).push(c));

  data.rounds.forEach(r => {
    const created = candidatesByRound[r.round_num] || [];
    const debates = debatesByRound[r.round_num] || [];
    const cut = created.filter(c => c.status.startsWith('pruned'));
    const card = el('div', 'round-card');

    const head = el('div', 'round-head');
    head.innerHTML = `<span class="num">Round ${r.round_num}</span>
      <span class="badge strat">${created.length} generated</span>
      <span class="badge strat">${debates.length} debates</span>
      ${cut.length ? `<span class="badge st-pruned_pool_trim">${cut.length} cut</span>` : ''}`;
    card.appendChild(head);

    const body = el('div', 'round-body');

    if (r.meta_review_guidance) {
      body.appendChild(el('blockquote', null, esc(r.meta_review_guidance)));
    }

    if (debates.length) {
      const list = el('div', 'debate-list');
      debates.forEach(d => {
        const a = byId[d.candidate_a_id], b = byId[d.candidate_b_id];
        if (!a || !b) return;
        const det = el('details', 'debate');
        const aWon = d.winner_id === a.candidate_id;
        det.innerHTML = `
          <summary>
            <span class="side ${aWon ? 'winner' : ''}">${esc(snippetFor(a, 60))}</span>
            <span>vs</span>
            <span class="side ${!aWon ? 'winner' : ''}">${esc(snippetFor(b, 60))}</span>
            <span class="delta">${Math.round(d.elo_a_before)}\u2192${Math.round(d.elo_a_after)} / ${Math.round(d.elo_b_before)}\u2192${Math.round(d.elo_b_after)}</span>
          </summary>
          <div class="reasoning">${esc(d.reasoning)}</div>`;
        list.appendChild(det);
      });
      body.appendChild(list);
    }

    const grid = el('div', 'mini-grid');
    created.forEach(c => {
      const card2 = el('div', 'mini-card' + (c.status.startsWith('pruned') ? ' pruned' : ''));
      card2.innerHTML = `
        <div class="head"><span class="badge strat">${esc(c.source_strategy)}</span><span class="elo">${Math.round(c.elo)}</span></div>
        <p>${esc(snippetFor(c, 160))}</p>
        <details><summary>full idea</summary>${fieldsHtml(c)}</details>
        ${c.pruned_reason ? `<div class="reason">cut: ${esc(c.pruned_reason)}</div>` : ''}
      `;
      grid.appendChild(card2);
    });
    body.appendChild(grid);

    card.appendChild(body);
    roundsDiv.appendChild(card);
  });

  // ---- Browser ----
  $('#total-count').textContent = data.candidates.length;
  const statuses = [...new Set(data.candidates.map(c => c.status))];
  const filters = $('#filters');
  let activeFilter = 'all';
  const allBtn = el('button', 'active', 'All');
  allBtn.dataset.status = 'all';
  filters.appendChild(allBtn);
  statuses.forEach(s => {
    const b = el('button', null, STATUS_LABEL[s] || s);
    b.dataset.status = s;
    filters.appendChild(b);
  });

  const browser = $('#browser');
  const sorted = [...data.candidates].sort((a, b) => b.elo - a.elo);
  sorted.forEach(c => {
    const row = el('details', 'row');
    row.dataset.status = c.status;
    row.innerHTML = `
      <summary>
        <span class="elo">${Math.round(c.elo)}</span>
        <span class="badge st-${c.status}">${esc(STATUS_LABEL[c.status] || c.status)}</span>
        <span class="badge strat">${esc(c.source_strategy)}</span>
        <span class="snippet">${titleFor(c) ? esc(titleFor(c)) + ' \u2014 ' : ''}${esc(snippetFor(c, 110))}</span>
      </summary>
      <div class="full">${fieldsHtml(c)}</div>
      ${c.pruned_reason ? `<div class="reason">Cut: ${esc(c.pruned_reason)}</div>` : ''}
      <div class="meta">Round ${c.round_num} \u00b7 id ${c.candidate_id.slice(0, 8)}${c.parent_ids && c.parent_ids.length ? ' \u00b7 built from ' + c.parent_ids.length + ' parent idea(s)' : ''}</div>
    `;
    browser.appendChild(row);
  });

  filters.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    activeFilter = btn.dataset.status;
    [...filters.children].forEach(b => b.classList.toggle('active', b === btn));
    [...browser.children].forEach(row => {
      row.hidden = activeFilter !== 'all' && row.dataset.status !== activeFilter;
    });
  });
})();
</script>
"""

html = TEMPLATE.replace("__RUN_DATA__", embedded)
with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({len(html)} bytes)")
