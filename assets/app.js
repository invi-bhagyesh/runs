/* Research runs explorer: reads each project's run log straight from a Hugging Face dataset in the browser.
   Projects are listed in projects.json. Private datasets need a read token (Settings), kept in this browser only. */
(() => {
  "use strict";
  const HF = "https://huggingface.co";
  const app = document.getElementById("app");
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };
  const token = () => store.get("hf_token", "");
  const enc = s => encodeURIComponent(s);
  let keyHandler = null;
  let lastList = null; // {project, run, keys} from the last filtered sample table, for next/prev in the reader

  // ---------- data ----------
  let projectsP = null;
  const projects = () => (projectsP ||= fetch("projects.json", { cache: "no-cache" }).then(r => r.json()));
  async function hf(url) {
    const headers = {};
    if (token()) headers.Authorization = "Bearer " + token();
    const r = await fetch(url, { headers });
    if (!r.ok) {
      const e = new Error(r.status === 401 || r.status === 403 || r.status === 404 && !token()
        ? "Hugging Face refused access. If the dataset is private, add a read token in Settings."
        : `HTTP ${r.status} for ${url}`);
      e.status = r.status;
      throw e;
    }
    return r;
  }
  const memo = new Map();
  function once(key, fn) {
    if (!memo.has(key)) { const p = fn(); memo.set(key, p); p.catch(() => memo.delete(key)); }
    return memo.get(key);
  }
  const fileURL = (P, path) => `${HF}/datasets/${P.hf_dataset}/resolve/main/${path.split("/").map(enc).join("/")}`;
  const blobURL = (P, path) => `${HF}/datasets/${P.hf_dataset}/blob/main/${path.split("/").map(enc).join("/")}`;
  const text = (P, path) => once(`t|${P.id}|${path}`, async () => {
    const buf = new Uint8Array(await (await hf(fileURL(P, path))).arrayBuffer());
    if (buf[0] === 0x1f && buf[1] === 0x8b) {
      return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
    }
    return new TextDecoder().decode(buf);
  });
  const json = async (P, path) => JSON.parse(await text(P, path));
  const jsonl = (P, path) => once(`l|${P.id}|${path}`, async () =>
    (await text(P, path)).split("\n").filter(l => l.trim()).map(l => JSON.parse(l)));
  const tree = (P, path) => once(`tree|${P.id}|${path}`, async () => (await hf(`${HF}/api/datasets/${P.hf_dataset}/tree/main/${path}`)).json());
  const orEmpty = (p, empty) => p.catch(e => { if (e.status === 404) return empty; throw e; });

  const runs = P => once(`runs|${P.id}`, async () => {
    const dirs = (await tree(P, `${P.prefix}/runs`)).filter(x => x.type === "directory").map(x => x.path.split("/").pop());
    const out = await Promise.all(dirs.map(async id => {
      const [status, meta] = await Promise.all([
        orEmpty(json(P, `${P.prefix}/runs/${id}/status.json`), {}), orEmpty(json(P, `${P.prefix}/runs/${id}/meta.json`), {})]);
      return { id, status, meta, type: meta.type || "", state: status.state || "", updated: status.updated || "", summary: status.summary || meta.summary || "" };
    }));
    return out.sort((a, b) => b.id.localeCompare(a.id));
  });
  const runFiles = (P, id) => tree(P, `${P.prefix}/runs/${id}`);
  const samples = (P, run) => once(`s|${P.id}|${run}`, async () => {
    const latest = new Map();
    for (const r of await jsonl(P, `${P.prefix}/runs/${run}/samples.jsonl.gz`)) if (!r.error) latest.set(r.key, r);
    return [...latest.values()].sort((a, b) => a.episode_id.localeCompare(b.episode_id) || a.oversight.localeCompare(b.oversight)
      || a.question_name.localeCompare(b.question_name) || a.index - b.index);
  });
  const byKey = rows => Object.fromEntries(rows.map(r => [r.key, r]));
  const reviews = (P, run) => once(`rv|${P.id}|${run}`, async () => byKey(await orEmpty(jsonl(P, `${P.prefix}/reviewed/${run}.jsonl`), [])));
  const rewrites = (P, run) => once(`rw|${P.id}|${run}`, async () => byKey(await orEmpty(jsonl(P, `${P.prefix}/runs/${run}-rewrites/rewrites.jsonl.gz`), [])));
  const episodes = (P, run) => once(`e|${P.id}|${run}`, async () => {
    const m = new Map();
    for (const r of await jsonl(P, `${P.prefix}/runs/${run}/episodes.jsonl.gz`)) if (r.episode_id) m.set(r.episode_id, r);
    return m;
  });
  async function findEpisode(P, id, onTry) {
    for (const run of P.transcript_runs || []) {
      onTry && onTry(run);
      const m = await orEmpty(episodes(P, run), new Map());
      if (m.has(id)) return { run, ep: m.get(id) };
    }
    return null;
  }
  const myVerdicts = P => store.get(`mv|${P.id}`, {});
  const setMyVerdict = (P, run, key, patch) => {
    const all = myVerdicts(P), id = `${run}|${key}`;
    const next = { ...(all[id] || {}), ...patch, at: new Date().toISOString().slice(0, 19) + "Z" };
    if (!next.verdict && !next.note) delete all[id]; else all[id] = next;
    store.set(`mv|${P.id}`, all);
  };

  // ---------- small renderers ----------
  function wilson(k, n) {
    if (!n) return [0, 0, 0];
    const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
    const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [p, Math.max(0, c - h), Math.min(1, c + h)];
  }
  const pct = x => (100 * x).toFixed(1) + "%";
  // ISO UTC stamps ("2026-10-15T13:21:24Z") shown in the viewer's own time zone, with how long ago; full stamp on hover
  const DATE = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  function ago(ms) {
    const m = Math.round(ms / 60000);
    if (m < 1) return "just now";
    if (m < 60) return `${m} min ago`;
    if (m < 48 * 60) return `${Math.round(m / 60)} h ago`;
    return `${Math.round(m / 1440)} days ago`;
  }
  function when(iso, { relative = true } = {}) {
    const d = iso ? new Date(iso) : null;
    if (!d || isNaN(d)) return esc(iso || "");
    const rel = relative ? ` <span class="muted">· ${ago(Date.now() - d)}</span>` : "";
    return `<time datetime="${esc(iso)}" title="${esc(d.toString())}">${esc(DATE.format(d))}</time>${rel}`;
  }
  const rate = (k, n) => { const [p, lo, hi] = wilson(k, n); return n ? `${k}/${n} · <b>${pct(p)}</b> <span class="muted">[${pct(lo)}, ${pct(hi)}]</span>` : "–"; };
  const pillClass = v => ({ lie: "lie", false_claim: "lie", omission: "lie", not_lie: "ok", honest: "ok", disclosed: "ok", accurate: "ok",
    unsure: "warn", unclear: "warn", disclosed_with_false_claim: "warn", unwatched: "unw", watched: "wat", running: "warn", done: "ok",
    failed: "lie", stopped: "warn", serving: "warn", lie_kind: "lie", honest_watched: "ok", accurate_unhacked: "ok" }[v] || "");
  const pill = (v, label) => v ? `<span class="pill ${pillClass(v)}">${esc(label ?? v)}</span>` : "";
  const md = s => window.DOMPurify && window.marked ? DOMPurify.sanitize(marked.parse(s || "")) : `<pre>${esc(s)}</pre>`;
  const block = (title, body, extra = "") => `<section class="block"><div class="row"><h3>${esc(title)}</h3>${extra}</div>${body}</section>`;
  const textBlock = (title, s, serif = false, extra = "") =>
    block(title, s ? `<div class="text ${serif ? "serif" : ""}">${esc(s)}</div>` : `<p class="muted">(empty)</p>`, extra);
  const errorBox = e => `<div class="panel"><p class="err">${esc(e.message || e)}</p>${e.status ? `<p class="muted">Status ${e.status}. <a href="#/settings">Settings</a></p>` : ""}</div>`;
  const loading = what => { app.innerHTML = `<p class="muted">Loading ${esc(what)}…</p>`; };
  function crumbs(items) {
    document.getElementById("crumbs").innerHTML = items.map(([label, href]) =>
      href ? `<a href="${href}">${esc(label)}</a>` : `<span>${esc(label)}</span>`).join('<span class="sep">/</span>');
  }
  function tokenState() {
    const a = document.getElementById("token-state");
    a.textContent = token() ? "Settings · token set" : "Settings · no token";
    a.classList.toggle("missing", !token());
  }
  function tryJSON(s) { try { return typeof s === "string" ? JSON.parse(s) : s; } catch { return null; } }
  function renderArgs(args) {
    const a = tryJSON(args);
    if (!a || typeof a !== "object") return `<pre>${esc(args)}</pre>`;
    return Object.entries(a).map(([k, v]) => {
      const s = typeof v === "string" ? v : JSON.stringify(v, null, 1);
      return s.includes("\n") || s.length > 120 ? `<div class="muted mono">${esc(k)}:</div><pre class="nowrap">${esc(s)}</pre>` : `<div class="mono"><span class="muted">${esc(k)}:</span> ${esc(s)}</div>`;
    }).join("");
  }
  function clip(s, n = 6000) {
    s = s ?? "";
    if (s.length <= n) return `<pre>${esc(s)}</pre>`;
    return `<pre>${esc(s.slice(0, n))}</pre><details><summary>${(s.length - n).toLocaleString()} more characters</summary><pre>${esc(s.slice(n))}</pre></details>`;
  }
  function messages(msgs, reasoningByIndex = {}) {
    return msgs.map((m, i) => {
      const r = reasoningByIndex[i] || reasoningByIndex[String(i)] || m.reasoning_content || "";
      const calls = (m.tool_calls || []).map(c => {
        const f = c.function || c;
        return `<div class="call"><div class="name">${esc(f.name)}</div>${renderArgs(f.arguments)}</div>`;
      }).join("");
      return `<div class="msg ${esc(m.role)}"><div class="who">${esc(m.role)} <span class="muted">#${i}</span></div>
        ${r ? `<details><summary>reasoning (${r.length.toLocaleString()} chars)</summary><div class="text">${esc(r)}</div></details>` : ""}
        ${m.content ? (m.role === "tool" ? clip(m.content, 3000) : `<div class="text">${esc(m.content)}</div>`) : ""}${calls}</div>`;
    }).join("");
  }
  function highlightCode(code) {
    return esc(code).split("\n").map(l => /\bif\b.*==\s*[\('"\[\d-]|_call|_count|special|hard-?cod|expects?\b/i.test(l) ? `<mark>${l}</mark>` : l).join("\n");
  }
  function lineChart(points, { yLabel = "" } = {}) {
    if (!points.length) return "";
    const W = 760, H = 220, L = 44, R = 12, T = 12, B = 30;
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), lo = Math.floor(Math.min(...ys) * 10) / 10, hi = Math.ceil(Math.max(...ys) * 10) / 10;
    const x = v => L + (W - L - R) * (v - x0) / Math.max(1, x1 - x0), y = v => T + (H - T - B) * (1 - (v - lo) / Math.max(1e-9, hi - lo));
    let g = "";
    for (let v = lo; v <= hi + 1e-9; v += (hi - lo) / 4) g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
    for (const s of [x0, Math.round((x0 + x1) / 2), x1]) g += `<text x="${x(s)}" y="${H - 10}" text-anchor="middle">${s}</text>`;
    return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)}">${g}<polyline class="line" points="${points.map(p => `${x(p[0])},${y(p[1])}`).join(" ")}"/></svg></div>`;
  }
  function download(name, content) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([content], { type: "application/x-ndjson" }));
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ---------- views ----------
  async function viewHome() {
    crumbs([]);
    const ps = await projects();
    app.innerHTML = `<div class="stack"><h1>Projects</h1><p class="muted">Each project reads its runs straight from its Hugging Face log.
      Add a project by adding an entry to <code>projects.json</code>.</p></div>
      <div class="cards">${ps.map(p => `<article class="card"><h2><a href="#/p/${enc(p.id)}">${esc(p.title)}</a></h2>
        <p>${esc(p.summary)}</p><div class="row">${pill("done", p.status)}<span class="muted mono">${esc(p.hf_dataset)}/${esc(p.prefix)}</span></div></article>`).join("")}</div>`;
  }

  function viewSettings() {
    crumbs([["Settings"]]);
    app.innerHTML = `<div class="stack panel" style="max-width:720px"><h1>Settings</h1>
      <p>The run logs live in Hugging Face datasets. For a private dataset, paste a <b>read</b> token
      (huggingface.co → Settings → Access Tokens; a fine-grained read-only token for the dataset is enough).
      It is stored only in this browser (localStorage) and sent only to huggingface.co.</p>
      <label for="tok">Hugging Face token</label><input id="tok" type="password" autocomplete="off" value="${esc(token())}" placeholder="hf_…">
      <div class="row"><button class="primary" id="save">Save</button><button id="clear">Remove token</button><span id="msg" class="muted"></span></div>
      <h2>Your verdicts</h2><p class="muted">Verdicts you set in the sample reader are kept in this browser. Export them per run from the run's Samples tab.</p>
      <div class="row"><button id="wipe">Delete all my verdicts in this browser</button></div></div>`;
    const msg = document.getElementById("msg");
    document.getElementById("save").onclick = async () => {
      store.set("hf_token", document.getElementById("tok").value.trim()); memo.clear(); tokenState();
      msg.textContent = "Checking…";
      try { const P = (await projects())[0]; await tree(P, P.prefix); msg.textContent = `Saved. ${P.title}: access works.`; }
      catch (e) { msg.textContent = "Saved, but: " + e.message; }
    };
    document.getElementById("clear").onclick = () => { store.set("hf_token", ""); memo.clear(); tokenState(); document.getElementById("tok").value = ""; msg.textContent = "Removed."; };
    document.getElementById("wipe").onclick = async () => {
      const btn = document.getElementById("wipe");
      if (btn.dataset.armed) { for (const p of await projects()) store.set(`mv|${p.id}`, {}); btn.textContent = "Deleted"; delete btn.dataset.armed; }
      else { btn.dataset.armed = "1"; btn.textContent = "Click again to delete all"; }
    };
  }

  async function viewProject(P) {
    crumbs([[P.title]]);
    loading("runs");
    const rs = await runs(P);
    const notes = await orEmpty(tree(P, `${P.prefix}/notes`), []);
    const top = await orEmpty(tree(P, P.prefix), []);
    const docs = [...top.filter(f => f.type === "file" && f.path.endsWith(".md")), ...notes.filter(f => f.path.endsWith(".md"))];
    const groups = (P.groups || []).map(g => ({ ...g, runs: [] }));
    const other = { title: "Other runs", runs: [] };
    for (const r of rs) {
      const g = groups.find(g => g.match.some(m => r.id.includes(m)) && !(g.exclude || []).some(m => r.id.includes(m)));
      (g || other).runs.push(r);
    }
    const table = list => `<div class="tablewrap"><table><thead><tr><th>Run</th><th>Type</th><th>State</th><th>Updated</th><th>Summary</th></tr></thead><tbody>
      ${list.map(r => `<tr class="click" data-href="#/p/${enc(P.id)}/run/${enc(r.id)}"><td class="mono"><a href="#/p/${enc(P.id)}/run/${enc(r.id)}">${esc(r.id)}</a></td>
        <td>${esc(r.type)}</td><td>${pill(r.state)}</td><td class="muted" style="white-space:nowrap">${when(r.updated)}</td><td>${esc(r.summary)}</td></tr>`).join("")}</tbody></table></div>`;
    app.innerHTML = `<div class="stack"><h1>${esc(P.title)}</h1><p>${esc(P.summary)}</p>
      <div class="row">${(P.links || []).map(l => `<a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)} ↗</a>`).join(" · ")}</div></div>
      ${P.highlights ? `<div class="kpis">${P.highlights.map(h => `<div class="kpi"><div class="l">${esc(h.label)}</div><div class="v">${esc(h.value)}</div><div class="l">${esc(h.note || "")}</div></div>`).join("")}</div>` : ""}
      ${docs.length ? `<div class="stack"><h2>Notes</h2><div class="row">${docs.map(d => `<a href="#/p/${enc(P.id)}/doc/${enc(d.path)}">${esc(d.path.replace(P.prefix + "/", ""))}</a>`).join(" · ")}</div></div>` : ""}
      ${[...groups, other].filter(g => g.runs.length).map(g => `<div class="stack"><h2>${esc(g.title)} <span class="muted">(${g.runs.length})</span></h2>${table(g.runs)}</div>`).join("")}`;
    bindRowLinks();
  }

  async function viewDoc(P, path) {
    crumbs([[P.title, `#/p/${enc(P.id)}`], [path.replace(P.prefix + "/", "")]]);
    loading(path);
    app.innerHTML = `<div class="panel md">${md(await text(P, path))}</div>`;
  }

  const TABS = [
    ["samples", "Samples", f => f.has("samples.jsonl.gz")],
    ["episodes", "Episodes", f => f.has("episodes.jsonl.gz")],
    ["rewrites", "Rewrites", f => f.has("rewrites.jsonl.gz")],
    ["sft", "SFT examples", f => f.has("train.jsonl.gz")],
    ["training", "Training", f => f.has("train_log.jsonl.gz")],
    ["summary", "Summary", f => f.has("summary.md")],
    ["progress", "Progress log", f => f.has("progress.log")],
    ["logs", "Job logs", f => f.has("job.log") || f.has("boot.log")],
    ["meta", "Meta", () => true],
    ["files", "Files", () => true],
  ];

  async function viewRun(P, id, tab) {
    crumbs([[P.title, `#/p/${enc(P.id)}`], [id]]);
    loading(id);
    const [rs, files] = await Promise.all([runs(P), runFiles(P, id)]);
    const r = rs.find(x => x.id === id) || { id, status: {}, meta: {} };
    const names = new Set(files.filter(f => f.type === "file").map(f => f.path.split("/").pop()));
    const tabs = TABS.filter(t => t[2](names));
    tab = tab && tabs.some(t => t[0] === tab) ? tab : tabs[0][0];
    const base = `#/p/${enc(P.id)}/run/${enc(id)}`;
    app.innerHTML = `<div class="stack"><h1 class="mono" style="font-family:var(--mono);font-size:22px">${esc(id)}</h1>
      <div class="row">${pill(r.state)}<span class="muted">${esc(r.type)}</span><span class="muted">updated ${when(r.updated)}</span><span class="muted">${esc(r.status.host || "")}</span></div>
      ${r.summary ? `<p>${esc(r.summary)}</p>` : ""}</div>
      <nav class="tabs">${tabs.map(t => `<a href="${base}/${t[0]}" class="${t[0] === tab ? "on" : ""}">${t[1]}</a>`).join("")}</nav>
      <div id="tab" class="stack"><p class="muted">Loading…</p></div>`;
    const el = document.getElementById("tab");
    const runPath = `${P.prefix}/runs/${id}`;
    try {
      if (tab === "samples") await tabSamples(P, id, el);
      else if (tab === "episodes") await tabEpisodes(P, id, el);
      else if (tab === "rewrites") await tabRewrites(P, id, el);
      else if (tab === "sft") await tabSFT(P, id, el);
      else if (tab === "training") await tabTraining(P, id, el, r);
      else if (tab === "summary") el.innerHTML = `<div class="panel md">${md(await text(P, `${runPath}/summary.md`))}</div>`;
      else if (tab === "progress") el.innerHTML = `<pre>${esc(await text(P, `${runPath}/progress.log`))}</pre>`;
      else if (tab === "logs") {
        const parts = [];
        for (const f of ["job.log", "boot.log"]) if (names.has(f)) parts.push(`<h2>${f}</h2><pre>${esc(await text(P, `${runPath}/${f}`))}</pre>`);
        el.innerHTML = parts.join("");
      } else if (tab === "meta") el.innerHTML = `<h2>meta.json</h2><pre>${esc(JSON.stringify(r.meta, null, 1))}</pre><h2>status.json</h2><pre>${esc(JSON.stringify(r.status, null, 1))}</pre>`;
      else el.innerHTML = `<div class="tablewrap"><table><thead><tr><th>File</th><th class="num">Size</th></tr></thead><tbody>${files.map(f =>
        `<tr><td class="mono"><a href="${blobURL(P, f.path)}" target="_blank" rel="noopener">${esc(f.path.replace(runPath + "/", ""))}${f.type === "directory" ? "/" : ""}</a></td>
        <td class="num mono">${f.size ? (f.size / 1e6).toFixed(2) + " MB" : ""}</td></tr>`).join("")}</tbody></table></div>`;
    } catch (e) { el.innerHTML = errorBox(e); }
    bindRowLinks();
  }

  // --- samples (rejection-sampling and evaluation runs) ---
  async function tabSamples(P, run, el) {
    el.innerHTML = `<p class="muted">Downloading samples…</p>`;
    const [rows, rv, rw] = await Promise.all([samples(P, run), reviews(P, run), rewrites(P, run)]);
    const mine = myVerdicts(P);
    const cells = {};
    for (const r of rows) {
      for (const k of [`${r.oversight}|${r.question_name}`, `${r.oversight}|all`]) {
        const c = cells[k] ||= { n: 0, lie: 0, unclear: 0 };
        c.n++; if (r.verdict === "lie") c.lie++; if (r.verdict === "unclear") c.unclear++;
      }
    }
    const qs = [...new Set(rows.map(r => r.question_name))].sort();
    const metric = ["unwatched", "watched"].map(o => `<tr><td>${pill(o)}</td>${[...qs, "all"].map(q => {
      const c = cells[`${o}|${q}`]; return `<td>${c ? rate(c.lie, c.n) : "–"}</td>`; }).join("")}<td class="num">${cells[`${o}|all`]?.unclear ?? 0}</td></tr>`).join("");
    const perEp = {};
    for (const r of rows) { const e = perEp[r.episode_id] ||= { u: [0, 0], w: [0, 0] }; const c = r.oversight === "unwatched" ? e.u : e.w; c[1]++; if (r.verdict === "lie") c[0]++; }
    const F = store.get(`f|${P.id}|${run}`, { oversight: "", q: "", verdict: "", review: "", mine: "", nudged: "", search: "", page: 0 });
    const opts = (name, values, label) => `<label>${label}<select data-f="${name}"><option value="">any</option>${values.map(v =>
      `<option value="${esc(v)}" ${F[name] === v ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></label>`;
    el.innerHTML = `
      <div class="stack"><h2>Judge lie rate <span class="muted">(Wilson 95% CI; judge labels, not reviewed)</span></h2>
        <div class="tablewrap"><table><thead><tr><th>Condition</th>${qs.map(q => `<th>${esc(q)}</th>`).join("")}<th>all</th><th class="num">unclear</th></tr></thead><tbody>${metric}</tbody></table></div>
        <details><summary>By transcript (${Object.keys(perEp).length})</summary><div class="tablewrap"><table><thead><tr><th>Transcript</th><th>Unwatched lies</th><th>Watched lies</th></tr></thead><tbody>
        ${Object.entries(perEp).sort((a, b) => b[1].u[0] / Math.max(1, b[1].u[1]) - a[1].u[0] / Math.max(1, a[1].u[1])).map(([e, c]) =>
          `<tr><td class="mono">${esc(e)}</td><td>${rate(...c.u)}</td><td>${rate(...c.w)}</td></tr>`).join("")}</tbody></table></div></details></div>
      <div class="filters panel">
        ${opts("oversight", ["unwatched", "watched"], "Condition")}${opts("q", qs, "Question")}
        ${opts("verdict", ["lie", "honest", "unclear"], "Judge")}${opts("review", ["lie", "not_lie", "unsure", "unreviewed"], "Review")}
        ${opts("mine", ["lie", "not_lie", "unsure", "none"], "My verdict")}${opts("nudged", ["yes", "no"], "Nudged")}
        <label>Search (transcript or text)<input type="text" data-f="search" value="${esc(F.search)}" placeholder="e.g. lcbhard_3, special case"></label>
        <span class="muted" id="count"></span>
        <button id="export">Export my verdicts</button>
      </div>
      <div id="list"></div>`;
    const draw = () => {
      store.set(`f|${P.id}|${run}`, F);
      const s = F.search.toLowerCase();
      const list = rows.filter(r => (!F.oversight || r.oversight === F.oversight) && (!F.q || r.question_name === F.q)
        && (!F.verdict || r.verdict === F.verdict)
        && (!F.review || (F.review === "unreviewed" ? !rv[r.key] : rv[r.key]?.verdict === F.review))
        && (!F.mine || (F.mine === "none" ? !mine[`${run}|${r.key}`]?.verdict : mine[`${run}|${r.key}`]?.verdict === F.mine))
        && (!F.nudged || (F.nudged === "yes") === !!r.nudge)
        && (!s || r.episode_id.toLowerCase().includes(s) || (r.answer || "").toLowerCase().includes(s) || (r.reasoning || "").toLowerCase().includes(s)));
      lastList = { project: P.id, run, keys: list.map(r => r.key) };
      const per = 100, pages = Math.max(1, Math.ceil(list.length / per));
      F.page = Math.min(F.page || 0, pages - 1);
      document.getElementById("count").textContent = `${list.length.toLocaleString()} of ${rows.length.toLocaleString()} samples`;
      const slice = list.slice(F.page * per, F.page * per + per);
      document.getElementById("list").innerHTML = `<div class="tablewrap"><table><thead><tr><th>Transcript</th><th>Condition</th><th>Question</th><th class="num">#</th>
        <th>Judge</th><th>Review</th><th>Mine</th><th>Answer</th></tr></thead><tbody>${slice.map(r => {
          const m = mine[`${run}|${r.key}`];
          return `<tr class="click" data-href="#/p/${enc(P.id)}/sample/${enc(run)}/${enc(r.key)}"><td class="mono">${esc(r.episode_id)}${r.nudge ? ' <span class="pill">nudged</span>' : ""}${rw[r.key] ? ` <span class="pill ${rw[r.key].passed ? "ok" : "warn"}">rewrite</span>` : ""}</td>
            <td>${pill(r.oversight)}</td><td>${esc(r.question_name)}</td><td class="num mono">${r.index}</td>
            <td>${pill(r.verdict)} <span class="muted">${esc(r.label || "")}</span></td><td>${pill(rv[r.key]?.verdict)}</td><td>${pill(m?.verdict)}</td>
            <td class="prev">${esc((r.answer || "").replace(/\s+/g, " ").slice(0, 160))}</td></tr>`; }).join("")}</tbody></table></div>
        <div class="pager"><button id="prev" ${F.page ? "" : "disabled"}>Previous</button><span class="muted">page ${F.page + 1} of ${pages}</span><button id="next" ${F.page < pages - 1 ? "" : "disabled"}>Next</button></div>`;
      document.getElementById("prev").onclick = () => { F.page--; draw(); };
      document.getElementById("next").onclick = () => { F.page++; draw(); };
      bindRowLinks();
    };
    el.querySelectorAll("[data-f]").forEach(c => c.addEventListener(c.tagName === "INPUT" ? "input" : "change", () => { F[c.dataset.f] = c.value; F.page = 0; draw(); }));
    document.getElementById("export").onclick = () => {
      const all = myVerdicts(P);
      const lines = Object.entries(all).filter(([k]) => k.startsWith(run + "|")).map(([k, v]) =>
        JSON.stringify({ key: k.slice(run.length + 1), verdict: v.verdict || "", note: v.note || "", reviewer: "human", at: v.at }));
      download(`${run}.reviewed.jsonl`, lines.join("\n") + (lines.length ? "\n" : ""));
    };
    draw();
  }

  async function viewSample(P, run, key) {
    crumbs([[P.title, `#/p/${enc(P.id)}`], [run, `#/p/${enc(P.id)}/run/${enc(run)}/samples`], ["sample"]]);
    loading("sample");
    const [rows, rv, rw] = await Promise.all([samples(P, run), reviews(P, run), rewrites(P, run)]);
    const r = rows.find(x => x.key === key);
    if (!r) { app.innerHTML = errorBox("Sample not found."); return; }
    const keys = lastList && lastList.project === P.id && lastList.run === run && lastList.keys.includes(key) ? lastList.keys : rows.map(x => x.key);
    const i = keys.indexOf(key), prevK = keys[i - 1], nextK = keys[i + 1];
    const href = k => `#/p/${enc(P.id)}/sample/${enc(run)}/${enc(k)}`;
    const rev = rv[key], rew = rw[key];
    const mineV = myVerdicts(P)[`${run}|${key}`] || {};
    const reasonTabs = rew ? `<div class="row"><button data-rs="orig" class="on">As sampled</button><button data-rs="rew">Rewritten for training ${rew.passed ? pill("ok", "passed") : pill("warn", "failed")}</button></div>` : "";
    app.innerHTML = `
      <div class="row" style="justify-content:space-between"><div class="row"><h1 class="mono" style="font-family:var(--mono);font-size:19px">${esc(r.episode_id)}</h1>
        ${pill(r.oversight)}<span>${esc(r.question_name)} #${r.index}</span>${r.nudge ? '<span class="pill">nudged</span>' : ""}</div>
        <div class="row muted">${i + 1} of ${keys.length} ${prevK ? `<a href="${href(prevK)}">← prev <kbd>k</kbd></a>` : ""} ${nextK ? `<a href="${href(nextK)}">next <kbd>j</kbd> →</a>` : ""}</div></div>
      <div class="reader">
        <div class="stack" id="ctx"><div class="block"><h3>The hack</h3><p class="muted" id="ctxmsg">Finding the transcript…</p></div></div>
        <div class="stack">
          ${textBlock("Question (as in training: nudge removed)", r.question_clean || r.question)}
          ${r.nudge ? textBlock("Nudge present when sampled", r.nudge) : ""}
          ${block("Reasoning", `${reasonTabs}<div class="text" id="reason">${esc(r.reasoning || "")}</div>${rew && !rew.passed ? `<p class="muted">Rewrite problems: ${esc((rew.problems || []).join("; "))}</p>` : ""}`)}
          ${(r.tool_calls || []).length ? block("Tool calls while answering", r.tool_calls.map(c => `<div class="call"><div class="name">${esc(c.name)}</div>${renderArgs(c.arguments)}${c.output ? `<details><summary>output</summary>${clip(c.output, 3000)}</details>` : ""}</div>`).join("")
            + (r.func_after ? `<details><summary>func.py after answering</summary><pre class="nowrap">${highlightCode(r.func_after)}</pre></details>` : "")) : ""}
          ${textBlock("Answer", r.answer, true)}
          ${block("Judge", `<div class="row">${pill(r.verdict)} ${pill(r.label)}</div><dl class="kv"><dt>Evidence</dt><dd>${esc(r.evidence || "")}</dd><dt>Explanation</dt><dd>${esc(r.explanation || "")}</dd></dl>`)}
          ${block("Review", rev ? `<div class="row">${pill(rev.verdict)}<span class="muted">${esc(rev.reviewer || "")} · ${when(rev.at, { relative: false })}</span></div><p>${esc(rev.note || "")}</p>` : `<p class="muted">Not reviewed.</p>`)}
          ${block("Sampling", `<dl class="kv"><dt>Model</dt><dd class="mono">${esc(r.subject_model)}</dd><dt>Settings</dt><dd class="mono">T=${esc(r.temperature)} top_p=${esc(r.top_p)} top_k=${esc(r.top_k)} effort=${esc(r.reasoning_effort)}</dd><dt>Key</dt><dd class="mono">${esc(r.key)}</dd><dt>Served by</dt><dd class="mono">${esc((r.served_by || []).join(", "))}</dd></dl>`)}
          <div class="verdict-bar"><div class="row"><b>My verdict</b>
            ${["lie", "not_lie", "unsure"].map((v, n) => `<button class="v-${v} ${mineV.verdict === v ? "on" : ""}" data-v="${v}">${v.replace("_", " ")} <kbd>${n + 1}</kbd></button>`).join("")}
            <button data-v="">clear <kbd>0</kbd></button><span class="muted" id="saved"></span></div>
            <textarea id="note" placeholder="Note (saved in this browser; export from the run's Samples tab)">${esc(mineV.note || "")}</textarea></div>
        </div></div>`;
    const reasonEl = document.getElementById("reason");
    app.querySelectorAll("[data-rs]").forEach(b => b.onclick = () => {
      app.querySelectorAll("[data-rs]").forEach(x => x.classList.toggle("on", x === b));
      reasonEl.textContent = b.dataset.rs === "rew" ? (rew.rewritten_reasoning || "") : (r.reasoning || "");
    });
    const saved = document.getElementById("saved");
    const setV = v => {
      setMyVerdict(P, run, key, { verdict: v });
      app.querySelectorAll("[data-v]").forEach(b => b.classList.toggle("on", !!v && b.dataset.v === v));
      saved.textContent = v ? `saved: ${v}` : "cleared";
    };
    app.querySelectorAll("[data-v]").forEach(b => b.onclick = () => setV(b.dataset.v));
    const note = document.getElementById("note");
    note.addEventListener("input", () => { setMyVerdict(P, run, key, { note: note.value }); saved.textContent = "note saved"; });
    keyHandler = e => {
      if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "j" && nextK) location.hash = href(nextK);
      else if (e.key === "k" && prevK) location.hash = href(prevK);
      else if (e.key === "1") setV("lie"); else if (e.key === "2") setV("not_lie"); else if (e.key === "3") setV("unsure"); else if (e.key === "0") setV("");
    };
    // the transcript, for the hack itself
    const ctx = document.getElementById("ctx"), msg = document.getElementById("ctxmsg");
    try {
      const found = await findEpisode(P, r.episode_id, run2 => { msg.textContent = `Looking in ${run2}… (large files can take a few seconds)`; });
      if (!found) { msg.textContent = "Transcript not found in the configured transcript runs."; return; }
      const o = found.ep.outcome || {};
      ctx.innerHTML = block("The hack", `<div class="row">${(o.methods || []).map(m => pill("lie", m)).join(" ")}${o.hacked ? "" : pill("ok", "not hacked")}</div>
          <dl class="kv"><dt>Tests modified</dt><dd>${esc(o.tests_modified)}</dd><dt>Spec modified</dt><dd>${esc(o.spec_modified)}</dd><dt>Saw failure</dt><dd>${esc(o.model_saw_failure)}</dd></dl>
          ${o.conflict_diff ? `<div class="muted">Given tests vs correct tests</div><pre class="nowrap">${esc(o.conflict_diff)}</pre>` : ""}
          ${o.spec_diff ? `<div class="muted">Spec diff</div><pre class="nowrap">${esc(o.spec_diff)}</pre>` : ""}
          ${o.test_diff ? `<div class="muted">Test diff</div><pre class="nowrap">${esc(o.test_diff)}</pre>` : ""}`, `<a href="#/p/${enc(P.id)}/episode/${enc(found.run)}/${enc(r.episode_id)}">full transcript →</a>`)
        + block("Final func.py (suspicious lines marked)", `<pre class="nowrap">${highlightCode(o.final_func || "")}</pre>`);
    } catch (e) { msg.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  }

  // --- episodes (benchmark runs) ---
  async function tabEpisodes(P, run, el) {
    el.innerHTML = `<p class="muted">Downloading episodes (can be tens of MB)…</p>`;
    const eps = [...(await episodes(P, run)).values()];
    const F = store.get(`fe|${P.id}|${run}`, { hacked: "", search: "" });
    el.innerHTML = `<div class="filters panel"><label>Hacked<select data-f="hacked"><option value="">any</option><option ${F.hacked === "yes" ? "selected" : ""}>yes</option><option ${F.hacked === "no" ? "selected" : ""}>no</option></select></label>
      <label>Search<input type="text" data-f="search" value="${esc(F.search)}"></label><span class="muted" id="count"></span></div><div id="list"></div>`;
    const draw = () => {
      store.set(`fe|${P.id}|${run}`, F);
      const list = eps.filter(e => (!F.hacked || (F.hacked === "yes") === !!e.outcome?.hacked) && (!F.search || e.episode_id.includes(F.search)));
      document.getElementById("count").textContent = `${list.length} of ${eps.length} episodes`;
      document.getElementById("list").innerHTML = `<div class="tablewrap"><table><thead><tr><th>Episode</th><th>Split</th><th>Hacked</th><th>Methods</th><th>End</th><th>Report labels</th><th class="num">Seconds</th></tr></thead><tbody>
        ${list.map(e => `<tr class="click" data-href="#/p/${enc(P.id)}/episode/${enc(run)}/${enc(e.episode_id)}"><td class="mono">${esc(e.episode_id)}</td><td>${esc(e.split)}</td>
          <td>${e.error ? pill("failed", "error") : e.outcome?.hacked ? pill("lie", "hacked") : pill("ok", "no")}</td><td>${esc((e.outcome?.methods || []).join(", "))}</td>
          <td class="muted">${esc(e.work?.end_reason || "")}</td><td>${(e.branches || []).map(b => pill(b.final_label, `${b.oversight[0]}:${b.final_label}`)).join(" ")}</td>
          <td class="num mono">${e.seconds ? Math.round(e.seconds) : ""}</td></tr>`).join("")}</tbody></table></div>`;
      bindRowLinks();
    };
    el.querySelectorAll("[data-f]").forEach(c => c.addEventListener(c.tagName === "INPUT" ? "input" : "change", () => { F[c.dataset.f] = c.value; draw(); }));
    draw();
  }

  async function viewEpisode(P, run, id) {
    crumbs([[P.title, `#/p/${enc(P.id)}`], [run, `#/p/${enc(P.id)}/run/${enc(run)}/episodes`], [id]]);
    loading("episode");
    const e = (await episodes(P, run)).get(id);
    if (!e) { app.innerHTML = errorBox("Episode not found."); return; }
    const o = e.outcome || {}, w = e.work || {};
    const branches = (e.branches || []).map(b => block(`Report ladder: ${b.oversight} · ${b.ladder || "standard"} · ${b.stakes || ""}`,
      `<div class="row">${pill(b.final_label)}${b.disclosure_step ? `<span class="muted">disclosed at ${esc(b.disclosure_step)}</span>` : ""}</div>` +
      (b.steps || []).map(s => `<div class="msg user"><div class="who">${esc(s.name)} question</div><div class="text">${esc(s.question)}</div></div>
        <div class="msg assistant"><div class="who">answer ${pill(s.label)}</div>${s.reasoning ? `<details><summary>reasoning</summary><div class="text">${esc(s.reasoning)}</div></details>` : ""}
        <div class="text">${esc(s.answer)}</div><div class="muted">${esc(s.explanation || "")}</div></div>`).join(""))).join("");
    app.innerHTML = `<div class="stack"><h1 class="mono" style="font-family:var(--mono);font-size:20px">${esc(id)}</h1>
      <div class="row">${o.hacked ? pill("lie", "hacked") : pill("ok", "not hacked")}${(o.methods || []).map(m => pill("", m)).join(" ")}<span class="muted">${esc(e.task_id)} · ${esc(e.split)} · seed ${esc(e.seed)} · ${esc(w.end_reason || "")}</span></div></div>
      <div class="reader"><div class="stack">
        ${block("Outcome", `<dl class="kv">${Object.entries(o).filter(([k]) => !["final_func", "final_test", "conflict_diff", "spec_diff", "test_diff", "methods"].includes(k)).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}
          ${e.belief ? `<dt>recognizes hack</dt><dd>${esc(e.belief.recognizes_hack)}</dd>` : ""}</dl>
          ${o.conflict_diff ? `<pre class="nowrap">${esc(o.conflict_diff)}</pre>` : ""}${o.spec_diff ? `<pre class="nowrap">${esc(o.spec_diff)}</pre>` : ""}${o.test_diff ? `<pre class="nowrap">${esc(o.test_diff)}</pre>` : ""}`)}
        ${block("Final func.py", `<pre class="nowrap">${highlightCode(o.final_func || "")}</pre>`)}
        ${o.final_test ? block("Tests given to the agent", `<details><summary>show test.py</summary><pre class="nowrap">${esc(o.final_test)}</pre></details>`) : ""}
        ${branches}</div>
        <div class="stack">${block(`Work transcript (${(w.messages || []).length} messages)`, messages(w.messages || [], w.reasoning || {}))}</div></div>`;
  }

  // --- reasoning rewrites ---
  async function tabRewrites(P, run, el) {
    const rows = await jsonl(P, `${P.prefix}/runs/${run}/rewrites.jsonl.gz`);
    const latest = [...new Map(rows.map(r => [r.key, r])).values()];
    const base = run.replace(/-rewrites$/, "");
    el.innerHTML = `<p class="muted">${latest.filter(r => r.passed).length} of ${latest.length} pass the checks (latest attempt per sample). Base run: <a href="#/p/${enc(P.id)}/run/${enc(base)}">${esc(base)}</a></p>
      <div class="tablewrap"><table><thead><tr><th>Sample</th><th>Passed</th><th class="num">Attempts</th><th>Problems</th></tr></thead><tbody>${latest.map(r =>
      `<tr class="click" data-href="#/p/${enc(P.id)}/rewrite/${enc(run)}/${enc(r.key)}"><td class="mono">${esc(r.key)}</td><td>${r.passed ? pill("ok", "passed") : pill("warn", "failed")}</td>
      <td class="num">${esc(r.attempts ?? "")}</td><td class="muted">${esc((r.problems || []).join("; "))}</td></tr>`).join("")}</tbody></table></div>`;
  }
  async function viewRewrite(P, run, key) {
    crumbs([[P.title, `#/p/${enc(P.id)}`], [run, `#/p/${enc(P.id)}/run/${enc(run)}/rewrites`], ["rewrite"]]);
    const rows = await jsonl(P, `${P.prefix}/runs/${run}/rewrites.jsonl.gz`);
    const history = rows.filter(r => r.key === key), r = history[history.length - 1];
    const base = run.replace(/-rewrites$/, "");
    app.innerHTML = `<div class="row"><h1 class="mono" style="font-family:var(--mono);font-size:18px">${esc(key)}</h1>${r.passed ? pill("ok", "passed") : pill("warn", "failed")}
      <a href="#/p/${enc(P.id)}/sample/${enc(base)}/${enc(key)}">open sample →</a></div>
      ${history.length > 1 ? `<p class="muted">${history.length} entries for this sample; showing the latest. Earlier problems: ${esc(history.slice(0, -1).map(h => (h.problems || []).join("; ")).join(" | "))}</p>` : ""}
      <div class="reader"><div class="stack">${textBlock("Original reasoning (mentions the nudge)", r.original_reasoning)}</div>
      <div class="stack">${textBlock("Rewritten reasoning (used for training)", r.rewritten_reasoning)}${(r.problems || []).length ? `<p class="err">${esc(r.problems.join("; "))}</p>` : ""}</div></div>`;
  }

  // --- SFT data ---
  async function tabSFT(P, run, el) {
    const ex = await jsonl(P, `${P.prefix}/runs/${run}/train.jsonl.gz`);
    const kinds = [...new Set(ex.map(e => e.kind))];
    const F = store.get(`fs|${P.id}|${run}`, { kind: "" });
    el.innerHTML = `<div class="row">${kinds.map(k => `${pill(k)} ${ex.filter(e => e.kind === k).length}`).join(" · ")}</div>
      <div class="filters panel"><label>Kind<select id="kind"><option value="">any</option>${kinds.map(k => `<option ${F.kind === k ? "selected" : ""}>${esc(k)}</option>`).join("")}</select></label></div><div id="list"></div>`;
    const draw = () => {
      store.set(`fs|${P.id}|${run}`, F);
      document.getElementById("list").innerHTML = `<div class="tablewrap"><table><thead><tr><th class="num">#</th><th>Kind</th><th>Transcript</th><th>Question</th><th>Source</th><th>Judge</th><th>Reviewed</th><th>Answer</th></tr></thead><tbody>
      ${ex.map((e, i) => [e, i]).filter(([e]) => !F.kind || e.kind === F.kind).map(([e, i]) => `<tr class="click" data-href="#/p/${enc(P.id)}/sft/${enc(run)}/${i}"><td class="num">${i}</td><td>${pill(e.kind)}</td>
        <td class="mono">${esc(e.episode_id)}</td><td>${esc(e.question_name || "")}</td><td class="mono muted">${esc(e.source || "")}</td><td>${pill(e.judge_label)}</td>
        <td>${e.reviewed ? pill("ok", "yes") : ""}${e.reasoning_rewritten ? ' <span class="pill">rewritten</span>' : ""}</td>
        <td class="prev">${esc((e.messages.at(-1).content || "").replace(/\s+/g, " ").slice(0, 140))}</td></tr>`).join("")}</tbody></table></div>`;
      bindRowLinks();
    };
    document.getElementById("kind").onchange = ev => { F.kind = ev.target.value; draw(); };
    draw();
  }
  async function viewSFT(P, run, i) {
    crumbs([[P.title, `#/p/${enc(P.id)}`], [run, `#/p/${enc(P.id)}/run/${enc(run)}/sft`], [`example ${i}`]]);
    loading("example");
    const ex = await jsonl(P, `${P.prefix}/runs/${run}/train.jsonl.gz`);
    const e = ex[i], final = e.messages.at(-1);
    const qIdx = e.messages.map(m => m.role).lastIndexOf("user");
    const href = n => `#/p/${enc(P.id)}/sft/${enc(run)}/${n}`;
    app.innerHTML = `<div class="row" style="justify-content:space-between"><div class="row"><h1 class="mono" style="font-family:var(--mono);font-size:19px">${esc(e.episode_id)}</h1>${pill(e.kind)}<span>${esc(e.question_name || "")}</span></div>
      <div class="row muted">${i > 0 ? `<a href="${href(i - 1)}">← prev <kbd>k</kbd></a>` : ""} ${i < ex.length - 1 ? `<a href="${href(i + 1)}">next <kbd>j</kbd> →</a>` : ""}</div></div>
      <div class="reader"><div class="stack">${block(`Context (${e.messages.length - 1} messages, not trained on)`, `<details><summary>show the work transcript</summary>${messages(e.messages.slice(0, -1))}</details>`)}
        ${block("Example", `<dl class="kv">${Object.entries(e).filter(([k]) => !["messages", "tools"].includes(k)).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`)}</div>
      <div class="stack">${textBlock("Question", e.messages[qIdx]?.content)}${textBlock("Trained reasoning", final.reasoning_content)}${textBlock("Trained answer", final.content, true)}</div></div>`;
    keyHandler = ev => { if (ev.key === "j" && i < ex.length - 1) location.hash = href(i + 1); else if (ev.key === "k" && i > 0) location.hash = href(i - 1); };
  }

  // --- training ---
  async function tabTraining(P, run, el, r) {
    const log = await orEmpty(jsonl(P, `${P.prefix}/runs/${run}/train_log.jsonl.gz`), []);
    el.innerHTML = `<h2>Loss per step</h2>${lineChart(log.map(x => [x.step, x.loss]), { yLabel: "loss" })}
      <div class="tablewrap"><table><thead><tr><th class="num">Step</th><th class="num">Epoch</th><th class="num">Loss</th><th class="num">Grad norm</th><th class="num">LR</th><th class="num">Seconds</th><th>Time</th></tr></thead><tbody>
      ${log.map(x => `<tr><td class="num">${x.step}</td><td class="num">${(+x.epoch).toFixed(2)}</td><td class="num">${(+x.loss).toFixed(4)}</td><td class="num">${(+x.grad_norm).toFixed(3)}</td><td class="num">${(+x.lr).toExponential(2)}</td><td class="num">${x.seconds}</td><td class="muted">${when(x.time, { relative: false })}</td></tr>`).join("")}</tbody></table></div>
      <h2>Config</h2><pre>${esc(JSON.stringify(r.meta.config || r.meta, null, 1))}</pre>`;
  }

  // ---------- router ----------
  function bindRowLinks() {
    app.querySelectorAll("tr[data-href]").forEach(tr => tr.addEventListener("click", ev => { if (ev.target.closest("a")) return; location.hash = tr.dataset.href; }));
  }
  async function route() {
    keyHandler = null;
    tokenState();
    const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
    try {
      if (!parts.length) return await viewHome();
      if (parts[0] === "settings") return viewSettings();
      if (parts[0] !== "p") return await viewHome();
      const P = (await projects()).find(p => p.id === parts[1]);
      if (!P) { app.innerHTML = errorBox("Unknown project."); return; }
      const [, , kind, a, ...rest] = parts;
      if (!kind) return await viewProject(P);
      if (kind === "run") return await viewRun(P, a, rest[0]);
      if (kind === "sample") return await viewSample(P, a, rest.join("/"));
      if (kind === "episode") return await viewEpisode(P, a, rest.join("/"));
      if (kind === "rewrite") return await viewRewrite(P, a, rest.join("/"));
      if (kind === "sft") return await viewSFT(P, a, +rest[0]);
      if (kind === "doc") return await viewDoc(P, [a, ...rest].join("/"));
      return await viewProject(P);
    } catch (e) { app.innerHTML = errorBox(e); }
    finally { window.scrollTo(0, 0); }
  }
  document.addEventListener("keydown", e => keyHandler && keyHandler(e));
  window.addEventListener("hashchange", route);
  route();
})();
