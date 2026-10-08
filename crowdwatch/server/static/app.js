/* CrowdWatch dashboard. Plain JavaScript, no build step. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const LEVEL_WORD = ["Normal", "Busy", "Nearly full", "Over capacity"];
  const BOARD_WORD = ["All clear", "Busy", "Nearly full", "Over capacity"];
  const SERIES = ["#16203a", "#0b7a75", "#6c4ab6", "#9a5b13", "#b5336b", "#566078"];
  const METER_MAX = 1.25;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const state = {
    config: null, snap: null, history: {}, log: [], runId: null,
    editing: false, draft: null, drawing: null, drag: null,
    sound: false, audio: null, seenCritical: new Set(), boardKey: "", viewSynced: false,
  };

  // ------------------------------------------------------------ helpers

  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return res.json();
  }
  const post = (path, body, method = "POST") => api(path, {
    method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const esc = (text) => String(text).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clock = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

  function duration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    if (s < 60) return `${s} s`;
    const m = Math.floor(s / 60), rest = s % 60;
    return rest >= 5 ? `${m} min ${rest} s` : `${m} min`;
  }
  function eta(seconds) {
    if (seconds < 15) return "a few seconds";
    if (seconds < 50) return `about ${Math.max(10, Math.round(seconds / 10) * 10)} seconds`;
    const m = seconds / 60;
    if (m < 1.25) return "about 1 minute";
    if (m < 1.75) return "about 1.5 minutes";
    return `about ${Math.round(m)} minutes`;
  }
  function trendText(perMin) {
    if (Math.abs(perMin) < 1) return "steady";
    return `${perMin > 0 ? "rising" : "falling"} by ${Math.round(Math.abs(perMin))} a minute`;
  }
  function when(alert) {
    const live = state.snap && state.snap.source && state.snap.source.live;
    if (live) return new Date(alert.started_wall * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return clock(alert.started_t);
  }
  const glyph = (id) => `<svg class="glyph" aria-hidden="true"><use href="#g-${id}"/></svg>`;

  // ------------------------------------------------------------- config

  async function loadConfig() {
    state.config = await api("/api/config");
    const cfg = state.config;
    $("site").textContent = cfg.name;
    document.title = `CrowdWatch: ${cfg.name}`;
    $("t-blur-wrap").hidden = cfg.source === "simulator";
    $("scenario-wrap").hidden = cfg.scenarios.length === 0;
    if (cfg.scenarios.length) {
      const current = state.snap && state.snap.source ? state.snap.source.scenario : null;
      $("scenario").innerHTML = cfg.scenarios.map((s) =>
        `<option value="${esc(s.name)}"${s.name === current ? " selected" : ""}>${esc(s.title)}</option>`).join("");
      describeScenario();
    }
    buildZones();
  }

  function describeScenario() {
    const picked = state.config.scenarios.find((s) => s.name === $("scenario").value);
    $("scenario-note").textContent = picked ? picked.description : "";
  }

  function buildZones() {
    const box = $("zones");
    const th = state.config.thresholds;
    if (!state.config.zones.length) {
      box.innerHTML = '<p class="empty">No zones yet. Choose "Edit zones" under the live view and draw the areas you want counted.</p>';
      return;
    }
    box.innerHTML = state.config.zones.map((z) => `
      <article class="zone" id="zone-${esc(z.id)}" data-level="0">
        <div class="zone-head"><span class="zone-glyph">${glyph(0)}</span><span>${esc(z.name)}</span><span class="zone-state">Normal</span></div>
        <div class="zone-count"><span class="n">0</span> <span class="zone-of">of ${z.capacity}</span></div>
        <div class="meter" role="img" aria-label="Occupancy meter">
          <div class="meter-fill" style="width:0"></div>
          <div class="meter-tick" style="left:${(th.warning / METER_MAX) * 100}%" title="Nearly full from here"></div>
          <div class="meter-tick" style="left:${(th.critical / METER_MAX) * 100}%" title="Capacity"></div>
        </div>
        <canvas class="spark" aria-hidden="true"></canvas>
        <div class="zone-predict" hidden></div>
        <div class="zone-facts"></div>
      </article>`).join("");
  }

  // --------------------------------------------------------------- live

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => setLink(true, "Connected");
    ws.onmessage = (event) => onSnapshot(JSON.parse(event.data));
    ws.onclose = () => { setLink(false, "Connection lost. Retrying"); setTimeout(connect, 1500); };
    ws.onerror = () => ws.close();
  }
  function setLink(ok, text) {
    $("link-state").dataset.ok = String(ok);
    $("link-state").textContent = text;
  }

  async function onSnapshot(snap) {
    if (!snap || !snap.zones) return;
    state.snap = snap;
    const ids = snap.zones.map((z) => z.id).join("|");
    const known = state.config ? state.config.zones.map((z) => z.id).join("|") : null;
    if (snap.run_id !== state.runId || (ids !== known && !state.loading)) {
      state.runId = snap.run_id;
      state.loading = true;
      state.history = {};
      try { await loadConfig(); await pollHistory(); await pollAlerts(); } finally { state.loading = false; }
    }
    if (!state.config) return;
    if (!state.viewSynced) {
      $("t-markers").checked = snap.view.markers;
      $("t-heatmap").checked = snap.view.heatmap;
      $("t-blur").checked = snap.view.blur;
      state.viewSynced = true;
    }
    renderBoard(snap);
    renderZones(snap);
    renderOverlay(snap);
    renderAlerts(snap);
    renderDoors(snap);
    const note = $("stage-note");
    if (snap.error) { note.hidden = false; note.textContent = `Video problem: ${snap.error}`; }
    else if (snap.ended) { note.hidden = false; note.textContent = "The recording has finished. The last counts are shown."; }
    else note.hidden = true;
  }

  function renderBoard(snap) {
    const board = $("board");
    const level = snap.status.level;
    const predicted = snap.status.predicted && level < 2;
    const key = `${level}:${predicted}`;
    board.dataset.level = level;
    board.dataset.predicted = String(predicted);
    $("status-word").textContent = predicted ? "Filling up" : BOARD_WORD[level];
    $("board-glyph").innerHTML = `<use href="#g-${predicted ? "p" : level}"/>`;
    $("headline").textContent = snap.status.headline;
    $("people").textContent = snap.people;
    $("clock").textContent = snap.clock;
    const src = snap.source || {};
    $("clock-label").textContent = src.live ? "Time" : (src.speed > 1 ? `Video time, ${src.speed}× speed` : "Video time");
    if (state.boardKey && key !== state.boardKey) {
      board.classList.remove("changed");
      void board.offsetWidth;
      board.classList.add("changed");
    }
    state.boardKey = key;

    for (const alert of snap.alerts) {
      if (alert.level === "critical" && !state.seenCritical.has(alert.id)) {
        state.seenCritical.add(alert.id);
        beep();
      }
    }
  }

  function renderZones(snap) {
    for (const z of snap.zones) {
      const el = $(`zone-${z.id}`);
      if (!el) continue;
      if (el.dataset.level !== String(z.level)) {
        el.dataset.level = z.level;
        el.querySelector(".zone-glyph").innerHTML = glyph(z.level);
      }
      el.querySelector(".zone-state").textContent = `${LEVEL_WORD[z.level]}, ${Math.round(z.ratio * 100)}%`;
      el.querySelector(".n").textContent = z.count;
      el.querySelector(".zone-of").textContent = `of ${z.capacity}`;
      el.querySelector(".meter-fill").style.width = `${(Math.min(z.ratio, METER_MAX) / METER_MAX) * 100}%`;

      const predict = el.querySelector(".zone-predict");
      if (z.predicted && z.level < 3) {
        const chance = z.probability !== null ? ` (${Math.round(z.probability * 100)}% likely)` : "";
        const text = z.eta_s !== null ? `Full in ${eta(z.eta_s)} at this rate` : "Likely to fill within 2 minutes";
        predict.hidden = false;
        predict.innerHTML = `${glyph("p")}<span>${text}${chance}</span>`;
      } else if (z.congested) {
        predict.hidden = false;
        predict.innerHTML = `${glyph(2)}<span>The crowd here has almost stopped moving</span>`;
      } else predict.hidden = true;

      const facts = [`<span><b>${trendText(z.trend_per_min)}</b></span>`];
      if (z.approaching >= 1) facts.push(`<span><b>${z.approaching}</b> heading this way</span>`);
      if (z.method === "density") facts.push("<span>Counted by <b>density map</b></span>");
      if (z.density !== null) facts.push(`<span>Density <b>${z.density.toFixed(2)}</b> per m²</span>`);
      if (z.speed_mps !== null) facts.push(`<span>Walking speed <b>${z.speed_mps.toFixed(1)}</b> m/s${snap.calibrated ? "" : " (approx.)"}</span>`);
      if (z.dwell_s !== null) facts.push(`<span>Average stay <b>${duration(z.dwell_s)}</b></span>`);
      facts.push(`<span>Risk <b>${z.risk}</b> of 100</span>`);
      el.querySelector(".zone-facts").innerHTML = facts.join("");
    }
  }

  function renderDoors(snap) {
    $("doors-panel").hidden = snap.lines.length === 0;
    $("doors").innerHTML = snap.lines.map((l) =>
      `<tr><td>${esc(l.name)}</td><td>${l.in}</td><td>${l.out}</td><td>${l.net > 0 ? "+" : ""}${l.net}</td></tr>`).join("");
  }

  function renderAlerts(snap) {
    const active = snap.alerts;
    $("alerts-active").innerHTML = active.length ? active.map((a) => `
      <div class="alert" data-level="${a.level}">
        ${glyph(a.level === "critical" ? 3 : a.level === "warning" ? 2 : "p")}
        <div class="alert-text">${esc(a.message)}</div>
        <div class="alert-meta"><span>Since ${when(a)}</span>
          ${a.acknowledged ? "<span>Acknowledged</span>"
            : `<button type="button" data-ack="${a.id}">Acknowledge</button>`}</div>
      </div>`).join("") : '<p class="empty">Nothing needs attention right now.</p>';

    const activeIds = new Set(active.map((a) => a.id));
    const past = state.log.filter((a) => !a.active && !activeIds.has(a.id) && a.run_id === snap.run_id).slice(0, 8);
    $("h-past").hidden = past.length === 0;
    $("alerts-past").innerHTML = past.map((a) => {
      const what = a.kind === "predicted" ? "Early warning" : (a.level === "critical" ? "Over capacity" : "Nearly full");
      const peak = a.kind === "predicted" ? "" : `, peak ${Math.round(a.peak_count)} people (${Math.round(a.peak_ratio * 100)}%)`;
      return `<div class="alert past" data-level="${a.level}">
        ${glyph(a.level === "critical" ? 3 : a.level === "warning" ? 2 : "p")}
        <div class="alert-text">${esc(a.zone_name)}: ${what.toLowerCase()} for ${duration(a.ended_t - a.started_t)}${peak}</div>
        <div class="alert-meta"><span>Started ${when(a)}</span><span>Ended: ${esc(a.outcome || "resolved")}</span></div>
      </div>`;
    }).join("");
  }

  // ------------------------------------------------------------ overlay

  function renderOverlay(snap) {
    const W = snap.frame.width, H = snap.frame.height;
    const stage = $("stage"), svg = $("overlay");
    stage.style.aspectRatio = `${W} / ${H}`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const byId = Object.fromEntries(snap.zones.map((z) => [z.id, z]));
    const zones = state.editing ? state.draft : state.config.zones;
    let shapes = "", labels = "";

    zones.forEach((z, zi) => {
      const live = byId[z.id];
      const level = state.editing ? null : (live ? live.level : 0);
      const pts = z.polygon.map(([x, y]) => `${x * W},${y * H}`).join(" ");
      shapes += `<polygon class="zone-shape${state.editing ? " draft" : ""}" data-level="${level}" points="${pts}"/>`;
      if (state.editing) {
        z.polygon.forEach(([x, y], vi) => {
          shapes += `<circle class="handle" data-zone="${zi}" data-vertex="${vi}" cx="${x * W}" cy="${y * H}" r="${W * 0.008}"/>`;
        });
      }
      const minX = Math.min(...z.polygon.map((p) => p[0])), minY = Math.min(...z.polygon.map((p) => p[1]));
      const text = state.editing || !live ? esc(z.name) : `${esc(z.name)}&ensp;${live.count} / ${live.capacity}`;
      labels += `<div class="zone-label${minY < 0.07 ? " below" : ""}${state.editing ? " draft" : ""}" data-level="${level}"
        style="left:${minX * 100}%;top:${minY * 100}%">${text}</div>`;
    });

    if (state.drawing && state.drawing.length) {
      const pts = state.drawing.map(([x, y]) => `${x * W},${y * H}`).join(" ");
      shapes += `<polyline class="zone-shape draft" fill="none" points="${pts}"/>`;
      state.drawing.forEach(([x, y]) => {
        shapes += `<circle class="handle" cx="${x * W}" cy="${y * H}" r="${W * 0.008}"/>`;
      });
    }
    if (!state.editing) {
      for (const line of state.config.lines) {
        const [ax, ay] = line.a, [bx, by] = line.b;
        const coords = `x1="${ax * W}" y1="${ay * H}" x2="${bx * W}" y2="${by * H}"`;
        shapes += `<line class="door-line" ${coords}/><line class="door-line inner" ${coords}/>`;
        labels += `<div class="door-label" style="left:${((ax + bx) / 2) * 100}%;top:${((ay + by) / 2) * 100}%">${esc(line.name)}</div>`;
      }
    }
    svg.innerHTML = shapes;
    $("labels").innerHTML = labels;
  }

  // ------------------------------------------------------------- charts

  function prepare(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
      canvas.width = Math.round(w * ratio);
      canvas.height = Math.round(h * ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function drawSparks() {
    if (!state.snap || !state.config) return;
    const now = state.snap.t, span = 300, lead = state.config.thresholds.forecast_lead_s;
    for (const z of state.snap.zones) {
      const el = $(`zone-${z.id}`);
      const rows = (state.history[z.id] || []).filter((r) => r[0] >= now - span);
      if (!el || rows.length < 2) continue;
      const { ctx, w, h } = prepare(el.querySelector(".spark"));
      const critical = z.level === 3;
      const ink = critical ? "#ffffff" : css("--ink");
      const pastW = w * 0.78;
      const top = Math.max(1.3, ...rows.map((r) => r[2])) * 1.05;
      const x = (t) => ((t - (now - span)) / span) * pastW;
      const y = (v) => h - 3 - (v / top) * (h - 8);
      ctx.strokeStyle = critical ? "rgba(255,255,255,.7)" : css("--muted");
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(0, y(1)); ctx.lineTo(w, y(1)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.lineJoin = "round";
      ctx.beginPath();
      rows.forEach((r, i) => (i ? ctx.lineTo(x(r[0]), y(r[2])) : ctx.moveTo(x(r[0]), y(r[2]))));
      ctx.stroke();
      if (z.forecast_count !== null && Math.abs(z.trend_per_min) >= 1) {
        const last = rows[rows.length - 1];
        const fy = y(Math.min(z.forecast_count / z.capacity, top));
        ctx.strokeStyle = critical ? "#ffffff" : css("--info");
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x(last[0]), y(last[2])); ctx.lineTo(w - 4, fy); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath(); ctx.arc(w - 4, fy, 3, 0, Math.PI * 2); ctx.fill();
      }
      el.querySelector(".spark").title = `Last 5 minutes, with the forecast for ${Math.round(lead / 60)} minutes ahead (dashed)`;
    }
  }

  function drawChart() {
    if (!state.snap || !state.config) return;
    const { ctx, w, h } = prepare($("chart"));
    const now = state.snap.t, span = 600;
    const th = state.config.thresholds;
    const pad = { left: 44, right: 150, top: 14, bottom: 26 };
    const series = state.config.zones.map((z, i) => ({
      zone: z, colour: SERIES[i % SERIES.length],
      rows: (state.history[z.id] || []).filter((r) => r[0] >= now - span),
    }));
    const peak = Math.max(th.critical * 1.2, ...series.flatMap((s) => s.rows.map((r) => r[2])));
    const top = Math.ceil(peak * 4) / 4;
    const x = (t) => pad.left + ((t - (now - span)) / span) * (w - pad.left - pad.right);
    const y = (v) => h - pad.bottom - (v / top) * (h - pad.top - pad.bottom);
    const font = `${css("font-family") || "sans-serif"}`;
    ctx.font = `500 12px ${font}`; ctx.textBaseline = "middle";

    ctx.strokeStyle = css("--rule"); ctx.fillStyle = css("--muted"); ctx.lineWidth = 1;
    for (let v = 0; v <= top + 1e-9; v += top > 2 ? 0.5 : 0.25) {
      ctx.beginPath(); ctx.moveTo(pad.left, y(v)); ctx.lineTo(w - pad.right, y(v)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(`${Math.round(v * 100)}%`, pad.left - 8, y(v));
    }
    ctx.textAlign = "center";
    [[0, "10 min ago"], [300, "5 min ago"], [600, "now"]].forEach(([dt, label]) => {
      ctx.fillText(label, Math.min(Math.max(x(now - span + dt), pad.left + 28), w - pad.right - 10), h - 9);
    });
    [[th.warning, css("--warning"), "Nearly full"], [th.critical, css("--critical"), "Capacity"]].forEach(([v, colour, label]) => {
      ctx.strokeStyle = colour; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pad.left, y(v)); ctx.lineTo(w - pad.right, y(v)); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = colour; ctx.textAlign = "left";
      ctx.font = `600 11px ${font}`;
      ctx.fillText(label, pad.left + 6, y(v) - 8);
      ctx.font = `500 12px ${font}`;
    });

    const ends = [];
    for (const s of series) {
      if (s.rows.length < 2) continue;
      ctx.strokeStyle = s.colour; ctx.lineWidth = 2; ctx.lineJoin = "round";
      ctx.beginPath();
      s.rows.forEach((r, i) => (i ? ctx.lineTo(x(r[0]), y(r[2])) : ctx.moveTo(x(r[0]), y(r[2]))));
      ctx.stroke();
      const last = s.rows[s.rows.length - 1];
      ends.push({ y: y(last[2]), label: `${s.zone.name} ${Math.round(last[2] * 100)}%`, colour: s.colour });
    }
    // name each line at its right-hand end, nudged apart so labels never overlap
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
    const overflow = ends.length ? ends[ends.length - 1].y - (h - pad.bottom) : 0;
    if (overflow > 0) ends.forEach((e) => { e.y -= overflow; });
    ctx.textAlign = "left"; ctx.font = `600 12px ${font}`;
    ends.forEach((e) => { ctx.fillStyle = e.colour; ctx.fillText(e.label, w - pad.right + 8, e.y); });
  }

  async function pollHistory() {
    try {
      state.history = (await api("/api/history?seconds=600")).zones;
      drawChart(); drawSparks();
    } catch (_) { /* the next poll will try again */ }
  }
  async function pollAlerts() {
    try { state.log = (await api("/api/alerts?limit=40")).alerts; } catch (_) { /* retry next time */ }
  }

  // -------------------------------------------------------------- sound

  function beep() {
    if (!state.sound || !state.audio) return;
    const ctx = state.audio;
    [0, 0.28].forEach((offset, i) => {
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.frequency.value = i ? 660 : 880;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime + offset);
      gain.gain.exponentialRampToValueAtTime(0.25, ctx.currentTime + offset + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + offset + 0.24);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + offset); osc.stop(ctx.currentTime + offset + 0.26);
    });
  }

  // ------------------------------------------------------------- editor

  function slug(name, taken) {
    const base = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 32) || "zone";
    let id = base, n = 2;
    while (taken.has(id)) id = `${base}_${n++}`;
    return id;
  }

  function startEditing() {
    state.editing = true;
    state.draft = JSON.parse(JSON.stringify(state.config.zones));
    state.drawing = null;
    $("stage").classList.add("editing");
    $("zones-panel").hidden = true;
    $("editor-panel").hidden = false;
    $("editor-error").hidden = true;
    buildEditor();
    if (state.snap) renderOverlay(state.snap);
  }
  function stopEditing() {
    state.editing = false; state.draft = null; state.drawing = null; state.drag = null;
    $("stage").classList.remove("editing", "drawing");
    $("zones-panel").hidden = false;
    $("editor-panel").hidden = true;
    $("finish-shape").hidden = true;
    if (state.snap) renderOverlay(state.snap);
  }
  function buildEditor() {
    $("editor-list").innerHTML = state.draft.length ? state.draft.map((z, i) => `
      <div class="edit-row" data-index="${i}">
        <label>Name<input type="text" data-field="name" value="${esc(z.name)}" maxlength="60"></label>
        <label>Capacity<input type="number" data-field="capacity" value="${z.capacity || 30}" min="1" step="1"></label>
        <button type="button" data-remove="${i}">Remove</button>
        <label class="walkway"><input type="checkbox" data-field="walkway"${z.walkway ? " checked" : ""}>
          People should keep moving here (flag a stalled crowd)</label>
      </div>`).join("") : '<p class="empty">No zones. Draw the first one.</p>';
  }
  function pointFromEvent(event) {
    const box = $("stage").getBoundingClientRect();
    const clamp = (v) => Math.min(1, Math.max(0, v));
    return [+clamp((event.clientX - box.left) / box.width).toFixed(4),
            +clamp((event.clientY - box.top) / box.height).toFixed(4)];
  }
  function finishShape() {
    if (!state.drawing || state.drawing.length < 3) return;
    const taken = new Set(state.draft.map((z) => z.id));
    const name = `Zone ${state.draft.length + 1}`;
    state.draft.push({ id: slug(name, taken), name, polygon: state.drawing, capacity: 30, walkway: false });
    state.drawing = null;
    $("stage").classList.remove("drawing");
    $("finish-shape").hidden = true;
    $("editor-hint").textContent = "Give the new zone a name and the number of people it can safely hold, then save.";
    buildEditor();
    if (state.snap) renderOverlay(state.snap);
  }
  async function saveZones() {
    const error = $("editor-error");
    error.hidden = true;
    const zones = state.draft.map((z) => ({ ...z, capacity: Math.max(1, Math.round(Number(z.capacity) || 0)) }));
    if (zones.some((z) => !z.name.trim())) {
      error.textContent = "Every zone needs a name."; error.hidden = false; return;
    }
    try {
      await post("/api/zones", { zones }, "PUT");
      stopEditing();
      setTimeout(async () => { await loadConfig(); await pollHistory(); }, 500);
    } catch (exc) {
      error.textContent = `Could not save the zones: ${exc.message}`; error.hidden = false;
    }
  }

  // ------------------------------------------------------------- events

  function wire() {
    $("video").src = "/stream.mjpg";
    $("video").addEventListener("error", () => setTimeout(() => { $("video").src = `/stream.mjpg?${Date.now()}`; }, 2000));

    for (const key of ["markers", "heatmap", "blur"]) {
      $(`t-${key}`).addEventListener("change", (e) => post("/api/view", { [key]: e.target.checked }).catch(() => {}));
    }
    $("scenario").addEventListener("change", describeScenario);
    $("scenario-go").addEventListener("click", () => post("/api/scenario", { name: $("scenario").value }).catch(() => {}));

    $("sound-toggle").addEventListener("click", () => {
      state.sound = !state.sound;
      if (state.sound && !state.audio) state.audio = new (window.AudioContext || window.webkitAudioContext)();
      $("sound-toggle").setAttribute("aria-pressed", String(state.sound));
      $("sound-toggle").textContent = state.sound ? "Sound alarm is on" : "Sound alarm is off";
      if (state.sound) beep();
    });

    $("alerts-active").addEventListener("click", async (e) => {
      const id = e.target.dataset.ack;
      if (!id) return;
      e.target.disabled = true;
      try { await post(`/api/alerts/${id}/ack`, {}); } catch (_) { e.target.disabled = false; }
    });

    $("edit-zones").addEventListener("click", () => (state.editing ? stopEditing() : startEditing()));
    $("cancel-edit").addEventListener("click", stopEditing);
    $("save-zones").addEventListener("click", saveZones);
    $("finish-shape").addEventListener("click", finishShape);
    $("draw-zone").addEventListener("click", () => {
      state.drawing = [];
      $("stage").classList.add("drawing");
      $("editor-hint").textContent = "Click each corner of the new zone on the live view. Choose \"Finish shape\" after the last corner.";
    });
    $("editor-list").addEventListener("input", (e) => {
      const row = e.target.closest(".edit-row");
      const field = e.target.dataset.field;
      if (!row || !field) return;
      const zone = state.draft[Number(row.dataset.index)];
      zone[field] = e.target.type === "checkbox" ? e.target.checked
        : (field === "capacity" ? Number(e.target.value) : e.target.value);
      if (state.snap) renderOverlay(state.snap);
    });
    $("editor-list").addEventListener("click", (e) => {
      if (e.target.dataset.remove === undefined) return;
      state.draft.splice(Number(e.target.dataset.remove), 1);
      buildEditor();
      if (state.snap) renderOverlay(state.snap);
    });

    const svg = $("overlay");
    svg.addEventListener("pointerdown", (e) => {
      if (!state.editing) return;
      if (state.drawing) {
        state.drawing.push(pointFromEvent(e));
        $("finish-shape").hidden = state.drawing.length < 3;
      } else if (e.target.classList.contains("handle") && e.target.dataset.zone !== undefined) {
        state.drag = { zone: Number(e.target.dataset.zone), vertex: Number(e.target.dataset.vertex) };
        svg.setPointerCapture(e.pointerId);
      }
      if (state.snap) renderOverlay(state.snap);
    });
    svg.addEventListener("pointermove", (e) => {
      if (!state.drag) return;
      state.draft[state.drag.zone].polygon[state.drag.vertex] = pointFromEvent(e);
      if (state.snap) renderOverlay(state.snap);
    });
    svg.addEventListener("pointerup", () => { state.drag = null; });
    window.addEventListener("resize", () => { drawChart(); drawSparks(); });
  }

  wire();
  connect();
  setInterval(pollHistory, 2000);
  setInterval(pollAlerts, 3000);
})();
