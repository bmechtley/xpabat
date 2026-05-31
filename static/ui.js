// ─── Tooltip ─────────────────────────────────────────────────
function showTooltip(c, cx, cy) {
  const tt = document.getElementById('tooltip');
  tt.innerHTML = `
    <div class="sp-name" style="color:${c.color}">${c.species}</div>
    <div class="param" style="color:#555">call #${c.id}</div>
    <div class="param">t: <span>${fmt(c.t0)} – ${fmt(c.t1)}</span></div>
    <div class="param">dur: <span>${c.dur.toFixed(1)} ms</span></div>
    <div class="param">Fpeak: <span>${c.Fpeak.toFixed(1)} kHz</span></div>
    <div class="param">Fmin: <span>${c.Fmin.toFixed(1)} kHz</span></div>
    <div class="param">Fmax: <span>${c.Fmax.toFixed(1)} kHz</span></div>
    <div class="param">sweep: <span>${c.sweep.toFixed(2)} kHz/ms</span></div>
    <div class="param">sp. confidence: <span>${(c.conf * 100).toFixed(0)}%</span></div>
    ${c.det_prob > 0 ? `<div class="param">det. score: <span>${(c.det_prob).toFixed(2)}</span></div>` : ''}`;
  const wrap = canvasWrap.getBoundingClientRect();
  let left = cx - wrap.left + 14;
  let top  = cy - wrap.top  + 14;
  if (left + 230 > wrap.width)  left = cx - wrap.left - 230;
  if (top  + 180 > wrap.height) top  = cy - wrap.top  - 180;
  tt.style.left    = left + 'px';
  tt.style.top     = top  + 'px';
  tt.style.display = 'block';
}
function hideTooltip() {
  document.getElementById('tooltip').style.display = 'none';
}

// ─── Call zoom / navigation ───────────────────────────────────
function zoomToCall(c) {
  if (!c) return;
  const pad = Math.max(0.4, (c.t1 - c.t0) * 1.5);
  S.viewStart = Math.max(0, c.t0 - pad);
  S.viewDur   = Math.min(S.duration - S.viewStart, c.t1 + pad - S.viewStart);
  scheduleRender();
}

// Returns true when the fixed ruler was placed by a double-click on `c` —
// i.e. the loop/filter bounds exactly match the call's bounding box.
// Used by navigateCall to decide whether to carry the marquee to the next call.
function _marqueeMatchesCall(c) {
  return !!(c && S.rulerFixed &&
            S.rulerLoopT0 === c.t0  && S.rulerLoopT1 === c.t1 &&
            S.rulerLoopF0 === c.Fmin && S.rulerLoopF1 === c.Fmax);
}

// Places the ruler marquee around call `c`, using the current viewport
// (call after any zoom so tToX/fToY reflect the new view).
function _applyCallMarquee(c) {
  S.rulerX0 = tToX(c.t0);    S.rulerX1 = tToX(c.t1);
  S.rulerY0 = fToY(c.Fmax);  S.rulerY1 = fToY(c.Fmin);
  S.rulerFixed  = true;
  S.isRuling    = false;
  S.rulerLoopT0 = c.t0;
  S.rulerLoopT1 = c.t1;
  S.rulerLoopF0 = c.Fmin;
  S.rulerLoopF1 = c.Fmax;
  audioUpdateBPF();
}

function navigateCall(delta) {
  const visible = S.calls.filter(c => !S.hiddenSpecies.has(c.species) && c.conf >= S.minConf);
  if (!visible.length) return;

  // Capture marquee state before we switch selectedCall
  const keepMarquee = _marqueeMatchesCall(S.selectedCall);

  let idx = S.selectedCall ? visible.findIndex(c => c === S.selectedCall) : -1;
  if (idx < 0) {
    // No selected call — find first call after (or before) view centre
    const mid = S.viewStart + S.viewDur / 2;
    idx = delta > 0
      ? Math.max(0, visible.findIndex(c => c.t0 >= mid))
      : visible.findLastIndex(c => c.t0 < mid);
    if (idx < 0) idx = delta > 0 ? 0 : visible.length - 1;
  } else {
    idx = Math.max(0, Math.min(visible.length - 1, idx + delta));
  }
  const call = visible[idx];
  S.selectedCall = call;
  document.getElementById('input-call-id').value = call.id;
  // Advance playhead to the selected call
  S.playheadTime = call.t0;
  if (S.isPlaying && typeof audioSeek === 'function') audioSeek(call.t0);
  renderDetail(call);
  zoomToCall(call);   // updates S.viewStart/S.viewDur synchronously

  if (keepMarquee) {
    // Carry the marquee to the new call — pixel coords computed post-zoom
    _applyCallMarquee(call);
  } else if (S.rulerFixed) {
    // Clear any pre-existing manual marquee when navigating without one
    S.rulerFixed  = false;
    S.rulerLoopT0 = null;  S.rulerLoopT1 = null;
    S.rulerLoopF0 = null;  S.rulerLoopF1 = null;
    audioUpdateBPF();
  }
}

function jumpToCallId(id, zoom = true) {
  const call = S.calls.find(c => c.id === id);
  if (!call) return;
  S.selectedCall = call;
  const inp = document.getElementById('input-call-id');
  if (inp) inp.value = call.id;
  // Advance playhead to the selected call
  S.playheadTime = call.t0;
  if (S.isPlaying && typeof audioSeek === 'function') audioSeek(call.t0);
  renderDetail(call);
  if (zoom) zoomToCall(call);
}

// ─── Accordion state machine ──────────────────────────────────
// _openAcc: null | 'call' | species-name-string
let _openAcc = null;

function _setAccordionState(who) {
  _openAcc = who;
  const callWrap = document.getElementById('acc-call-wrap');

  // Reset call accordion
  callWrap.classList.remove('acc-open');
  document.getElementById('acc-call-chev').textContent = '▸';

  // Reset all species inline items
  document.querySelectorAll('.sp-acc-item').forEach(item => {
    item.classList.remove('acc-open');
    const body = item.querySelector('.sp-acc-body');
    if (body) body.innerHTML = '';
    const hdr = item.querySelector('.sp-acc-header');
    if (hdr) {
      hdr.classList.remove('acc-active');
      const a = hdr.querySelector('.sp-acc-arrow');
      if (a) a.textContent = '▴';
    }
  });

  if (who === 'call') {
    callWrap.classList.add('acc-open');
    document.getElementById('acc-call-chev').textContent = '▾';
  } else if (who) {
    // Species name — expand that item inline and scroll it into view
    const item = document.querySelector(`.sp-acc-item[data-sp="${CSS.escape(who)}"]`);
    if (item) {
      item.classList.add('acc-open');
      const body = item.querySelector('.sp-acc-body');
      if (body) body.innerHTML = _buildSpContent(who);
      const hdr = item.querySelector('.sp-acc-header');
      if (hdr) {
        hdr.classList.add('acc-active');
        const a = hdr.querySelector('.sp-acc-arrow');
        if (a) a.textContent = '▾';
      }
    }
  }
}

function toggleCallAcc() {
  _setAccordionState(_openAcc === 'call' ? null : 'call');
}

function renderDetail(c) {
  _scheduleURLSync();   // update ?call= whenever selection changes
  const body    = document.getElementById('acc-call-body');
  const meta    = document.getElementById('acc-call-meta');
  const callInp = document.getElementById('input-call-id');
  if (callInp) callInp.value = c ? c.id : '';
  if (!c) {
    body.innerHTML = '<span class="acc-empty">Click a call to inspect it</span>';
    meta.textContent = '';
    return;
  }
  meta.textContent = `#${c.id} · ${c.short}`;
  // Show classifier disagreement: always show the *other* classifier's result
  const v1sp = c.species_v1 ?? c.species;
  const v2sp = c.species_v2 ?? c.species;
  const disagrees = v1sp !== v2sp;
  const usingV1 = S.classifier === 'v1';
  const altLabel = usingV1 ? 'v2 says' : 'v1 says';
  const altShort = usingV1 ? (c.short_v2  ?? c.short)   : c.short_v1;
  const altColor = usingV1 ? (c.color_v2  ?? c.color)   : c.color_v1;
  const altConf  = usingV1 ? (c.conf_v2   ?? c.conf)    : c.conf_v1;
  const cmpRow = disagrees
    ? `<tr style="color:#aaa;font-size:10px"><td>${altLabel}</td>
         <td><span style="background:${altColor};color:#fff;padding:1px 4px;border-radius:2px;font-size:9px">${altShort}</span> ${(altConf*100).toFixed(0)}%</td></tr>`
    : '';
  body.innerHTML = `
    <div class="acc-sp-badge" style="background:${c.color}">${c.short} — ${c.species}</div>
    <table class="acc-table">
      <tr><td>Call ID</td><td>#${c.id}</td></tr>
      <tr><td>Confidence</td><td>${(c.conf*100).toFixed(0)}%</td></tr>
      ${cmpRow}
      <tr><td>Time</td><td>${fmt(c.t0)} – ${fmt(c.t1)}</td></tr>
      <tr><td>Duration</td><td>${c.dur.toFixed(1)} ms</td></tr>
      <tr><td>Fmax</td><td>${c.Fmax.toFixed(1)} kHz</td></tr>
      <tr><td>Fpeak</td><td>${c.Fpeak.toFixed(1)} kHz</td></tr>
      <tr><td>Fmin</td><td>${c.Fmin.toFixed(1)} kHz</td></tr>
      <tr><td>Bandwidth</td><td>${(c.bw ?? c.Fmax - c.Fmin).toFixed(1)} kHz</td></tr>
      <tr><td>CF fraction</td><td>${c.cf_frac != null ? (c.cf_frac*100).toFixed(0)+'%' : '—'}</td></tr>
      <tr><td>Sweep rate</td><td>${c.sweep.toFixed(2)} kHz/ms</td></tr>
      ${c.det_prob > 0 ? `<tr><td>Det. score</td><td>${c.det_prob.toFixed(2)}</td></tr>` : ''}
    </table>
    <button id="btn-zoom-call"
      style="margin-top:8px;width:100%;background:#222;border:1px solid #3a3a3a;
             color:#ccc;padding:4px 0;border-radius:3px;cursor:pointer;font-size:11px;">
      ⊕ Zoom to call
    </button>
  `;
  document.getElementById('btn-zoom-call').onclick = () => zoomToCall(S.selectedCall);
  // Clicking a call always opens the call pane and closes any open species
  _setAccordionState('call');
}

// ─── Species accordion (bottom) ──────────────────────────────
// Shared stats helper
function _spStat(arr) {
  if (!arr.length) return null;
  const n    = arr.length;
  const mean = arr.reduce((s, x) => s + x, 0) / n;
  const sd   = Math.sqrt(arr.reduce((s, x) => s + (x - mean) ** 2, 0) / n);
  return { n, mean, sd, min: Math.min(...arr), max: Math.max(...arr) };
}
function _spStatRow(label, s, unit, d=1) {
  const f = (x) => x.toFixed(d);
  return s
    ? `<tr><td>${label}</td><td>${f(s.mean)}</td><td>±${f(s.sd)}</td><td>${f(s.min)}–${f(s.max)}</td><td>${unit}</td></tr>`
    : `<tr><td colspan="5" style="color:#333">${label}: no data</td></tr>`;
}

function _buildSpContent(sp) {
  const prof  = _profiles.find(p => p.name === sp);
  const col   = S.colors[sp] || '#888';
  const calls = S.calls.filter(c => c.species === sp);
  const total = S.calls.length;
  const pct   = total ? (calls.length / total * 100).toFixed(1) : '0';

  const fpeak = _spStat(calls.map(c => c.Fpeak));
  const fmin  = _spStat(calls.map(c => c.Fmin));
  const fmax  = _spStat(calls.map(c => c.Fmax));
  const bw    = _spStat(calls.map(c => c.Fmax - c.Fmin));
  const dur   = _spStat(calls.map(c => c.dur));
  const swp   = _spStat(calls.map(c => c.sweep));
  const conf  = _spStat(calls.map(c => c.conf * 100));

  return `
    <div class="sp-section">
      <h4>Recording — ${calls.length} calls (${pct}%)</h4>
      ${calls.length === 0
        ? '<p>No calls detected.</p>'
        : `<table class="sp-stats-tbl">
          <thead><tr><th>Param</th><th>Mean</th><th>±SD</th><th>Range</th><th></th></tr></thead>
          <tbody>
            ${_spStatRow('Fpeak', fpeak, 'kHz')}
            ${_spStatRow('Fmin',  fmin,  'kHz')}
            ${_spStatRow('Fmax',  fmax,  'kHz')}
            ${_spStatRow('BW',    bw,    'kHz')}
            ${_spStatRow('Dur',   dur,   'ms')}
            ${_spStatRow('Sweep', swp,   'kHz/ms', 2)}
            ${_spStatRow('Conf',  conf,  '%', 0)}
          </tbody>
        </table>`}
    </div>
    ${prof ? `
    ${prof.Fchar ? `
    <div class="sp-section">
      <h4>Classification Profile</h4>
      <div class="sp-profile-row"><span class="prl">Char. freq (Fchar)</span><span class="prv">${prof.Fchar[0]}–${prof.Fchar[1]} kHz</span></div>
      <div class="sp-profile-row"><span class="prl">Min freq (Fmin)</span><span class="prv">${prof.Fmin[0]}–${prof.Fmin[1]} kHz</span></div>
      <div class="sp-profile-row"><span class="prl">Duration</span><span class="prv">${prof.dur[0]}–${prof.dur[1]} ms</span></div>
      <div class="sp-profile-row"><span class="prl">FM sweep</span><span class="prv">${prof.sweep[0]}–${prof.sweep[1]} kHz/ms</span></div>
      <div class="sp-profile-row"><span class="prl">Typical IPI</span><span class="prv">${prof.ipi_ms} ms</span></div>
    </div>` : ''}
    <div class="sp-section">
      <h4>Call Type</h4>
      <p>${prof.call_type}</p>
    </div>
    <div class="sp-section">
      <h4>Natural History</h4>
      <p>${prof.desc}</p>
    </div>
    <div class="sp-section">
      <h4>Habitat · Range</h4>
      <p>${prof.habitat}</p>
      <p style="margin-top:4px">${prof.range}</p>
    </div>
    ${prof.refs.length ? `
    <div class="sp-section">
      <h4>References</h4>
      ${prof.refs.map(r => {
        const [text, url] = Array.isArray(r) ? r : [r, null];
        return url
          ? `<a class="ref-tag" href="${url}" target="_blank" rel="noopener">${text}</a>`
          : `<span class="ref-tag">${text}</span>`;
      }).join('')}
    </div>` : ''}
    ` : ''}
  `;
}

function soloSpecies(sp) {
  if (S.soloedSpecies === sp) {
    // Un-solo: restore visibility from checkboxes
    S.soloedSpecies = null;
    S.hiddenSpecies.clear();
    document.querySelectorAll('.sp-acc-header').forEach(hdr => {
      if (!hdr.querySelector('.sp-acc-chk').checked)
        S.hiddenSpecies.add(hdr.dataset.sp);
    });
  } else {
    // Solo: hide everyone except this species
    S.soloedSpecies = sp;
    S.hiddenSpecies.clear();
    for (const s of Object.keys(S.colors)) {
      if (s !== sp) S.hiddenSpecies.add(s);
    }
  }
  // Refresh row visuals in-place (avoid full rebuild)
  document.querySelectorAll('.sp-acc-header').forEach(hdr => {
    const s = hdr.dataset.sp;
    hdr.classList.toggle('hidden-sp', S.hiddenSpecies.has(s));
    const btn = hdr.querySelector('.sp-solo-btn');
    if (btn) {
      const active = (s === S.soloedSpecies);
      btn.classList.toggle('soloed', active);
      btn.textContent = active ? '●' : '○';
      btn.title = active ? `Un-solo ${s}` : `Solo: show only ${s}`;
    }
  });
  scheduleRender();
}

function buildLegend(colors) {
  const el = document.getElementById('acc-sp-scroll');
  el.innerHTML = '';
  S.soloedSpecies = null;   // reset solo whenever legend is rebuilt
  // If a species was open, close it (the items are being destroyed anyway)
  if (_openAcc && _openAcc !== 'call') _openAcc = null;

  const counts = {};
  for (const c of S.calls) counts[c.species] = (counts[c.species] || 0) + 1;

  for (const [sp, col] of Object.entries(colors)) {
    const n      = counts[sp] || 0;
    // Hide species with no detected calls once detection is complete
    if (n === 0 && S.calls.length > 0) continue;
    const hidden = S.hiddenSpecies.has(sp);

    // Outer item — carries data-sp for _setAccordionState querySelector
    const item = document.createElement('div');
    item.className = 'sp-acc-item';
    item.dataset.sp = sp;

    // Header row (always visible)
    const hdr = document.createElement('div');
    hdr.className = 'sp-acc-header' + (hidden ? ' hidden-sp' : '');
    hdr.dataset.sp = sp;
    hdr.innerHTML = `
      <input type="checkbox" class="sp-acc-chk" ${hidden ? '' : 'checked'} title="Show/hide ${sp}">
      <div class="sp-acc-swatch" style="background:${col}"></div>
      <span class="sp-acc-name">${sp}</span>
      ${n ? `<span class="sp-acc-count">${n}</span>` : ''}
      <button class="sp-solo-btn" data-sp="${sp}" title="Solo: show only ${sp}">○</button>
      <span class="sp-acc-arrow">▴</span>
    `;

    // Collapsible body — content injected by _setAccordionState
    const body = document.createElement('div');
    body.className = 'sp-acc-body';

    item.appendChild(hdr);
    item.appendChild(body);

    // Checkbox → toggle visibility; exits solo mode if active
    const chk = hdr.querySelector('.sp-acc-chk');
    chk.addEventListener('change', e => {
      e.stopPropagation();
      if (S.soloedSpecies !== null) {
        S.soloedSpecies = null;
        document.querySelectorAll('.sp-solo-btn').forEach(b => {
          b.classList.remove('soloed'); b.textContent = '○';
          b.title = `Solo: show only ${b.dataset.sp}`;
        });
      }
      if (S.hiddenSpecies.has(sp)) S.hiddenSpecies.delete(sp);
      else                          S.hiddenSpecies.add(sp);
      hdr.classList.toggle('hidden-sp', S.hiddenSpecies.has(sp));
      scheduleRender();
    });

    // Solo button → show only this species (or un-solo if already soloed)
    const soloBtn = hdr.querySelector('.sp-solo-btn');
    soloBtn.addEventListener('click', e => {
      e.stopPropagation();
      soloSpecies(sp);
    });

    // Header click (not checkbox, not solo btn) → inline accordion toggle
    hdr.addEventListener('click', e => {
      if (e.target === chk || e.target === soloBtn) return;
      _setAccordionState(_openAcc === sp ? null : sp);
    });

    el.appendChild(item);
  }
}

// ─── Helpers ─────────────────────────────────────────────────
function fmt(t) {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}

// ── PSD mode toggle ───────────────────────────────────────────────
function setPsdMode(mode) {
  S.psdMode = mode;
  _updatePsdModeButtons();
  // Force a fresh fetch by invalidating the stale-check state
  _psdT0 = -1; _psdT1 = -1;
  if (typeof fetchPSD === 'function') fetchPSD();
}

function _updatePsdModeButtons() {
  const avg = document.getElementById('psd-avg');
  const ph  = document.getElementById('psd-ph');
  if (!avg || !ph) return;
  avg.classList.toggle('clf-active', S.psdMode === 'view');
  ph.classList.toggle('clf-active',  S.psdMode === 'playhead');
}

// ── Follow-playhead toggle ────────────────────────────────────────
function toggleFollowPlayhead() {
  S.followPlayhead = !S.followPlayhead;
  updateFollowButton();
}

function updateFollowButton() {
  const btn = document.getElementById('btn-follow');
  if (!btn) return;
  if (S.followPlayhead) {
    btn.style.background = '#0d2a20';
    btn.style.border     = '1px solid #1a4a3a';
    btn.style.color      = '#00d488';
  } else {
    btn.style.background = '#1a1a1a';
    btn.style.border     = '1px solid #333';
    btn.style.color      = '#555';
  }
}

// Format seconds as HH:MM:SS.sss (playhead position display)
function fmtHMS(t) {
  const h   = Math.floor(t / 3600);
  const m   = Math.floor((t % 3600) / 60);
  const s   = Math.floor(t % 60);
  const ms  = Math.round((t % 1) * 1000);
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
}

// Absolute wall-clock time: recordingStart epoch ms + offset seconds.
// Returns "HH:MM:SS".  If two absolute times span a date, the second gets "+1d" etc.
function fmtAbsMs(ms) {
  const d = new Date(ms);
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}
function fmtAbs(offsetS) {
  if (!S.recordingStart) return fmt(offsetS);
  return fmtAbsMs(S.recordingStart + offsetS * 1000);
}
function fmtAbsFull(offsetS) {
  if (!S.recordingStart) return fmt(offsetS);
  const d = new Date(S.recordingStart + offsetS * 1000);
  const mo = String(d.getMonth()+1).padStart(2,'0');
  const dd = String(d.getDate()).padStart(2,'0');
  return `${d.getFullYear()}-${mo}-${dd} ${fmtAbsMs(S.recordingStart + offsetS*1000)}`;
}
// Returns true if offsetS0 and offsetS1 are on different calendar days
function _spansMidnight(offsetS0, offsetS1) {
  if (!S.recordingStart) return false;
  const d0 = new Date(S.recordingStart + offsetS0 * 1000);
  const d1 = new Date(S.recordingStart + offsetS1 * 1000);
  return d0.toDateString() !== d1.toDateString();
}

// ─── Viewport boost ───────────────────────────────────────────
// Sends the current viewport time range to the server so the scheduler
// can bump those tiles to highest priority.
let _lastBoostKey = null;
setInterval(() => {
  if (!S.duration) return;
  const key = `${S.viewStart.toFixed(1)}_${S.viewDur.toFixed(1)}`;
  if (key === _lastBoostKey) return;
  _lastBoostKey = key;
  fetch(`/api/boost?f=${S.fid}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ t0: S.viewStart, t1: S.viewStart + S.viewDur }),
  }).catch(() => {});
}, 500);

// ─── Tile progress overlay ────────────────────────────────────
function _tileProgressDone(tp) {
  if (!tp) return true;
  return ['raw', 'flat', 'mask'].every(k =>
    tp[k].status === 'done' || tp[k].status === 'idle');
}

function _updateTileProgress(tp) {
  const el = document.getElementById('tile-prog');
  if (!tp || _tileProgressDone(tp)) { el.style.display = 'none'; return; }

  const LABELS = { raw: 'raw', flat: 'flat', mask: 'mask' };
  let html = '<div class="tp-title">Tiles</div>';
  for (const [key, label] of Object.entries(LABELS)) {
    const p = tp[key];
    if (p.status === 'done' || p.status === 'idle') continue;
    const pct   = p.total > 0 ? (p.done / p.total * 100).toFixed(1) : 0;
    const cnt   = p.status === 'waiting' ? 'waiting' : `${p.done} / ${p.total}`;
    const cls   = p.status === 'running' ? ' running' : '';
    html += `<div class="tp-row">
      <span class="tp-lbl">${label}</span>
      <span class="tp-bar"><span class="tp-fill${cls}" style="width:${pct}%"></span></span>
      <span class="tp-cnt">${cnt}</span>
    </div>`;
  }
  el.innerHTML = html;
  el.style.display = 'block';
}

// ─── Init ─────────────────────────────────────────────────────
async function init() {
  window.addEventListener('resize', resize);
  resize();

  // Read file ID from URL (?f=<fid>); empty string means "use server default"
  S.fid = new URLSearchParams(window.location.search).get('f') || '';

  // Fetch info — retry until we get a 200 OK.
  // A background file may still be loading (ffprobe / ffmpeg decode) and will
  // return 503 until ready; we keep polling so the user just sees a spinner.
  let info;
  for (let attempt = 0; attempt < 120; attempt++) {
    try {
      const r = await fetch(`/api/info?f=${S.fid}`);
      if (r.ok) { info = await r.json(); break; }
    } catch {}
    await sleep(1000);
  }
  if (!info) { console.error('Could not load file info after 120 s'); return; }

  // Canonicalise S.fid from server response and update the URL so refreshing
  // always returns to the same file, even if we loaded via bare /.
  S.fid = info.fid || S.fid;
  {
    const url = new URL(window.location);
    if (url.searchParams.get('f') !== S.fid) {
      url.searchParams.set('f', S.fid);
      window.history.replaceState({}, '', url);
    }
  }

  _fileInfo     = info;
  S.duration    = info.duration_s;
  S.freqLow     = info.freq_low;
  S.freqHigh    = info.freq_high;
  S.tileDur     = info.tile_duration;
  S.nTiles      = info.n_tiles;
  S.tileVersion = info.tile_version ?? 0;
  S.colors      = info.colors;
  S.viewDur   = Math.min(30, S.duration);
  S.ovDur     = S.duration;   // overview starts showing the full recording
  TILE_FREQ_LOW  = info.freq_low;
  TILE_FREQ_HIGH = info.freq_high;
  S.nyquist      = info.freq_high;  // sr/2 in kHz
  psdViewLow  = TILE_FREQ_LOW;   // floor = preprocessing cutoff, same as canvas min
  psdViewHigh = S.nyquist;
  if (info.recording_start)
    S.recordingStart = new Date(info.recording_start).getTime();
  updateScrollbar();

  // Set up audio player with the new file (lazy: AudioContext created on first play)
  if (typeof audioInit === 'function')
    audioInit(S.fid, info.sr, Math.round(info.duration_s * info.sr));
  if (typeof _initRateSlider === 'function')
    _initRateSlider();

  // BPF attenuation slider
  {
    const sl  = document.getElementById('slider-bpf-att');
    const val = document.getElementById('bpf-att-val');
    if (sl && val) {
      sl.addEventListener('input', () => {
        const pct = +sl.value;
        val.textContent = pct + '%';
        if (typeof updateTrack === 'function') updateTrack(sl);
        audioSetBPFAtt(pct / 100);
      });
      if (typeof updateTrack === 'function') updateTrack(sl);
    }
  }

  updateFollowButton();
  _updatePsdModeButtons();

  try { _profiles = await (await fetch(`/api/profiles?f=${S.fid}`)).json(); } catch {}
  S.colors = info.colors;
  buildLegend(S.colors);

  function _renderFileMeta(callCount) {
    const parts = [
      `${(info.duration_s / 60).toFixed(1)} min`,
      `${(info.sr / 1000).toFixed(0)} kHz`,
    ];
    if (info.bit_depth) parts.push(info.bit_depth);
    parts.push(`${info.n_tiles} tiles`);
    if (callCount != null) parts.push(`${callCount} calls`);
    if (info.recording_start) {
      const d = new Date(info.recording_start);
      const mo = String(d.getMonth()+1).padStart(2,'0');
      const dd = String(d.getDate()).padStart(2,'0');
      const HH = String(d.getHours()).padStart(2,'0');
      const MM = String(d.getMinutes()).padStart(2,'0');
      parts.push(`${d.getFullYear()}-${mo}-${dd} ${HH}:${MM}`);
    }
    if (info.location) parts.push(info.location);
    document.getElementById('file-meta').textContent = parts.join('  ·  ');
  }
  _renderFileMeta(null);

  // Populate file selector (values are fids, labels are filenames)
  try {
    const fres = await (await fetch(`/api/files?f=${S.fid}`)).json();
    const sel  = document.getElementById('file-select');
    sel.innerHTML = '';
    for (const f of fres.files) {
      const opt       = document.createElement('option');
      opt.value       = f.fid;
      opt.textContent = f.name;
      if (f.fid === fres.current) opt.selected = true;
      sel.appendChild(opt);
    }
    // Hide selector if there's only one file
    sel.style.display = fres.files.length > 1 ? '' : 'none';
  } catch {}

  // Poll for detection progress
  const overlay  = document.getElementById('progress-overlay');
  const msgEl    = document.getElementById('progress-msg');
  const pbar     = document.getElementById('pbar');
  while (true) {
    const st = await (await fetch(`/api/status?f=${S.fid}`)).json();
    msgEl.textContent = st.progress.status;
    const pct = st.progress.total > 0 ? st.progress.done / st.progress.total * 100 : 0;
    pbar.style.width  = pct + '%';
    if (st.ready) break;
    await sleep(1500);
  }
  overlay.style.display = 'none';

  // Poll tile generation progress; fast while generating, slow once done.
  ;(async () => {
    let interval = 0;   // first poll immediately
    while (true) {
      await sleep(interval);
      let st;
      try { st = await (await fetch(`/api/status?f=${S.fid}`)).json(); } catch {
        interval = 2000; continue;
      }
      _updateTileProgress(st.tile_progress);
      interval = _tileProgressDone(st.tile_progress) ? 60_000 : 2_000;
    }
  })();

  // Fetch calls for the default detector (batdetect2)
  const res  = await (await fetch(`/api/calls?f=${S.fid}&detector=batdetect2`)).json();
  S.calls = res.calls;
  // Stash both classifier results so setModel() can switch between them client-side
  if (typeof _stashClassifierFields === 'function') _stashClassifierFields(S.calls);
  // Cache in the detector store so switching away and back doesn't re-fetch
  if (typeof _callsByDetector !== 'undefined') _callsByDetector['batdetect2'] = S.calls;
  _renderFileMeta(S.calls.length);
  document.getElementById('status-bar').textContent = '';
  buildLegend(S.colors);  // rebuild with call counts now available

  // ─── Restore viewport / selection / modal from URL params ────
  // Applied after calls are loaded so jumpToCallId can find the call.
  {
    const p     = new URLSearchParams(window.location.search);
    const tP    = parseFloat(p.get('t')  ?? '');
    const vdP   = parseFloat(p.get('vd') ?? '');
    const flP   = parseFloat(p.get('fl') ?? '');
    const fhP   = parseFloat(p.get('fh') ?? '');
    const callP = p.get('call');
    const modalP = p.get('modal');

    let hasViewport = false;
    if (isFinite(tP))  { S.viewStart = Math.max(0, Math.min(S.duration - 0.5, tP)); hasViewport = true; }
    if (isFinite(vdP)) { S.viewDur   = Math.max(0.5, Math.min(S.duration, vdP));     hasViewport = true; }
    if (isFinite(flP) && isFinite(fhP) && fhP > flP) {
      S.freqLow  = Math.max(TILE_FREQ_LOW, flP);
      S.freqHigh = Math.min(S.nyquist, fhP);
      updateScrollbar();
    }
    // Select the call; skip zoom-to-call if an explicit viewport was supplied
    if (callP !== null) {
      const id = parseInt(callP);
      if (!isNaN(id)) jumpToCallId(id, !hasViewport);
    }
    if (modalP === 'about')   openAbout();
    if (modalP === 'session') openSession();
  }

  _urlReady = true;   // allow _scheduleURLSync to start writing from here on
  scheduleRender();
}

let _profiles = [];   // loaded from /api/profiles in init()
let _fileInfo = {};   // raw /api/info response — used by openAbout()

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── File switcher ────────────────────────────────────────────
// Navigate to ?f=<fid> — the page reloads and init() picks up the new file.
// No server call needed; all state is derived from the fid in the URL.
function switchFile(fid) {
  const url = new URL(window.location);
  url.searchParams.set('f', fid);
  // Clear per-file viewport state when switching files
  for (const k of ['t', 'vd', 'fl', 'fh', 'call', 'modal'])
    url.searchParams.delete(k);
  window.location.href = url.toString();
}

// ─── URL state sync ───────────────────────────────────────────
// Keeps ?t, ?vd, ?fl, ?fh, ?call, ?modal in sync with the live view so
// users can copy a URL that links to the exact viewport/selection they see.
// Uses replaceState (not pushState) so panning doesn't pollute browser history.
let _urlSyncTimer    = null;
let _urlReady        = false;   // set at end of init(); guards against premature writes
let _urlLastSchedule = 0;       // throttle: skip redundant clearTimeout+setTimeout at 60fps

function _scheduleURLSync() {
  if (!_urlReady) return;
  // Throttle: reschedule at most once per 250ms (just under the 300ms debounce
  // delay), so rapid scheduleRender() calls don't spam clearTimeout+setTimeout
  // at 60fps.  The actual URL write still fires 300ms after the last reschedule.
  const now = performance.now();
  if (_urlSyncTimer && now - _urlLastSchedule < 250) return;
  _urlLastSchedule = now;
  clearTimeout(_urlSyncTimer);
  _urlSyncTimer = setTimeout(_syncURL, 300);
}

function _syncURL() {
  _urlSyncTimer = null;
  const url = new URL(window.location);
  const t   = S.viewStart.toFixed(1);
  const vd  = S.viewDur.toFixed(1);
  const fl  = S.freqLow.toFixed(1);
  const fh  = S.freqHigh.toFixed(1);
  const cid = S.selectedCall ? String(S.selectedCall.id) : null;
  if (url.searchParams.get('t')  === t  &&
      url.searchParams.get('vd') === vd &&
      url.searchParams.get('fl') === fl &&
      url.searchParams.get('fh') === fh &&
      (url.searchParams.get('call') ?? null) === cid) return;
  url.searchParams.set('t',  t);
  url.searchParams.set('vd', vd);
  url.searchParams.set('fl', fl);
  url.searchParams.set('fh', fh);
  if (cid !== null) url.searchParams.set('call', cid);
  else              url.searchParams.delete('call');
  window.history.replaceState({}, '', url);
}

// ─── Modal helpers ────────────────────────────────────────────
function closeModal(id) {
  const url = new URL(window.location);
  url.searchParams.delete('modal');
  window.history.replaceState({}, '', url);
  document.getElementById(id).classList.remove('open');
}

function openAbout() {
  const url = new URL(window.location);
  url.searchParams.set('modal', 'about');
  window.history.replaceState({}, '', url);
  // Inject live file metadata so bit-depth, sample rate, etc. are always accurate.
  const el = document.getElementById('about-recording');
  if (el && _fileInfo.sr) {
    const sr     = (_fileInfo.sr / 1000).toFixed(0) + ' kHz';
    const depth  = _fileInfo.bit_depth || '??-bit';
    const ch     = _fileInfo.channels === 1 ? 'mono' : _fileInfo.channels === 2 ? 'stereo' : `${_fileInfo.channels}ch`;
    const mins   = Math.floor(_fileInfo.duration_s / 60);
    const secs   = Math.round(_fileInfo.duration_s % 60);
    const dur    = `${mins} min ${String(secs).padStart(2,'0')} sec`;
    el.textContent = `Recorded 2025-05-28 at 19:42, Campbell Ave bridge over the Rillito River, Tucson AZ, USA. Zoom F3 field recorder. ${sr} / ${depth} ${ch} FLAC, ${dur}.`;
  }
  document.getElementById('about-modal').classList.add('open');
}

let _sessionLoaded = false;
async function openSession() {
  const url = new URL(window.location);
  url.searchParams.set('modal', 'session');
  window.history.replaceState({}, '', url);
  document.getElementById('session-modal').classList.add('open');
  if (_sessionLoaded) return;
  _sessionLoaded = true;
  const body = document.getElementById('session-body');
  body.innerHTML = '<p style="color:#555;font-size:12px">Loading conversation…</p>';
  try {
    const data = await (await fetch('/api/conversation')).json();
    const msgs = data.messages;
    if (!msgs || msgs.length === 0) {
      body.innerHTML = '<p style="color:#555;font-size:12px">Conversation log not found.</p>';
      return;
    }
    // Deduplicate consecutive same-role TEXT messages (multi-part assistant turns).
    // Tool and note entries are never merged.
    const NEVER_MERGE = new Set(['tool', 'note']);
    const deduped = [msgs[0]];
    for (let i = 1; i < msgs.length; i++) {
      const cur  = msgs[i];
      const prev = deduped[deduped.length - 1];
      if (!NEVER_MERGE.has(cur.role) && !NEVER_MERGE.has(prev.role) && cur.role === prev.role) {
        prev.text += '\n\n' + cur.text;
      } else {
        deduped.push({...cur});
      }
    }
    const esc = s => (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

    // Helper: render a single tool row (used both standalone and inside a bubble)
    const renderTool = (t, insideBubble = false) => {
      const errCls    = t.is_error ? ' tool-error' : '';
      const hasDetail = !!t.detail;
      const extraCls  = insideBubble ? ' bubble-tool' : '';
      const durStr    = t.duration_s != null
        ? `<span class="bt-dur"> [${t.duration_s}s]</span>` : '';
      const result    = t.result
        ? `<span class="tool-result"> → ${esc(t.result)}</span>` : '';
      const expander  = hasDetail ? `<span class="tool-expander">▶</span>` : '';
      const detail    = hasDetail
        ? `<div class="tool-detail"><pre class="tool-cmd">${esc(t.detail)}</pre></div>` : '';
      return `<div class="conv-turn tool${errCls}${hasDetail ? ' has-detail' : ''}${extraCls}">
        <div class="tool-row"><em class="tool-icon">⚙</em
        ><span class="tool-name">${esc(t.name)}</span
        ><span class="tool-summary">${esc(t.summary)}</span>${result}${durStr}${expander}</div>${detail}</div>`;
    };

    // Helper: build stats HTML for an assistant message
    const renderStats = m => {
      if (!m.stats) return '';
      const st = m.stats, parts = [];
      if (st.duration_s   != null) parts.push(`<span class="cs-dur">⏱ ${st.duration_s}s</span>`);
      if (st.output_tokens != null) parts.push(`<span class="cs-out">${st.output_tokens.toLocaleString()}↓</span>`);
      if (st.input_tokens  != null && st.input_tokens > 0)
                                   parts.push(`<span class="cs-in">${(st.input_tokens/1000).toFixed(1)}k↑</span>`);
      return parts.length ? `<div class="conv-stats">${parts.join('')}</div>` : '';
    };

    // ── Map section messageIndex (original) → deduped index ──────────────────
    // sections[si].messageIndex references the *original* (pre-dedup) messages array.
    // We replay the dedup logic to find which deduped index each original index maps to.
    const sections = data.sections || [];
    const sectionAtDeduped = new Map();   // dedupedIdx → section index (si)
    {
      const NEVER_MERGE2 = new Set(['tool', 'note']);
      const tmp = [msgs[0]];
      const origToDeduped = new Map([[0, 0]]);
      for (let oi = 1; oi < msgs.length; oi++) {
        const cur  = msgs[oi];
        const prev = tmp[tmp.length - 1];
        if (!NEVER_MERGE2.has(cur.role) && !NEVER_MERGE2.has(prev.role) && cur.role === prev.role) {
          origToDeduped.set(oi, tmp.length - 1);
        } else {
          tmp.push(cur);
          origToDeduped.set(oi, tmp.length - 1);
        }
      }
      // Populate sectionAtDeduped; first section wins if two map to the same deduped index.
      for (let si = 0; si < sections.length; si++) {
        const di = origToDeduped.get(sections[si].messageIndex) ?? 0;
        if (!sectionAtDeduped.has(di)) sectionAtDeduped.set(di, si);
      }
    }

    // ── Walk deduped, injecting section headers inline ────────────────────────
    // Tool entries that follow an assistant are folded into the assistant bubble.
    // Section headers are inserted at the deduped index where the section starts,
    // so they always appear at exactly the right turn regardless of tool folding.
    const parts = [];
    let i = 0;
    while (i < deduped.length) {
      // Inject any section header that begins at this deduped index
      if (sectionAtDeduped.has(i)) {
        const si = sectionAtDeduped.get(i);
        parts.push(
          `<div class="conv-section-hdr" id="conv-sec-${si}">${esc(sections[si].title)}</div>`
        );
      }

      const m = deduped[i];

      if (m.role === 'assistant') {
        // Collect tool entries that immediately follow this assistant message
        let j = i + 1;
        while (j < deduped.length && deduped[j].role === 'tool') j++;
        const tools = deduped.slice(i + 1, j);

        const toolsHtml = tools.length
          ? `<div class="bubble-tools">${tools.map(t => renderTool(t, true)).join('')}</div>`
          : '';
        parts.push(`<div class="conv-turn assistant">
          <div class="role">🤖 Claude</div>
          <div class="bubble">${esc(m.text)}${toolsHtml}</div>
          ${renderStats(m)}
        </div>`);
        i = j;
        continue;
      }

      if (m.role === 'tool') {
        // Standalone tool not preceded by an assistant message (rare)
        parts.push(renderTool(m, false));
        i++; continue;
      }

      if (m.role === 'note') {
        parts.push(`<div class="conv-turn note">
          <div class="role">📋 Note</div>
          <div class="bubble">${esc(m.text)}</div>
        </div>`);
        i++; continue;
      }

      // user
      parts.push(`<div class="conv-turn user">
        <div class="role">👤 Brandon</div>
        <div class="bubble">${esc(m.text)}</div>
      </div>`);
      i++;
    }

    body.innerHTML = parts.join('');

    // ── Build section nav sidebar ─────────────────────────────────
    const nav = document.getElementById('session-nav');
    if (nav && sections.length > 1) {
      nav.innerHTML = sections.map((sec, si) =>
        `<div class="snav-item" data-si="${si}">
           <div class="snav-line"></div>
           <div class="snav-label">${esc(sec.title)}</div>
         </div>`
      ).join('');

      // Scroll helper: compute element's top relative to the scrollable body container
      const topInBody = el => {
        const elRect   = el.getBoundingClientRect();
        const bodyRect = body.getBoundingClientRect();
        return elRect.top - bodyRect.top + body.scrollTop;
      };

      // Click → scroll to section header
      nav.addEventListener('click', e => {
        const item = e.target.closest('.snav-item');
        if (!item) return;
        const si  = +item.dataset.si;
        const hdr = body.querySelector(`#conv-sec-${si}`);
        if (hdr) body.scrollTo({ top: topInBody(hdr), behavior: 'smooth' });
      });

      // Scrollspy: highlight the nav item matching the visible section
      const navItems = Array.from(nav.querySelectorAll('.snav-item'));
      const updateActive = () => {
        const hdrs     = sections.map((_, si) => body.querySelector(`#conv-sec-${si}`));
        const bodyRect = body.getBoundingClientRect();
        let active = 0;
        for (let si = 0; si < hdrs.length; si++) {
          if (hdrs[si]) {
            const top = hdrs[si].getBoundingClientRect().top - bodyRect.top;
            if (top <= 60) active = si;   // 60px lookahead
          }
        }
        navItems.forEach((item, i) => item.classList.toggle('snav-active', i === active));
      };
      body.addEventListener('scroll', updateActive, { passive: true });
      updateActive();
    }

    // Expand/collapse tool entries via event delegation
    body.addEventListener('click', e => {
      const row = e.target.closest('.has-detail .tool-row');
      if (row) row.closest('.conv-turn.tool').classList.toggle('open');
    });
  } catch(e) {
    body.innerHTML = `<p style="color:#c04;font-size:12px">Error loading conversation: ${e.message}</p>`;
  }
}

// Close modals on Escape — also clears ?modal= from URL
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const anyOpen = document.querySelector('.modal-backdrop.open');
    document.querySelectorAll('.modal-backdrop.open')
      .forEach(m => m.classList.remove('open'));
    if (anyOpen) {
      const url = new URL(window.location);
      url.searchParams.delete('modal');
      window.history.replaceState({}, '', url);
    }
  }
});

init();

// ── Range slider: track fill + cross-browser thumb position ─────────────
// updateTrack(el) recomputes the --pct CSS custom property so the linear-
// gradient background shows the correct filled portion left of the thumb.
function updateTrack(el) {
  const min = parseFloat(el.min) || 0;
  const max = parseFloat(el.max) || 100;
  const pct = ((parseFloat(el.value) - min) / (max - min) * 100).toFixed(2) + '%';
  el.style.setProperty('--pct', pct);
}

// Initialise every range slider:
//   • Use the HTML *attribute* value (getAttribute), not the IDL *property*
//     value (.value), because Firefox sometimes initialises the property to
//     max before the first paint, which would cause our fix to "restore" the
//     wrong value.
//   • Set el.value explicitly so all browsers render the thumb at the right
//     spot from the start, without needing a click.
document.querySelectorAll('input[type=range]').forEach(el => {
  const htmlVal = el.getAttribute('value') ?? el.value;
  el.value = htmlVal;
  updateTrack(el);
});
