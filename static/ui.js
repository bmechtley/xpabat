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

function navigateCall(delta) {
  const visible = S.calls.filter(c => !S.hiddenSpecies.has(c.species) && c.conf >= S.minConf);
  if (!visible.length) return;
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
  renderDetail(call);
  zoomToCall(call);
}

function jumpToCallId(id) {
  const call = S.calls.find(c => c.id === id);
  if (!call) return;
  S.selectedCall = call;
  renderDetail(call);
  zoomToCall(call);
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

// ─── Init ─────────────────────────────────────────────────────
async function init() {
  window.addEventListener('resize', resize);
  resize();

  // Fetch info
  let info;
  for (let attempt = 0; attempt < 30; attempt++) {
    try { info = await (await fetch('/api/info')).json(); break; }
    catch { await sleep(1000); }
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
  try { _profiles = await (await fetch('/api/profiles')).json(); } catch {}
  S.colors = info.colors;
  buildLegend(S.colors);

  {
    const parts = [
      `${(info.duration_s / 60).toFixed(1)} min`,
      `${(info.sr / 1000).toFixed(0)} kHz`,
    ];
    if (info.bit_depth) parts.push(info.bit_depth);
    parts.push(`${info.n_tiles} tiles`);
    if (info.recording_start) {
      const d = new Date(info.recording_start);
      const mo = String(d.getMonth()+1).padStart(2,'0');
      const dd = String(d.getDate()).padStart(2,'0');
      const HH = String(d.getHours()).padStart(2,'0');
      const MM = String(d.getMinutes()).padStart(2,'0');
      parts.push(`${d.getFullYear()}-${mo}-${dd} ${HH}:${MM}`);
    }
    document.getElementById('file-meta').textContent = parts.join('  ·  ');
  }

  // Poll for detection progress
  const overlay  = document.getElementById('progress-overlay');
  const msgEl    = document.getElementById('progress-msg');
  const pbar     = document.getElementById('pbar');
  while (true) {
    const st = await (await fetch('/api/status')).json();
    msgEl.textContent = st.progress.status;
    const pct = st.progress.total > 0 ? st.progress.done / st.progress.total * 100 : 0;
    pbar.style.width  = pct + '%';
    if (st.ready) break;
    await sleep(1500);
  }
  overlay.style.display = 'none';

  // Fetch calls
  const res  = await (await fetch('/api/calls')).json();
  S.calls = res.calls;
  // Stash both classifier results so setClassifier() can switch between them
  for (const c of S.calls) {
    c.species_v2 = c.species;   c.conf_v2 = c.conf;
    c.color_v2   = c.color;     c.short_v2 = c.short;
    c.species_v1 = c.species_v1 ?? c.species;
    c.conf_v1    = c.conf_v1    ?? c.conf;
    c.color_v1   = c.color_v1   ?? c.color;
    c.short_v1   = c.short_v1   ?? c.short;
  }
  document.getElementById('status-bar').textContent =
    `${S.calls.length} calls`;
  buildLegend(S.colors);  // rebuild with call counts now available
  scheduleRender();
}

let _profiles = [];   // loaded from /api/profiles in init()
let _fileInfo = {};   // raw /api/info response — used by openAbout()

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Modal helpers ────────────────────────────────────────────
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
}

function openAbout() {
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
    // Deduplicate consecutive same-role messages (multi-part assistant turns)
    const deduped = [msgs[0]];
    for (let i = 1; i < msgs.length; i++) {
      if (msgs[i].role === deduped[deduped.length-1].role) {
        deduped[deduped.length-1].text += '\n\n' + msgs[i].text;
      } else {
        deduped.push({...msgs[i]});
      }
    }
    body.innerHTML = deduped.map(m => {
      const ts = m.ts ? new Date(m.ts).toLocaleString() : '';
      const escaped = m.text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      return `<div class="conv-turn ${m.role}">
        <div class="role">${m.role === 'user' ? '👤 You' : '🤖 Claude'}</div>
        <div class="bubble">${escaped}</div>
        ${ts ? `<div class="ts">${ts}</div>` : ''}
      </div>`;
    }).join('');
  } catch(e) {
    body.innerHTML = `<p style="color:#c04;font-size:12px">Error loading conversation: ${e.message}</p>`;
  }
}

// Close modals on Escape
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-backdrop.open')
      .forEach(m => m.classList.remove('open'));
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
