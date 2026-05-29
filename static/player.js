// player.js — browser-side audio playback engine
//
// Public API (called from events.js, ui.js):
//   audioInit(fid, srcSr, totalFrames)  — call after /api/info returns
//   audioPlayPause()                     — toggle play / pause
//   audioSeek(t)                         — jump to absolute time t (seconds)
//   audioSetRate(rate)                   — set playback rate (0 < rate ≤ 1)
//   updatePlayButton()                   — sync button text to S.isPlaying
//
// Rate slider values 0–5 → 1/32×, 1/16×, 1/8×, 1/4×, 1/2×, 1×
// Default: index 1 → 1/16× (bat calls at 192 kHz → ~2.5 kHz; audible)

const _RING_SIZE = 1_920_000;
const _SAB_BYTES = 16 + _RING_SIZE * 4;   // 4 × Int32 control + Float32 ring

let _ctx         = null;   // AudioContext
let _node        = null;   // AudioWorkletNode
let _sab         = null;   // SharedArrayBuffer
let _ctrl        = null;   // Int32Array over _sab[0..15]
let _ringData    = null;   // Float32Array view of ring audio data (sab byte 16+)
let _worker      = null;   // fetch Web Worker
let _fid         = '';
let _srcSr       = 192000;
let _totalFrames = 0;
let _rafId       = null;

// ── Rate slider — continuous logarithmic mapping ────────────────
// Slider range 0–100, logarithmically mapped to 1/32×–1×.
// Snap points every 20 units (1/32, 1/16, 1/8, 1/4, 1/2, 1) with
// magnetic attraction within ±SNAP_THRESH of each snap value.
const _SNAP_VALUES = [0, 20, 40, 60, 80, 100];
const _SNAP_LABELS = ['1/32×', '1/16×', '1/8×', '1/4×', '1/2×', '1×'];
const _SNAP_THRESH = 3;   // slider units — within this, we snap to the stop

function _sliderToRate(v) {
  // rate = 2^(v/100 * 5 − 5)  →  v=0→1/32, v=20→1/16 … v=100→1×
  return Math.pow(2, (+v) / 100 * 5 - 5);
}

function _rateLabel(v) {
  // Exact snap point?
  const idx = _SNAP_VALUES.findIndex(s => Math.abs(v - s) < 0.5);
  if (idx >= 0) return _SNAP_LABELS[idx];
  // Arbitrary value: show as ×N with 2 significant figures
  const rate = _sliderToRate(v);
  return `×${parseFloat(rate.toPrecision(2))}`;
}

// ── Public ───────────────────────────────────────────────────────

function audioInit(fid, srcSr, totalFrames) {
  _audioTeardown();
  _fid         = fid;
  _srcSr       = srcSr;
  _totalFrames = totalFrames;
  S.playheadTime = 0;
  S.isPlaying    = false;
  updatePlayButton();
}

async function audioPlayPause() {
  if (S.isPlaying) {
    _audioPause();
  } else {
    await _audioPlay();
  }
}

let _seekInProgress = false;
let _seekQueued     = null;   // most-recent target queued while a seek is in-flight

// scrub=true uses a shorter initial buffer (0.25 s) so playback resumes faster
// during scrubbing; false uses the full 1 s buffer for clean initial play.
async function audioSeek(t, scrub = false) {
  t = Math.max(0, Math.min(S.duration, t));
  S.playheadTime = t;

  // If a seek is already in-flight, record the latest target so it can be
  // picked up when the current one finishes — avoids overlapping async chains.
  if (_seekInProgress) {
    _seekQueued = t;
    scheduleRender();
    return;
  }

  if (S.isPlaying && _ctrl) {
    _seekInProgress = true;
    try {
      const frame = Math.round(t * _srcSr);
      Atomics.store(_ctrl, 2, 0);                                     // pause
      Atomics.store(_ctrl, 0, frame);
      Atomics.store(_ctrl, 1, frame);
      if (_node)   _node.port.postMessage({ type: 'seek', frame });
      if (_worker) _worker.postMessage({ type: 'seek', frame });
      // Scrub seeks use a shorter prefetch (0.25 s) so playback resumes faster.
      await _waitForBuffer(frame, scrub ? 48_000 : 192_000);

      // A newer scrub target may have arrived while we were waiting.
      if (_seekQueued !== null) {
        const next = _seekQueued;
        _seekQueued = null;
        _seekInProgress = false;
        await audioSeek(next, scrub);   // tail-recurse to the latest position
        return;
      }
      Atomics.store(_ctrl, 2, 1);                                     // resume
    } finally {
      _seekInProgress = false;
      _seekQueued = null;
    }
  }
  scheduleRender();
}

function audioSetRate(rate) {
  if (_ctrl) Atomics.store(_ctrl, 3, Math.max(1, Math.round(rate * 1000)));
}

function updatePlayButton() {
  const btn = document.getElementById('btn-play');
  if (!btn) return;
  btn.textContent = S.isPlaying ? '⏸' : '▶';
  btn.title       = S.isPlaying ? 'Pause (Space)' : 'Play (Space)';
}

// ── Ring-buffer accessors (used by render.js for local PSD) ──────────────────

// Source sample rate — exposed so render.js doesn't need to re-derive it.
function audioSrcSr() { return _srcSr; }

// Copy `count` consecutive mono samples starting at `startFrame` out of the
// SAB ring buffer.  Returns a Float32Array, or null if the range isn't currently
// held in the buffer (too old / not yet fetched).
function audioGetFrames(startFrame, count) {
  if (!_ctrl || !_ringData) return null;
  if (startFrame < 0) { count += startFrame; startFrame = 0; }
  if (count <= 0) return null;
  const wf = Atomics.load(_ctrl, 1);          // how far worker has fetched
  if (startFrame < wf - _RING_SIZE) return null;   // evicted
  if (startFrame + count > wf)      return null;   // not yet written
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++)
    out[i] = _ringData[(startFrame + i) % _RING_SIZE];
  return out;
}

// ── Internal ─────────────────────────────────────────────────────

function _audioPause() {
  if (!S.isPlaying) return;
  S.isPlaying = false;
  if (_ctrl) Atomics.store(_ctrl, 2, 0);
  if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
  updatePlayButton();
  scheduleRender();
}

let _playPending = false;   // guard against double-tap while awaiting buffer

async function _audioPlay() {
  if (S.isPlaying || _playPending) return;
  _playPending = true;
  try {
    // Lazy-create or resume the AudioContext (browser policy: must be triggered
    // by a user gesture, so we can't create it in audioInit).
    if (!_ctx || _ctx.state === 'closed') {
      await _initContext();
    } else if (_ctx.state === 'suspended') {
      await _ctx.resume();
    }

    const startFrame = Math.round(S.playheadTime * _srcSr);
    Atomics.store(_ctrl, 0, startFrame);
    Atomics.store(_ctrl, 1, startFrame);

    // Tell the worklet exactly where to start reading (its internal _pos starts
    // at 0 and is never reset by the SAB control words — only port messages work).
    if (_node)   _node.port.postMessage({ type: 'seek', frame: startFrame });
    if (_worker) _worker.postMessage({ type: 'seek', frame: startFrame });

    // Wait until worker has prefetched a small initial buffer before enabling
    // the worklet — prevents the hard underrun click at the very first block.
    await _waitForBuffer(startFrame);

    Atomics.store(_ctrl, 2, 1);   // start playing now that data is ready
    S.isPlaying = true;
    updatePlayButton();
    _startRAF();
    scheduleRender();
  } catch(e) {
    console.error('Audio init failed:', e);
    S.isPlaying = false;
    updatePlayButton();
    const sb = document.getElementById('status-bar');
    if (sb) sb.textContent = typeof SharedArrayBuffer === 'undefined'
      ? 'Audio unavailable: browser requires a secure cross-origin context. Try reloading, or use Chrome/Firefox.'
      : `Audio error: ${e.message}`;
  } finally {
    _playPending = false;
  }
}

async function _initContext() {
  if (typeof SharedArrayBuffer === 'undefined')
    throw new Error('SharedArrayBuffer not available — page must be cross-origin-isolated (COOP + COEP headers).');
  if (_ctx && _ctx.state !== 'closed') await _ctx.close();
  _ctx  = new AudioContext();
  _sab      = new SharedArrayBuffer(_SAB_BYTES);
  _ctrl     = new Int32Array(_sab, 0, 4);
  _ringData = new Float32Array(_sab, 16, _RING_SIZE);

  Atomics.store(_ctrl, 2, 0);    // paused
  // Default rate: sync from slider if it exists, else 1/16×
  const slider = document.getElementById('slider-rate');
  const rate   = slider ? _sliderToRate(+slider.value) : _sliderToRate(20);
  Atomics.store(_ctrl, 3, Math.max(1, Math.round(rate * 1000)));

  await _ctx.audioWorklet.addModule('/static/audio-worklet.js');
  _node = new AudioWorkletNode(_ctx, 'bat-player', {
    processorOptions: { sab: _sab, srcSr: _srcSr },
    outputChannelCount: [2],
  });
  _node.connect(_ctx.destination);

  _worker = new Worker('/static/audio-worker.js');
  _worker.postMessage({
    type:        'init',
    sab:         _sab,
    fid:         _fid,
    totalFrames: _totalFrames,
    startFrame:  Math.round(S.playheadTime * _srcSr),
  });
}

function _audioTeardown() {
  _audioPause();
  S.isPlaying = false;
  if (_worker) { _worker.postMessage({ type: 'stop' }); _worker = null; }
  if (_node)   { _node.disconnect(); _node = null; }
  if (_ctx && _ctx.state !== 'closed') { _ctx.close(); _ctx = null; }
  _ctrl = null; _sab = null; _ringData = null;
}

// ── Playhead RAF loop ─────────────────────────────────────────────
// Reads the worklet's readFrame Atomic each animation frame and advances
// S.playheadTime.  Viewport scrolling is left entirely to the user.

function _startRAF() {
  if (_rafId) cancelAnimationFrame(_rafId);
  function tick() {
    if (!S.isPlaying || !_ctrl) return;
    const frame = Atomics.load(_ctrl, 0);
    const t     = frame / _srcSr;
    if (t >= S.duration || frame >= _totalFrames) {
      S.playheadTime = S.duration;
      _audioPause();
      scheduleRender();
      return;
    }
    // Don't overwrite playheadTime while the user is dragging the handle.
    if (!S.isDraggingPlayhead) S.playheadTime = t;

    // Auto-scroll: once the playhead reaches the centre of the visible window,
    // scroll the window so the playhead stays centred.  While the playhead is
    // still left of centre the window stays fixed and the playhead advances
    // toward the centre naturally.
    if (S.followPlayhead && !S.isDraggingPlayhead) {
      const centre = S.viewStart + S.viewDur / 2;
      if (t >= centre) {
        S.viewStart = Math.max(0, Math.min(S.duration - S.viewDur, t - S.viewDur / 2));
      }
    }

    scheduleRender();
    _rafId = requestAnimationFrame(tick);
  }
  _rafId = requestAnimationFrame(tick);
}

// ── Helpers ───────────────────────────────────────────────────────

// Wait until the fetch worker has prefetched at least framesNeeded ahead of
// startFrame, or until the deadline — whichever comes first.
async function _waitForBuffer(startFrame, framesNeeded = 192_000) {
  const deadline = Date.now() + 2000;
  while (_ctrl && Date.now() < deadline) {
    if (Atomics.load(_ctrl, 1) - startFrame >= framesNeeded) return;
    await new Promise(r => setTimeout(r, 16));
  }
}

// ── Rate slider setup ─────────────────────────────────────────────
// Called once from ui.js init() after the DOM is ready.

function _initRateSlider() {
  const slider = document.getElementById('slider-rate');
  const label  = document.getElementById('rate-val');
  if (!slider || !label) return;
  // Default: 1/16× = slider value 20
  slider.value      = '20';
  label.textContent = '1/16×';
  if (typeof updateTrack === 'function') updateTrack(slider);

  slider.addEventListener('input', () => {
    let v = +slider.value;
    // Magnetic snap: if within ±SNAP_THRESH of a stop, jump to it
    const nearest = _SNAP_VALUES.reduce((a, b) =>
      Math.abs(v - a) <= Math.abs(v - b) ? a : b);
    if (Math.abs(v - nearest) <= _SNAP_THRESH) {
      v = nearest;
      slider.value = v;   // visually snap the thumb
    }
    label.textContent = _rateLabel(v);
    if (typeof updateTrack === 'function') updateTrack(slider);
    audioSetRate(_sliderToRate(v));
  });
}
