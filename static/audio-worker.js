// audio-worker.js — prefetch worker for BatPlayerProcessor
// Runs as a regular Web Worker (not an AudioWorklet).
//
// Messages received:
//   { type: 'init', sab, fid, totalFrames, startFrame }
//   { type: 'seek', frame }
//   { type: 'stop' }
//
// The worker continually prefetches audio chunks from /api/audio_chunk and
// writes them into the SharedArrayBuffer ring buffer ahead of the worklet.

const RING_SIZE    = 1_920_000;   // must match audio-worklet.js (10 s at 192 kHz)
const CHUNK_FRAMES = 192_000;     // 1 s at 192 kHz per HTTP request (fewer round-trips)
const TARGET_AHEAD = 768_000;     // keep 4 s ahead — comfortable margin at 1× speed

let _ctrl        = null;    // Int32Array view of SAB header
let _data        = null;    // Float32Array view of SAB ring buffer
let _fid         = '';
let _totalFrames = 0;
let _writeFrame  = 0;
let _running     = false;
let _generation  = 0;       // bumped on seek/stop to cancel stale fetches

self.onmessage = ({ data: msg }) => {
  if (msg.type === 'init') {
    const { sab, fid, totalFrames, startFrame } = msg;
    _ctrl        = new Int32Array(sab, 0, 4);
    _data        = new Float32Array(sab, 16, RING_SIZE);
    _fid         = fid;
    _totalFrames = totalFrames;
    _running     = true;
    _seek(startFrame);
    _loop(_generation);
  } else if (msg.type === 'seek') {
    _seek(msg.frame);
    _loop(_generation);
  } else if (msg.type === 'loop_seek') {
    // Seamless loop: the ring already holds valid data at this frame, so we
    // must NOT reset ctrl[0]/ctrl[1] (that would cause an underrun).  Just
    // restart the fetch loop from the new position so future loops are ready.
    _generation++;
    _writeFrame = Math.max(0, Math.min(_totalFrames, msg.frame));
    _loop(_generation);
  } else if (msg.type === 'stop') {
    _running = false;
    _generation++;
  }
};

function _seek(frame) {
  _generation++;                                          // invalidate current loop
  _writeFrame = Math.max(0, Math.min(_totalFrames, frame));
  if (_ctrl) {
    Atomics.store(_ctrl, 0, _writeFrame);   // readFrame  = seek target
    Atomics.store(_ctrl, 1, _writeFrame);   // writeFrame = seek target
  }
}

async function _loop(gen) {
  while (_running && gen === _generation) {
    const readFrame = Atomics.load(_ctrl, 0);
    const ahead     = _writeFrame - readFrame;

    if (ahead >= TARGET_AHEAD || _writeFrame >= _totalFrames) {
      await _sleep(50);
      continue;
    }

    const n = Math.min(CHUNK_FRAMES, _totalFrames - _writeFrame);
    try {
      const res = await fetch(
        `/api/audio_chunk?f=${encodeURIComponent(_fid)}&frame=${_writeFrame}&n=${n}`
      );
      if (gen !== _generation) break;
      if (!res.ok) { await _sleep(200); continue; }
      const ab      = await res.arrayBuffer();
      if (gen !== _generation) break;
      const samples = new Float32Array(ab);
      // Write into ring buffer, handling wrap-around
      for (let i = 0; i < samples.length; i++) {
        _data[(_writeFrame + i) % RING_SIZE] = samples[i];
      }
      _writeFrame += samples.length;
      Atomics.store(_ctrl, 1, _writeFrame);
    } catch (_) {
      await _sleep(200);
    }
  }
}

function _sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
