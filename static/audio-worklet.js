// audio-worklet.js — variable-rate resampler for bat recording playback
//
// SharedArrayBuffer layout (passed via processorOptions.sab):
//   Int32[0]  readFrame   — absolute source frame the worklet has consumed (written here)
//   Int32[1]  writeFrame  — absolute source frame the fetch worker has filled up to
//   Int32[2]  playing     — 1 = running, 0 = paused/stopped
//   Int32[3]  rateX1000   — playback rate × 1000  (e.g. 63 ≈ 1/16×)
//   Float32 audio data starts at byte offset 16

const RING_SIZE = 1_920_000;   // 10 s at 192 kHz

class BatPlayerProcessor extends AudioWorkletProcessor {
  constructor (options) {
    super();
    const { sab, srcSr } = options.processorOptions;
    this._ctrl  = new Int32Array(sab, 0, 4);
    this._data  = new Float32Array(sab, 16, RING_SIZE);
    this._srcSr = srcSr;
    this._pos        = 0.0;   // fractional absolute source frame
    this._last       = 0.0;   // last valid sample (held during underrun / fade-out)
    this._gain       = 0.0;   // output envelope: 0 = silent, 1 = full; ramped, never jumps
    this._xfadePos   = 0.0;   // crossfade-out position (old loop-end position)
    this._xfadeN     = 0;     // remaining crossfade output samples (0 = inactive)

    // Main thread sends { type:'seek', frame } to reposition _pos.
    // 'seek'      — full seek: reset gain to 0 for a clean fade-in from silence.
    // 'loop_seek' — seamless loop: save old position for a short crossfade, reset
    //               only _pos; keep _gain so there is no audible fade gap.
    this.port.onmessage = ({ data }) => {
      if (data.type === 'seek') {
        this._pos    = data.frame;
        this._gain   = 0.0;
        this._last   = 0.0;
        this._xfadeN = 0;
      } else if (data.type === 'loop_seek') {
        this._xfadePos = this._pos;    // start crossfade from old (loop-end) position
        this._xfadeN   = 256;          // ~5 ms at 48 kHz — smooths waveform discontinuity
        this._pos      = data.frame;   // jump to loop start; keep _gain unchanged
      }
    };
  }

  process (_inputs, outputs) {
    const ch0 = outputs[0][0];   // left
    const ch1 = outputs[0][1];   // right (may be undefined for mono output)
    const n   = ch0.length;      // typically 128 samples per block

    const playing    = !!Atomics.load(this._ctrl, 2);
    const writeFrame =   Atomics.load(this._ctrl, 1);

    // Fast path: already silent and stopped — no work to do.
    if (!playing && this._gain <= 0) {
      ch0.fill(0);
      if (ch1) ch1.fill(0);
      Atomics.store(this._ctrl, 0, Math.floor(this._pos));
      return true;
    }

    // 20 ms linear fade: 1 / (sampleRate × 0.020) gain change per sample.
    // At 48 kHz that is 1/960 ≈ 0.00104 — reaches full silence in exactly 20 ms.
    const FADE_STEP = 1.0 / (sampleRate * 0.020);

    const rate = Atomics.load(this._ctrl, 3) / 1000;
    // step: source frames consumed per output sample
    const step = (this._srcSr / sampleRate) * rate;
    let   pos  = this._pos;

    for (let i = 0; i < n; i++) {
      const f0      = Math.floor(pos);
      const hasData = (writeFrame - f0) >= 2;

      // Target gain: 1 when actively playing with buffered data, 0 otherwise
      // (covers stop, pause, and buffer underrun with the same linear ramp).
      const target = (playing && hasData) ? 1.0 : 0.0;
      if (this._gain < target) this._gain = Math.min(1.0, this._gain + FADE_STEP);
      else if (this._gain > target) this._gain = Math.max(0.0, this._gain - FADE_STEP);

      if (!playing) {
        // Fading out to a stop: hold the last sample, do NOT advance pos so
        // the next play resumes exactly where we paused.
        ch0[i] = this._last * this._gain;
        if (ch1) ch1[i] = this._last * this._gain;
        continue;
      }

      if (!hasData) {
        // Buffer underrun: hold last sample and DO NOT advance pos.
        // Keeping pos frozen means: (a) ctrl[0] stays put so the fetch worker
        // knows exactly where to write next, and (b) when data arrives the first
        // ring-buffer read continues from the same position as this._last, so
        // there is no content discontinuity at recovery (only the smooth gain ramp).
        ch0[i] = this._last * this._gain;
        if (ch1) ch1[i] = this._last * this._gain;
        continue;
      }

      // Normal playback: linear interpolation between adjacent source samples.
      const frac = pos - f0;
      const i0   = f0 % RING_SIZE;
      const i1   = (f0 + 1) % RING_SIZE;
      let   s    = this._data[i0] + frac * (this._data[i1] - this._data[i0]);

      // Crossfade: blend new-position sample with old loop-end sample to prevent
      // a click at the loop boundary (waveform discontinuity).
      if (this._xfadeN > 0) {
        const xf0  = Math.floor(this._xfadePos);
        const xfrc = this._xfadePos - xf0;
        const xi0  = xf0 % RING_SIZE;
        const xi1  = (xf0 + 1) % RING_SIZE;
        const sOld = this._data[xi0] + xfrc * (this._data[xi1] - this._data[xi0]);
        const t    = 1 - this._xfadeN / 256;   // 0→1 as crossfade progresses
        s = s * t + sOld * (1 - t);
        this._xfadePos += step;
        this._xfadeN--;
      }

      this._last = s;
      ch0[i]     = s * this._gain;
      if (ch1) ch1[i] = s * this._gain;
      pos += step;
    }

    this._pos = pos;
    Atomics.store(this._ctrl, 0, Math.floor(pos));   // tell fetch worker where we are
    return true;
  }
}

registerProcessor('bat-player', BatPlayerProcessor);
