// ─── State ───────────────────────────────────────────────────
const S = {
  viewStart: 0,
  viewDur: 30,      // seconds visible
  duration: 0,
  freqLow: 13,
  freqHigh: 96,
  tileDur: 5,
  nTiles: 0,
  calls: [],
  tileImgs: new Map(),   // idx → Image (may be loading)
  tileReady: new Map(),  // idx → bool
  selectedCall: null,
  hoveredCall: null,
  mouseX: -1,         // canvas-relative px; -1 = not over spectrogram
  mouseY: -1,
  isRuling: false,    // ruler rubber-band being drawn
  rulerFixed: false,  // ruler sticks after drag release
  rulerX0: 0, rulerY0: 0,
  rulerX1: 0, rulerY1: 0,
  colors: {},
  hiddenSpecies: new Set(),
  soloedSpecies: null,
  showContour: true,
  showBoxes: false,       // bounding boxes shown only on hover/select by default
  contourAlpha: 1.0,      // default contour opacity (0–1)
  crossfade: 0,           // 0 = raw spectrogram, 1 = call-isolated view
  flatness:  0,           // 0 = raw, 1 = mic-response-flattened spectrogram
  logScale: 0,            // 0 = linear, 1 = fully logarithmic
  saturation: 1.0,        // CSS saturate() applied to spectrogram tiles (1=full color, 0=grey)
  pickRadius: 20,         // hover/click tolerance: max px from cursor to call bounding box
  minConf: 0,             // hide calls with confidence below this (0–1)
  ovStart: 0,             // overview transport: visible time window start (s)
  ovDur:   0,             // overview transport: visible duration (s; 0 until init)
  nyquist: 96,            // kHz — full scrollbar range (set from server)
  renderPending: false,
  tileWarpCache:      new Map(),  // `${idx}-${H}-${logScale}` → canvas
  maskTileWarpCache:  new Map(),
  flatTileWarpCache:  new Map(),
  maskTileImgs:  new Map(),
  maskTileReady: new Map(),
  flatTileImgs:  new Map(),
  flatTileReady: new Map(),
  classifier: 'v2',  // 'v1' (freq/dur/sweep) or 'v2' (+ bw/cf_frac)
  recordingStart: null,  // epoch ms; null until /api/info returns recording_start
};

// Fixed freq range of the server-rendered tile images (kHz)
let TILE_FREQ_LOW = 13, TILE_FREQ_HIGH = 96;

// Per-frame log-warp budget: cap how many tiles can be warped in a single
// render() call so logScale slider changes don't freeze the main thread.
// Tiles over budget show a linear-crop fallback until the next frame catches up.
let _logWarpBudget = 0;
const LOG_WARP_PER_FRAME = 8;

// ─── Canvas refs ─────────────────────────────────────────────
const canvasWrap = document.getElementById('canvas-wrap');
const canvas     = document.getElementById('mainCanvas');
const ctx        = canvas.getContext('2d');
const psdCanvas  = document.getElementById('psdCanvas');
const psdCtx     = psdCanvas.getContext('2d');
const ovCanvas   = document.getElementById('overviewCanvas');
const octx       = ovCanvas.getContext('2d');

// ─── PSD sidebar state ───────────────────────────────────────
let _psdData    = null;   // {freqs:[], powers:[]} from server
let _psdPending = false;
let _psdTimer   = null;
let _psdT0 = -1, _psdT1 = -1;   // last-fetched window

// ─── Pan-drag state (Cmd+drag) ───────────────────────────────
let _panDrag = false;
let _panX0 = 0, _panY0 = 0;
let _panVS0 = 0, _panVD0 = 0, _panFL0 = 0, _panFH0 = 0;

// ─── PSD transport drag state ────────────────────────────────
const PSD_EDGE_PX = 8;          // px hit zone for freq-edge handles
let _psdDrag   = null;          // null | 'top' | 'bot' | 'pan'
let _psdY0     = 0;             // clientY at drag start
let _psdFH0    = 0;             // freqHigh at drag start
let _psdFL0    = 0;             // freqLow  at drag start
let _psdHoverY = null;          // mouse Y over psdCanvas (null = not hovering)
// PSD viewport — independent zoom, like the time overview vs main canvas
let psdViewLow  = TILE_FREQ_LOW; // kHz — bottom of PSD canvas
let psdViewHigh = null;         // kHz — top of PSD canvas (null → S.nyquist)

const YAXIS_W  = 52;   // px for freq axis
const SPEC_H   = () => canvas.height;
const OV_H     = 64;

