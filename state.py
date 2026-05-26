import threading
from matplotlib.cm import get_cmap
from flask import Flask

from config import TILE_NORM_VERSION  # noqa: F401 — imported so tiles.py can use it via state

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
app          = Flask(__name__)
audio_lock   = threading.Lock()
audio_fh     = None
finfo        = {}
all_calls    = []
calls_ready  = threading.Event()
progress     = {"done": 0, "total": 1, "status": "starting"}
tile_cache   = {}          # in-memory: idx → PNG bytes (no eviction limit)
tile_lock    = threading.Lock()
TILE_DIR     = ""          # set in startup() — directory for on-disk PNG cache
_inferno     = get_cmap("inferno")

# Global spectrogram normalization — computed once from a sample of tiles so
# all tiles share the same dB scale and brightness is consistent across boundaries.

# Set by _init_tile_norm() at startup; used by make_tile() for every tile.
_global_vmin = -100.0
_global_vmax =  -30.0

# Per-frequency stats for flat tiles: shape (n_freq,) matching the display STFT
# frequency bins inside [FREQ_LOW, FREQ_HIGH].  None until _init_tile_norm() runs.
# Using global (time-invariant) stats rather than per-tile keeps tile boundaries seamless.
_global_vmin_f = None   # per-bin 2nd-percentile  (noise floor at each frequency)
_global_vmax_f = None   # per-bin 99.9th-percentile (signal ceiling at each frequency)

# Separate cache + lock for RGBA mask tiles (call-isolation overlay).
mask_tile_cache = {}
mask_tile_lock  = threading.Lock()

# Cache for frequency-compensated ("flat") tiles.
flat_tile_cache = {}
flat_tile_lock  = threading.Lock()

# Set by reset_and_switch() to ask the detection thread to abort early.
_stop_detection = threading.Event()
