import os

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
AUDIO_FILE    = "2025-05-28 1942 bats on campbell 1.flac"   # default; override with AUDIO_FILE or AUDIO_DIR env var

# ── Zoom-level tile pyramid ───────────────────────────────────────
# Each zoom level maps a tile-duration in seconds.  All levels output
# the same TILE_W × TILE_H pixels so the client sees seamless tiles.
ZOOM_LEVELS  = {0: 40.0, 1: 20.0, 2: 10.0, 3: 5.0, 4: 2.5, 5: 1.0}
ZOOM_DEFAULT = 3
TILE_DURATION = 5.0        # seconds per tile
TILE_W        = 1500       # px
TILE_H        = 400        # px
FREQ_LOW      = 13_000     # Hz
FREQ_HIGH     = 96_000     # Hz
FREQ_LOW_K    = FREQ_LOW  / 1000
FREQ_HIGH_K   = FREQ_HIGH / 1000

# Display spectrogram (coarser = faster tiles)
D_NPERSEG     = 1024
D_NOVERLAP    = 768

# PSD sidebar: Gaussian smoothing (in frequency bins) applied to the power
# spectrum before display.  ~1 bin ≈ doubling the effective resolution — a
# gentle smooth that keeps real structure.
PSD_SMOOTH_SIGMA = 1.0

# PSD display scale: the curve is scaled against the file-wide global minimum
# (→ 0) and 99.5th-percentile dB (→ full width); values above the 99.5th are
# not clipped.  Bump to recompute the cached scale (psd_p01 = global min,
# psd_p99 = 99.5th pct) without regenerating tiles.
PSD_SCALE_VERSION = 4

# Per-call AR feature version.  Bump to force the background backfill to
# recompute ar1/ar2/ar1c (e.g. when the fit method changes).  v2 = high-passed
# into the bat band before fitting.
AR_FEATURE_VERSION = 2

# Detection (fine)
A_NPERSEG     = 2048
A_NOVERLAP    = 1792
THRESH_SIGMA  = 3.0
MIN_CALL      = 0.0015     # s
MAX_CALL      = 0.120      # s
MERGE_GAP     = 0.003      # s  — only bridge dilation artefacts, not real inter-call gaps

# BatDetect2 settings (used when the package is installed)
BD2_THRESH    = 0.10       # detection-probability threshold (0–1)
BD2_CHUNK_S   = 30.0       # audio chunk fed to BD2 at once
BD2_OVERLAP_S = 0.5        # overlap between chunks to avoid edge misses


# Cache: detection results are saved next to the audio file so re-runs
# skip the ~7-minute BatDetect2 pass.  Delete the .calls.json file (or
# pass --redetect on the command line) to force a fresh detection.
CACHE_FILE    = os.path.splitext(AUDIO_FILE)[0] + ".calls.json"
CHUNK_SECS    = 10.0

# Maximum number of file-processing jobs that may run simultaneously.
# Each job is one heavy operation: json cache load (~900 MB peak for the 192 kHz
# file) or BatDetect2 detection (~1+ GB).  Set to 1 on a 2 GB server.
PROCESSING_WORKERS = 1

# All generated data (tiles + call JSON) lives under generated/<stem>/ next to
# the audio files.  See gen_paths.py for the full layout.
GENERATED_DIR = "generated"

TILE_NORM_VERSION = 10     # bump to force regeneration when norm strategy changes

# Map from audio file stem to human-readable recording location.
LOCATION_MAP = {
    "2025-05-28 1942 bats on campbell 1":
        "Campbell Ave Bridge, Rillito River, Tucson AZ",
    "2025-06-06-1912-bats-192khz":
        "Campbell Ave Bridge, Rillito River, Tucson AZ",
    "2025-09-25-1753-bracken-san-antonio-side-batsbatsbats":
        "Bracken Bat Cave, San Antonio TX",
    "2025-09-20-1732-bracken-san-antonio-back-bats":
        "Bracken Bat Cave, San Antonio TX",
}
