import os

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
AUDIO_FILE    = "2025-05-28 1942 bats on campbell 1.flac"
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

TILE_NORM_VERSION = 9      # bump to force regeneration when norm strategy changes
