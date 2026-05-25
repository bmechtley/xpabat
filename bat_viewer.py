#!/usr/bin/env python3
"""
Bat Spectrogram Viewer — interactive web UI
Run:  python3 bat_viewer.py
Open: http://localhost:5000
"""

import io, json, os, time, threading, warnings
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, send_file, render_template_string, request
from scipy import signal
from scipy.ndimage import label, binary_dilation, binary_erosion, gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
from matplotlib.cm import get_cmap
from PIL import Image

warnings.filterwarnings("ignore")

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

# ─────────────────────────────────────────────
# Species
# ─────────────────────────────────────────────
PROFILES = [
    # ── prior: relative weight for this site (bridge colony, Tucson AZ, late May)
    # Acoustic criteria alone are scored 0–1; the prior multiplies that score so
    # ties resolve in favour of the locally most likely species.  Values > 1 boost,
    # values < 1 suppress.  A species still needs ≥ 0.5 raw acoustic score to win.
    {
        "name": "Tadarida brasiliensis", "short": "TABR",
        "Fchar": (20, 27), "Fmin": (18, 25), "dur": (8, 25), "sweep": (0.1, 1.2),
        "bw": (1, 16), "cf_frac": (0.45, 1.0),   # narrow band, mostly CF
        "prior": 1.6,   # dominant bridge-colony bat in Tucson; very likely majority of calls
        "common": "Mexican Free-tailed Bat",
        "call_type": "Nearly constant-frequency (CF), extremely narrow bandwidth. Characteristic search-phase call at ~20–26 kHz. Molossid — not a vespertilionid.",
        "desc": (
            "Most abundant bat in North America and almost certainly the dominant species here. "
            "The Campbell Ave bridge hosts a large colony. Forms the largest mammal aggregations "
            "on earth (Bracken Cave, TX: ~15 million). Long narrow wings for fast, high flight. "
            "Narrow-band CF call is highly distinctive and rarely confused with other western NA species. "
            "Year-round resident in Tucson; colony emerges at dusk in a spectacular column."
        ),
        "habitat": "Open habitats — agricultural, suburban, over water. Roosts in vast cave colonies, buildings, and bridges. The Rillito corridor is prime foraging habitat.",
        "range": "Southern US through Central America and most of South America. Year-round resident in warmest parts of range including Tucson.",
        "ipi_ms": "50–80",
        "refs": [
            "Williams et al. (1973) Anim Behav 21:302–321",
            "Simmons & Stein (1980) J Comp Physiol 135:335–353",
            "O'Shea & Bogan (2003) Monitoring Trends in Bat Populations of the US and Territories, USGS",
        ],
    },
    {
        "name": "Eptesicus fuscus", "short": "EPFU",
        "Fchar": (24, 35), "Fmin": (20, 28), "dur": (8, 20), "sweep": (0.5, 3.5),
        "bw": (6, 22), "cf_frac": (0.10, 0.70),  # moderate FM sweep
        "prior": 1.0,
        "common": "Big Brown Bat",
        "call_type": "Shallow FM sweep with quasi-CF tail. Relatively low frequency for a vespertilionid; characteristic frequency slightly higher in SW populations (~25–33 kHz) than in the east.",
        "desc": (
            "Common, year-round urban bat in Tucson. Large body size (~15–20 g), slow powerful flight. "
            "Roosts in buildings and bridges year-round. Calls overlap in frequency with TABR "
            "but are broader-band and have a more pronounced FM component. "
            "Likely present in small numbers alongside the TABR colony."
        ),
        "habitat": "Highly adaptable — buildings, bridges, cave crevices; forages over water, open fields, forest edges, and suburban areas.",
        "range": "Across all of North America (except far north), the Caribbean, and parts of Central and South America.",
        "ipi_ms": "50–100",
        "refs": [
            "Fenton & Bell (1981) J Mammal 62:317–324",
            "Whitaker (2004) J Mammal 85:1–13",
            "Simmons (2005) Mammal Species of the World, 3rd ed.",
        ],
    },
    {
        "name": "Lasiurus cinereus", "short": "LACI",
        "Fchar": (16, 22), "Fmin": (13, 20), "dur": (10, 25), "sweep": (0.8, 4.0),
        "bw": (10, 42), "cf_frac": (0.10, 0.65),  # steep FM + low-CF tail; wide bandwidth
        "prior": 0.8,   # migratory, occasional in Tucson; possible in May
        "common": "Hoary Bat",
        "call_type": "Steep FM sweep ending in a prominent low-frequency CF tail (~16–19 kHz). Loudest calls of local vespertilionids; second harmonic often visible.",
        "desc": (
            "Largest bat native to North America (~26–35 g). Highly migratory — present in Tucson "
            "mainly during spring/fall migration and occasionally in summer. Solitary, roosts in "
            "tree foliage (cottonwoods along the Rillito are suitable). Very loud calls audible at "
            "the edge of human hearing. Strong second harmonic at ~35 kHz can confuse classifiers."
        ),
        "habitat": "Diverse habitats during migration; breeds in forest and edge habitats. Roosts in tree foliage, not structures. The Rillito riparian corridor is used during migration.",
        "range": "Breeds across most of North America; winters in south-central US, Mexico, Central America, and Hawaii.",
        "ipi_ms": "200–400",
        "refs": [
            "Betts (1998) J Mammal 79:1098–1105",
            "Cryan (2003) J Mammal 84:1020–1028",
            "Simmons (2005)",
        ],
    },
    {
        "name": "Lasiurus blossevillii", "short": "LBOS",
        "Fchar": (33, 48), "Fmin": (25, 38), "dur": (8, 18), "sweep": (1.5, 5.0),
        "bw": (8, 28), "cf_frac": (0.05, 0.45),  # moderate-steep FM
        "prior": 0.7,   # uncommon in Tucson; riparian areas possible
        "common": "Western Red Bat",
        "call_type": "Steep FM sweep at moderate-high frequency. Calls intermediate between EPFU and Myotis spp. in frequency and sweep rate.",
        "desc": (
            "Western counterpart of the Eastern Red Bat (L. borealis), now treated as a distinct "
            "species. Solitary, migratory, roosts in tree foliage — cottonwood/willow riparian "
            "corridors along the Rillito are suitable habitat. Sexually dimorphic brick-red to "
            "chestnut fur. Less common than TABR and EPFU in Tucson but present in small numbers."
        ),
        "habitat": "Wooded areas, riparian corridors, parks; roosts in foliage of deciduous trees. Rillito riparian zone is prime habitat.",
        "range": "Western North America from British Columbia south through Central America. Winters in coastal areas and Mexico.",
        "ipi_ms": "100–200",
        "refs": [
            "Valdez & Cryan (2009) J Mammal 90:1308–1320",
            "Hoofer et al. (2006) J Mammal 87:252–257",
            "Simmons (2005)",
        ],
    },
    {
        "name": "Antrozous pallidus", "short": "ANPA",
        "Fchar": (28, 50), "Fmin": (22, 40), "dur": (2, 10), "sweep": (2.0, 10.0),
        "bw": (4, 22), "cf_frac": (0.10, 0.70),  # short FM; wide variation across call types
        "prior": 0.8,   # present in Tucson but less likely at bridge emergence
        "common": "Pallid Bat",
        "call_type": "Short, steep FM pulses; relatively quiet. Primarily a gleaning bat — echolocation used mainly for obstacle avoidance, not prey detection. Wide frequency range reflects variation across call types.",
        "desc": (
            "Large-eared, pale desert bat (~14–23 g). Unique among North American bats in regularly "
            "taking prey from the ground (scorpions, beetles, crickets). Emits relatively quiet, "
            "short-duration calls. Common in Tucson but typically forages in rocky/desert habitat "
            "rather than the open airspace above bridges. Also produces distinctive social calls. "
            "Immune to scorpion venom."
        ),
        "habitat": "Arid and semi-arid scrub, desert, open woodland; roosts in rock crevices, caves, buildings. Less associated with bridge roosts than TABR.",
        "range": "Arid western North America — BC/AB south through Mexico; disjunct population in Cuba.",
        "ipi_ms": "60–150",
        "refs": [
            "Bell (1982) Behav Ecol Sociobiol 10:1–6",
            "Hermanson & O'Shea (1983) Mammalian Species 213:1–8",
            "Simmons (2005)",
        ],
    },
    {
        "name": "Myotis velifer", "short": "MYVE",
        "Fchar": (26, 38), "Fmin": (20, 30), "dur": (4, 12), "sweep": (1.5, 6.0),
        "bw": (8, 26), "cf_frac": (0.03, 0.30),  # moderate FM, low CF fraction
        "prior": 1.1,   # very common in southern AZ; often roosts near TABR
        "common": "Cave Myotis",
        "call_type": "Moderate-steep FM sweep at lower frequency than other Myotis. Largest Myotis in the southwest; calls are longer and lower than the small/medium Myotis clusters.",
        "desc": (
            "Most common Myotis in southern Arizona (~7–9 g). Often roosts in the same structures "
            "as TABR. Emits lower-frequency calls than most Myotis (~28–38 kHz) which helps "
            "acoustic separation, though overlap with EPFU and LACI is possible. "
            "Forages over open desert, riparian areas, and over water. "
            "The Rillito corridor supports good numbers of this species."
        ),
        "habitat": "Cave and mine roosts; also bridges and buildings. Forages over open desert and riparian habitat. Common along major river corridors in the Sonoran Desert.",
        "range": "Southwestern US (TX, NM, AZ, southern NV/CA) south through Mexico and Central America.",
        "ipi_ms": "60–140",
        "refs": [
            "Watkins (1977) Mammalian Species 80:1–6",
            "Fenton & Bell (1981) J Mammal 62:317–324",
            "O'Shea & Bogan (2003) USGS Monitoring Report",
        ],
    },
    {
        "name": "Myotis (medium)", "short": "MYYU",
        "Fchar": (38, 55), "Fmin": (28, 45), "dur": (2, 7), "sweep": (3.0, 12.0),
        "bw": (12, 38), "cf_frac": (0.02, 0.22),  # steep broadband FM, very little CF
        "prior": 1.1,   # Yuma Myotis very common near Rillito water
        "common": "Medium Myotis (M. yumanensis group)",
        "call_type": "Steep broadband FM sweep, moderate-high characteristic frequency. Classic narrow-bandwidth FM call.",
        "desc": (
            "Heuristic cluster covering medium-sized western Myotis — most likely Myotis yumanensis "
            "(Yuma Myotis) in this region. The Yuma Myotis is strongly associated with water "
            "and is one of the most abundant bats along desert river systems including the Rillito. "
            "Body weight 4–7 g. Forages low over water. Often emerges very early at dusk."
        ),
        "habitat": "Strongly associated with water — streams, ponds, lakes. Roosts in buildings, bridges, mines, and caves near water. Common along the Rillito/Santa Cruz system.",
        "range": "Western North America from BC south through Mexico. M. yumanensis is one of the most common bats along Sonoran Desert waterways.",
        "ipi_ms": "50–120",
        "refs": [
            "Fenton & Bell (1981) J Mammal 62:317–324",
            "Hoffmeister (1986) Mammals of Arizona, Univ. of Arizona Press",
            "Simmons (2005)",
        ],
    },
    {
        "name": "Myotis (small)", "short": "MYCA",
        "Fchar": (50, 68), "Fmin": (35, 55), "dur": (1.5, 6), "sweep": (5.0, 20.0),
        "bw": (15, 45), "cf_frac": (0.02, 0.18),  # very steep broadband FM
        "prior": 1.0,
        "common": "Small Myotis (M. californicus / M. ciliolabrum group)",
        "call_type": "Very steep broadband FM sweep, high frequency, very short duration. Highest-frequency Myotis group in western NA.",
        "desc": (
            "Heuristic cluster covering small-bodied western Myotis — California Myotis (M. californicus) "
            "and Western Small-footed Myotis (M. ciliolabrum). Body weight 3–5 g. "
            "Both species present in the Tucson area; M. californicus is common in desert scrub and "
            "rocky areas, M. ciliolabrum favors open terrain. Notoriously difficult to separate "
            "acoustically from each other and from Parastrellus hesperus."
        ),
        "habitat": "Desert, scrub, open and rocky areas; roosts in rock crevices, cliff faces, occasionally buildings.",
        "range": "Western North America — BC south through Mexico.",
        "ipi_ms": "40–100",
        "refs": [
            "Fenton & Bell (1981) J Mammal 62:317–324",
            "Hoffmeister (1986) Mammals of Arizona",
            "Simmons (2005)",
        ],
    },
    {
        "name": "Parastrellus hesperus", "short": "PEHE",
        "Fchar": (52, 72), "Fmin": (40, 60), "dur": (2, 5), "sweep": (5.0, 18.0),
        "bw": (8, 32), "cf_frac": (0.04, 0.35),  # steep FM; slightly less steep than small Myotis
        "prior": 1.0,   # very common in Tucson; rocky/urban areas
        "common": "Canyon Bat (Western Pipistrelle)",
        "call_type": "Short steep FM sweep at very high frequency (~55–70 kHz). Among the highest-frequency bats in the region. Acoustically very similar to small Myotis.",
        "desc": (
            "Smallest bat in North America (~3–6 g). Formerly called Western Pipistrelle, renamed "
            "Canyon Bat after genetic revision (now Parastrellus, not Pipistrellus). "
            "Extremely common in Tucson — one of the first bats to emerge at dusk, often seen "
            "flying in daylight. Very high-frequency calls overlap with small Myotis but "
            "PEHE tends to have a more sinusoidal, lower-amplitude call and a characteristic "
            "two-part pulse shape. Common along the Rillito riparian corridor."
        ),
        "habitat": "Rocky desert, canyons, riparian areas, urban parks. One of the most common bats in the Sonoran Desert. Roosts in rock crevices, cliff faces, buildings.",
        "range": "Arid western North America — WA south through central Mexico.",
        "ipi_ms": "60–150",
        "refs": [
            "Czaplewski (1983) Mammalian Species 199:1–5",
            "Hoffmeister (1986) Mammals of Arizona",
            "Hoofer & Van Den Bussche (2003) J Mammal 84:698–707",
        ],
    },
]
COLORS = {
    "Tadarida brasiliensis":  "#59a14f",
    "Eptesicus fuscus":       "#4e79a7",
    "Lasiurus cinereus":      "#f28e2b",
    "Lasiurus blossevillii":  "#e15759",
    "Antrozous pallidus":     "#b07aa1",
    "Myotis velifer":         "#edc948",
    "Myotis (medium)":        "#76b7b2",
    "Myotis (small)":         "#ff9da7",
    "Parastrellus hesperus":  "#bab0ac",
    "Unclassified":           "#888888",
}

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
TILE_NORM_VERSION = 9      # bump to force regeneration when norm strategy changes

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

# ─────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────
def score_v1(call, p):
    """v1: fraction of 4 basic criteria met (Fpeak, Fmin, dur, sweep)."""
    return sum([
        p["Fchar"][0] <= call["Fpeak"] <= p["Fchar"][1],
        p["Fmin"][0]  <= call["Fmin"]  <= p["Fmin"][1],
        p["dur"][0]   <= call["dur"]   <= p["dur"][1],
        p["sweep"][0] <= call["sweep"] <= p["sweep"][1],
    ]) / 4

def score_v2(call, p):
    """v2: fraction of 6 criteria met — adds bandwidth and CF fraction.

    bw (Fmax−Fmin) distinguishes narrow-band CF bats (TABR ~2–15 kHz) from
    broadband FM bats (Myotis ~15–40 kHz).
    cf_frac (fraction of contour frames within 2 kHz of median) distinguishes
    mostly-CF bats (TABR 0.45–1.0) from steep-FM bats (Myotis 0.02–0.25).
    Both features are computed in trim_call_contour() from the cleaned contour.
    """
    bw = call.get("bw", call["Fmax"] - call["Fmin"])
    cf = call.get("cf_frac", 0.5)   # default neutral if missing
    return sum([
        p["Fchar"][0] <= call["Fpeak"] <= p["Fchar"][1],
        p["Fmin"][0]  <= call["Fmin"]  <= p["Fmin"][1],
        p["dur"][0]   <= call["dur"]   <= p["dur"][1],
        p["sweep"][0] <= call["sweep"] <= p["sweep"][1],
        p["bw"][0]    <= bw           <= p["bw"][1],
        p["cf_frac"][0] <= cf         <= p["cf_frac"][1],
    ]) / 6

def _best_species(raw_scores):
    """Given {name: raw_score} return (best_name, raw_score) or Unclassified."""
    weighted = {n: s * next(p.get("prior", 1.0) for p in PROFILES if p["name"] == n)
                for n, s in raw_scores.items()}
    best = max(weighted, key=weighted.get)
    return (best, round(raw_scores[best], 2)) if raw_scores[best] >= 0.5 else ("Unclassified", 0.0)

def classify_v1(call):
    raw = {p["name"]: score_v1(call, p) for p in PROFILES}
    return _best_species(raw)

def classify_v2(call):
    raw = {p["name"]: score_v2(call, p) for p in PROFILES}
    return _best_species(raw)

# Keep classify() as an alias for v2 (used during fresh detection)
def classify(call):
    return classify_v2(call)

def merge(calls):
    if not calls:
        return []
    calls.sort(key=lambda c: c["t0"])
    out = [dict(calls[0])]
    for c in calls[1:]:
        p = out[-1]
        if c["t0"] - p["t1"] < MERGE_GAP:
            p["t1"]      = max(p["t1"], c["t1"])
            p["dur"]     = (p["t1"] - p["t0"]) * 1000
            p["Fmax"]    = max(p["Fmax"],  c["Fmax"])
            p["Fmin"]    = min(p["Fmin"],  c["Fmin"])
            p["Fpeak"]   = (p["Fpeak"] + c["Fpeak"]) / 2
            p["sweep"]   = (p["sweep"] + c["sweep"]) / 2
            p["contour"].extend(c["contour"])
            # Contour search windows extend ±1ms past t0/t1, so the seam of
            # two merged calls can have time-reversed points.  Sorting by time
            # interleaves the two frequency tracks, creating zig-zag loops.
            # Instead keep only time-forward points (strict monotone filter):
            # the first contour's points take priority at the overlap seam.
            mono = [p["contour"][0]]
            for pt in p["contour"][1:]:
                if pt[0] > mono[-1][0]:
                    mono.append(pt)
            p["contour"] = mono
        else:
            out.append(dict(c))
    return out

def track_fundamental(seg, seg_f, low_hz, high_hz, sr):
    """
    Extract a continuous frequency contour that tracks the call's fundamental,
    not its harmonics.

    Two-stage approach
    ------------------
    1.  Initialise by finding the best energy peak within BD2's predicted
        frequency range (averaged over the first few frames).
    2.  For each subsequent frame, only consider frequencies within
        MAX_JUMP_HZ of the previous frame.  Among those candidates pick the
        one with maximum energy — the hard gate rejects harmonic jumps
        (always ≥ one fundamental width away) while allowing even the fastest
        observed bat FM sweeps (~20 kHz/ms).

    MAX_JUMP is computed from the STFT hop time and the fastest known bat
    sweep rate so it adapts to whatever sample rate the recording uses.
    """
    n = seg.shape[1]
    if n == 0:
        return np.array([], dtype=float)

    hop_s        = (A_NPERSEG - A_NOVERLAP) / sr   # seconds per STFT frame
    max_jump_hz  = 20_000 * hop_s * 1000            # 20 kHz/ms × step_ms
    # Hard cap: harmonics are always ≥ one fundamental-width away (typically
    # 20–60 kHz jump).  Capping at 15 kHz keeps us below all real harmonics
    # while still allowing MYCA's steep sweep at high sample-rates.
    max_jump_hz  = min(max_jump_hz, 15_000)

    tracked = np.empty(n)

    # Initialise: averaged first 3 frames, prefer BD2's frequency range
    n_init   = min(3, n)
    init_pow = seg[:, :n_init].mean(axis=1)
    in_bd2   = (seg_f >= low_hz * 0.85) & (seg_f <= high_hz * 1.15)
    if in_bd2.any():
        masked     = np.where(in_bd2, init_pow, 0.0)
        tracked[0] = seg_f[masked.argmax()]
    else:
        tracked[0] = seg_f[init_pow.argmax()]

    for i in range(1, n):
        prev_f    = tracked[i - 1]
        reachable = np.abs(seg_f - prev_f) <= max_jump_hz
        if reachable.any():
            e          = seg[:, i] * reachable     # zero out unreachable bins
            tracked[i] = seg_f[e.argmax()]
        else:                                       # shouldn't happen; safe fallback
            tracked[i] = seg_f[seg[:, i].argmax()]

    return tracked


def run_detection():
    global all_calls
    sr, nf = finfo["sr"], finfo["nframes"]
    raw    = []

    # ── Try BatDetect2 first ──────────────────────────────────────
    use_bd2 = False
    try:
        import torch
        import batdetect2.api as bd2
        device = (torch.device("mps")
                  if torch.backends.mps.is_available()
                  else torch.device("cpu"))
        progress["status"] = f"Loading BatDetect2 on {str(device).upper()}…"
        bd2_model, bd2_params = bd2.load_model(device=device)
        use_bd2 = True
        detector_label = f"BatDetect2/{str(device).upper()}"
    except Exception as exc:
        print(f"BatDetect2 unavailable ({exc}), using energy-threshold detector")
        detector_label = "threshold"

    chunk_frames   = int(BD2_CHUNK_S   * sr) if use_bd2 else int(CHUNK_SECS * sr)
    overlap_frames = int(BD2_OVERLAP_S * sr) if use_bd2 else 0
    total_ch       = int(np.ceil(nf / chunk_frames))
    progress["total"] = total_ch
    offset = 0; chunk_num = 0

    dur_s = nf / sr
    print(f"\nDetection starting  [{detector_label}]")
    print(f"  Recording: {dur_s:.1f} s  |  chunks: {total_ch}  ({BD2_CHUNK_S if use_bd2 else CHUNK_SECS:.0f} s each)")
    if use_bd2:
        print(f"  Rough ETA: ~{dur_s/60*3:.0f} min on MPS  /  ~{dur_s/60*6:.0f} min on CPU")
    t_detect_start = time.time()

    while offset < nf:
        # Read chunk + trailing overlap (so calls at the boundary aren't cut)
        end = min(nf, offset + chunk_frames + overlap_frames)
        with audio_lock:
            audio_fh.seek(offset)
            audio = audio_fh.read(end - offset, dtype="float32", always_2d=True)
        mono            = audio.mean(axis=1)
        chunk_offset_s  = offset / sr

        # Compute spectrogram once — used for contour extraction in both paths
        f, t, Sxx = signal.spectrogram(
            mono, fs=sr, nperseg=A_NPERSEG, noverlap=A_NOVERLAP, window="hann")
        bm = (f >= FREQ_LOW) & (f <= FREQ_HIGH)
        fb = f[bm];  Sb = Sxx[bm, :]

        if use_bd2:
            # ── BatDetect2 detection ──────────────────────────────
            preds, _, _ = bd2.process_audio(
                mono, sr, model=bd2_model, config=bd2_params, device=device)

            for p in preds:
                if p["det_prob"] < BD2_THRESH:
                    continue
                # Discard calls that start inside the overlap tail
                # (they'll be picked up by the next chunk without the gap)
                t0_rel = p["start_time"]
                if t0_rel >= BD2_CHUNK_S and (offset + chunk_frames) < nf:
                    continue
                t1_rel = p["end_time"]
                dur_s  = t1_rel - t0_rel
                if not (MIN_CALL <= dur_s <= MAX_CALL):
                    continue

                t0_abs = chunk_offset_s + t0_rel
                t1_abs = chunk_offset_s + t1_rel

                # Extract frequency contour from our own spectrogram
                i0 = max(0,       np.searchsorted(t, t0_rel - 0.001))
                i1 = min(Sb.shape[1], np.searchsorted(t, t1_rel + 0.001))

                if i1 - i0 < 2:
                    Fmin_k = p["low_freq"]  / 1000
                    Fmax_k = p["high_freq"] / 1000
                    fpeak  = (Fmin_k + Fmax_k) / 2
                    swp    = 0.0
                    contour = [[t0_abs, fpeak], [t1_abs, fpeak]]
                else:
                    # Gate the search band to BD2's predicted range (±25%)
                    # so floor noise in adjacent bands can't contaminate the contour.
                    flo_hz  = max(FREQ_LOW  * 1000, p["low_freq"]  * 0.75)
                    fhi_hz  = min(FREQ_HIGH * 1000, p["high_freq"] * 1.25)
                    bm_seg  = (fb >= flo_hz) & (fb <= fhi_hz)
                    if not bm_seg.any():
                        bm_seg = np.ones(len(fb), dtype=bool)
                    seg_f   = fb[bm_seg]
                    seg     = Sb[bm_seg, :][:, i0:i1]
                    fc_t    = t[i0:i1] + chunk_offset_s

                    # Continuity-constrained tracking prevents harmonic jumps
                    fc_hz   = track_fundamental(seg, seg_f,
                                                p["low_freq"], p["high_freq"], sr)

                    Fmax_k  = fc_hz.max() / 1000
                    Fmin_k  = fc_hz.min() / 1000
                    fpeak   = seg_f[seg.mean(axis=1).argmax()] / 1000
                    tms     = np.linspace(0, dur_s * 1000, len(fc_hz))
                    swp     = (abs(np.polyfit(tms, fc_hz / 1000, 1)[0])
                               if len(fc_hz) > 2 else 0.0)
                    contour = [[float(ct), float(cf / 1000)]
                               for ct, cf in zip(fc_t, fc_hz)]

                raw.append({
                    "t0":       t0_abs,    "t1":    t1_abs,
                    "dur":      dur_s * 1000,
                    "Fmax":     Fmax_k,    "Fmin":  Fmin_k,  "Fpeak": fpeak,
                    "sweep":    swp,
                    "contour":  contour,
                    "det_prob": round(float(p["det_prob"]), 3),
                })

        else:
            # ── Fallback: energy-threshold detector ───────────────
            noise_pf = np.median(Sb, axis=1, keepdims=True)
            energy   = np.maximum(0, Sb - noise_pf).max(axis=0)
            act      = energy > np.median(energy) + THRESH_SIGMA * energy.std()
            act      = binary_dilation(binary_erosion(act, iterations=1), iterations=2)
            lbl, n   = label(act)

            for i in range(1, n + 1):
                idx  = np.where(lbl == i)[0]
                i0, i1 = idx[0], idx[-1]
                dur_s  = t[i1] - t[i0]
                if not (MIN_CALL <= dur_s <= MAX_CALL):
                    continue
                seg    = Sb[:, i0:i1+1]
                ms     = seg.mean(axis=1)
                fpeak  = fb[ms.argmax()] / 1000
                fc_t   = t[i0:i1+1] + chunk_offset_s
                # Use full band with continuity constraint (no BD2 range available)
                fc_hz  = track_fundamental(seg, fb, FREQ_LOW, FREQ_HIGH, sr)
                tms    = np.linspace(0, dur_s * 1000, len(fc_hz))
                swp    = (abs(np.polyfit(tms, fc_hz / 1000, 1)[0])
                          if len(fc_hz) > 2 else 0.0)
                raw.append({
                    "t0":       chunk_offset_s + t[i0],
                    "t1":       chunk_offset_s + t[i1],
                    "dur":      dur_s * 1000,
                    "Fmax":     fc_hz.max() / 1000,
                    "Fmin":     fc_hz.min() / 1000,
                    "Fpeak":    fpeak,
                    "sweep":    swp,
                    "contour":  [[float(ct), float(cf / 1000)]
                                 for ct, cf in zip(fc_t, fc_hz)],
                    "det_prob": 0.0,
                })

        offset     += chunk_frames
        chunk_num  += 1
        progress["done"]   = chunk_num
        progress["status"] = (f"Detecting ({detector_label})…"
                              f" {chunk_num}/{total_ch}")
        if chunk_num % 5 == 0 or chunk_num == total_ch:
            elapsed = time.time() - t_detect_start
            eta     = elapsed / chunk_num * (total_ch - chunk_num) if chunk_num > 0 else 0
            print(f"  chunk {chunk_num:3d}/{total_ch}  ({100*chunk_num//total_ch:3d}%)  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  calls so far: {len(raw)}", flush=True)

    merged = merge(raw)
    for c in merged:
        trim_call_contour(c)
    for idx, c in enumerate(merged):
        sp, conf    = classify(c)
        c["id"]     = idx
        c["species"] = sp
        c["conf"]   = round(conf, 2)
        c["color"]  = COLORS.get(sp, "#888888")
        c["short"]  = next((p["short"] for p in PROFILES if p["name"] == sp), "????")

    all_calls.extend(merged)
    progress["status"] = f"Done — {len(all_calls)} calls  [{detector_label}]"
    print(f"\nDetection done in {time.time() - t_detect_start:.0f} s  —  "
          f"{len(all_calls)} calls", flush=True)

    # ── Persist results to disk ───────────────────────────────────
    try:
        cache = {
            "version":       2,
            "audio_file":    AUDIO_FILE,
            "audio_mtime":   os.path.getmtime(AUDIO_FILE),
            "detector":      detector_label,
            "bd2_thresh":    BD2_THRESH,
            "calls":         all_calls,
        }
        with open(CACHE_FILE, "w") as fh:
            json.dump(cache, fh)
        print(f"Results cached → {CACHE_FILE}")
    except Exception as exc:
        print(f"Warning: could not write cache ({exc})")

    calls_ready.set()
    print(progress["status"])
    threading.Thread(target=_pregenerate_tiles, daemon=True).start()

# ─────────────────────────────────────────────
# Tile generation
# ─────────────────────────────────────────────
def _init_tile_norm():
    """Compute global vmin/vmax from a sample of tiles and purge stale cache.

    Global normalization means every tile uses the same dB scale so brightness
    is consistent across the recording.  We sample ~30 tiles spread across the
    file, compute per-tile 2nd and 99.9th percentiles (bat calls are ~1% of
    pixels, so 99.9th is needed to land inside call energy), then take the
    median for vmin and 75th-percentile for vmax.

    Flat tiles use per-frequency per-tile normalization (see make_flat_tile),
    so they don't need global stats computed here.
    """
    global _global_vmin, _global_vmax, _global_vmin_f, _global_vmax_f

    norm_path = os.path.join(TILE_DIR, "norm.json")

    # Check whether cached stats already match the current version
    if os.path.exists(norm_path):
        try:
            with open(norm_path) as fh:
                ndata = json.load(fh)
            if ndata.get("version") == TILE_NORM_VERSION:
                _global_vmin   = ndata["vmin"]
                _global_vmax   = ndata["vmax"]
                _global_vmin_f = np.array(ndata["vmin_f"]) if "vmin_f" in ndata else None
                _global_vmax_f = np.array(ndata["vmax_f"]) if "vmax_f" in ndata else None
                n_f = len(_global_vmin_f) if _global_vmin_f is not None else 0
                print(f"  Tile norm: global mode (v{TILE_NORM_VERSION}), "
                      f"vmin={_global_vmin:.1f}  vmax={_global_vmax:.1f} dB  |  "
                      f"per-freq flat: {n_f} bins")
                return
            else:
                print("  Tile norm version changed — purging cached tiles…")
        except Exception:
            pass

    # Version mismatch or missing → purge stale tile PNGs
    if os.path.isdir(TILE_DIR):
        for fn in os.listdir(TILE_DIR):
            if (fn.startswith("tile_") or fn.startswith("mask_tile_")
                    or fn.startswith("flat_tile_")) and fn.endswith(".png"):
                try:
                    os.remove(os.path.join(TILE_DIR, fn))
                except Exception:
                    pass
    with tile_lock:
        tile_cache.clear()
    with mask_tile_lock:
        mask_tile_cache.clear()
    with flat_tile_lock:
        flat_tile_cache.clear()

    # Sample ~30 tiles evenly across the recording to compute global stats
    sr  = finfo["sr"]
    dur = finfo["duration_s"]
    ntiles = int(np.ceil(dur / TILE_DURATION))
    sample_idxs = np.linspace(0, ntiles - 1, min(30, ntiles), dtype=int)

    print(f"  Computing global tile normalization from {len(sample_idxs)} tiles…",
          flush=True)
    tile_vmins, tile_vmaxs = [], []
    tile_pct_los, tile_pct_his = [], []   # per-frequency arrays for flat normalization

    for tidx in sample_idxs:
        try:
            t0 = tidx * TILE_DURATION
            t1 = min(t0 + TILE_DURATION, dur)
            f0 = int(t0 * sr); f1 = int(t1 * sr)
            if f1 <= f0:
                continue
            with audio_lock:
                audio_fh.seek(f0)
                audio = audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            f_s, _, Sxx = signal.spectrogram(
                mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
            bm  = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
            Sdb = 10 * np.log10(Sxx[bm, :] + 1e-12)   # (n_freq, n_time)

            # Global (scalar) stats for raw tiles:
            # bat calls cover ~1% of pixels, so 99.9th captures call energy.
            tile_vmins.append(float(np.percentile(Sdb,   2.0)))
            tile_vmaxs.append(float(np.percentile(Sdb, 99.9)))

            # Per-frequency stats for flat tiles: one number per bin per tile.
            # Stacking across sample tiles then taking inter-tile percentiles
            # gives time-invariant, frequency-specific normalization constants.
            tile_pct_los.append(np.percentile(Sdb, 2.0,  axis=1))   # (n_freq,)
            tile_pct_his.append(np.percentile(Sdb, 99.9, axis=1))   # (n_freq,)

        except Exception as exc:
            print(f"    tile {tidx} failed: {exc}")

    if tile_vmins:
        _global_vmin = float(np.percentile(tile_vmins, 50))
        _global_vmax = float(np.percentile(tile_vmaxs, 75))
    else:
        _global_vmin = -100.0
        _global_vmax =  -30.0

    if tile_pct_los:
        plo = np.vstack(tile_pct_los)     # (n_samples, n_freq)
        phi = np.vstack(tile_pct_his)
        _global_vmin_f = np.percentile(plo, 50, axis=0)   # median noise floor per bin
        _global_vmax_f = np.percentile(phi, 75, axis=0)   # 75th-pct signal ceiling per bin
    else:
        _global_vmin_f = None
        _global_vmax_f = None

    n_f = len(_global_vmin_f) if _global_vmin_f is not None else 0
    print(f"  Tile norm: global mode (v{TILE_NORM_VERSION}), "
          f"vmin={_global_vmin:.1f}  vmax={_global_vmax:.1f} dB  |  "
          f"per-freq flat: {n_f} bins — tiles will be regenerated")

    try:
        os.makedirs(TILE_DIR, exist_ok=True)
        ndata = {
            "version": TILE_NORM_VERSION,
            "mode":    "global",
            "vmin":    _global_vmin,
            "vmax":    _global_vmax,
        }
        if _global_vmin_f is not None:
            ndata["vmin_f"] = _global_vmin_f.tolist()
            ndata["vmax_f"] = _global_vmax_f.tolist()
        with open(norm_path, "w") as fh:
            json.dump(ndata, fh)
    except Exception as exc:
        print(f"  Warning: could not write norm.json ({exc})")


def make_tile(tidx):
    # 1. RAM cache (fastest)
    with tile_lock:
        if tidx in tile_cache:
            return tile_cache[tidx]

    # 2. Disk cache (fast — avoids re-running STFT)
    if TILE_DIR:
        disk_path = os.path.join(TILE_DIR, f"tile_{tidx:04d}.png")
        if os.path.exists(disk_path):
            with open(disk_path, "rb") as fh:
                data = fh.read()
            with tile_lock:
                tile_cache[tidx] = data
            return data

    # 3. Generate from audio
    sr     = finfo["sr"]
    dur    = finfo["duration_s"]
    t0     = tidx * TILE_DURATION
    t1     = min(t0 + TILE_DURATION, dur)
    f0     = int(t0 * sr);  f1 = int(t1 * sr)

    with audio_lock:
        audio_fh.seek(f0)
        audio = audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f, _, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm   = (f >= FREQ_LOW) & (f <= FREQ_HIGH)
    Sdb  = 10 * np.log10(Sxx[bm, :] + 1e-12)

    # Global normalization: same dB scale for every tile so call-isolated
    # crossfade makes sense (both spectrograms share one scale).
    # _global_vmin/_global_vmax are computed at startup from ~30 sampled tiles.
    arr  = np.clip((Sdb - _global_vmin) / max(_global_vmax - _global_vmin, 1e-6), 0, 1)
    rgb  = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)

    pil  = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    # 4. Save to disk cache (persist across server restarts)
    if TILE_DIR:
        try:
            os.makedirs(TILE_DIR, exist_ok=True)
            with open(disk_path, "wb") as fh:
                fh.write(data)
        except Exception:
            pass

    # 5. Store in RAM cache (no eviction — 253 tiles ≈ 25 MB, negligible)
    with tile_lock:
        tile_cache[tidx] = data
    return data


def _pregenerate_tiles():
    """Background thread: walk every tile so they're disk-cached before the user zooms out.
    Also pre-generates mask tiles once detection results are available.
    """
    ntiles = int(np.ceil(finfo["duration_s"] / TILE_DURATION))
    missing = [i for i in range(ntiles)
               if i not in tile_cache and
               not os.path.exists(os.path.join(TILE_DIR, f"tile_{i:04d}.png"))]
    if not missing:
        print("All raw tiles already cached on disk.")
    else:
        print(f"Pre-generating {len(missing)} raw tiles in background…", flush=True)
        for i in missing:
            try:
                make_tile(i)
            except Exception as exc:
                print(f"  tile {i} failed: {exc}")
        print(f"Raw tile pre-generation done ({ntiles} tiles total).", flush=True)

    # Pre-generate flat (frequency-compensated) tiles in parallel with detection
    flat_missing = [i for i in range(ntiles)
                    if i not in flat_tile_cache and
                    not os.path.exists(os.path.join(TILE_DIR, f"flat_tile_{i:04d}.png"))]
    if not flat_missing:
        print("All flat tiles already cached on disk.")
    else:
        print(f"Pre-generating {len(flat_missing)} flat tiles in background…", flush=True)
        for i in flat_missing:
            try:
                make_flat_tile(i)
            except Exception as exc:
                print(f"  flat tile {i} failed: {exc}")
        print(f"Flat tile pre-generation done ({ntiles} tiles total).", flush=True)

    # Wait for detection to finish, then pre-generate mask tiles
    calls_ready.wait()
    mask_missing = [i for i in range(ntiles)
                    if i not in mask_tile_cache and
                    not os.path.exists(os.path.join(TILE_DIR, f"mask_tile_{i:04d}.png"))]
    if not mask_missing:
        print("All mask tiles already cached on disk.")
        return
    print(f"Pre-generating {len(mask_missing)} mask tiles in background…", flush=True)
    for i in mask_missing:
        try:
            make_mask_tile(i)
        except Exception as exc:
            print(f"  mask tile {i} failed: {exc}")
    print(f"Mask tile pre-generation done ({ntiles} tiles total).", flush=True)


def _compute_call_mask(tidx):
    """Build a soft 2-D mask marking call regions for tile `tidx`.

    Returns a float32 array of shape (n_freq, n_time) in the same coordinate
    system as Sdb: row 0 = lowest frequency (FREQ_LOW), row n-1 = highest
    (FREQ_HIGH).  Values are in [0, 1]: 1 = call energy, 0 = background.

    Algorithm
    ---------
    For each detected call whose time range overlaps this tile:
      • Walk every contour point that falls inside the tile's time window.
      • At each time frame, add a Gaussian bump in the frequency direction
        centred on the contour frequency.  σ = max(bandwidth/3, 2 kHz).
      • For constant-frequency bats (cf_frac > 0.35, e.g. TABR) also add a
        70%-amplitude Gaussian at twice the fundamental frequency so the
        second harmonic is preserved when the mask is applied.
    After all calls are accumulated, smooth in time (σ ≈ 1.5 frames) to
    connect adjacent contour points and soften hard edges.
    """
    sr  = finfo["sr"]
    dur = finfo["duration_s"]
    t0_tile = tidx * TILE_DURATION
    t1_tile = min(t0_tile + TILE_DURATION, dur)

    # Compute the display STFT just for the frequency/time arrays (Sxx unused)
    f0 = int(t0_tile * sr); f1 = int(t1_tile * sr)
    with audio_lock:
        audio_fh.seek(f0)
        audio = audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f_s, t_s, _ = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm    = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
    f_arr = f_s[bm]          # Hz, ascending
    n_freq = len(f_arr)
    n_time = len(t_s)

    mask = np.zeros((n_freq, n_time), dtype=np.float32)

    if not all_calls:
        return mask

    tile_dur = t1_tile - t0_tile

    for c in all_calls:
        if c["t1"] < t0_tile or c["t0"] > t1_tile:
            continue
        contour = c.get("contour")
        if not contour:
            continue

        # σ in Hz: one-third of the call's measured bandwidth, minimum 2 kHz
        bw_hz    = max(c.get("bw", max(c["Fmax"] - c["Fmin"], 2.0)) * 1000, 2000)
        sigma_hz = max(bw_hz / 3.0, 2000.0)
        cf_frac  = c.get("cf_frac", 0.5)
        add_harmonic = cf_frac > 0.35

        for ct, cf_khz in contour:
            if ct < t0_tile or ct > t1_tile:
                continue
            # Map absolute time → tile time index
            ti = int((ct - t0_tile) / tile_dur * n_time)
            ti = max(0, min(ti, n_time - 1))

            cf_hz = cf_khz * 1000.0
            # Gaussian in frequency
            gauss = np.exp(-0.5 * ((f_arr - cf_hz) / sigma_hz) ** 2)
            mask[:, ti] = np.maximum(mask[:, ti], gauss)

            # Second harmonic for CF bats (adds harmonic ridge to the mask)
            if add_harmonic:
                cf2_hz = cf_hz * 2.0
                if FREQ_LOW <= cf2_hz <= FREQ_HIGH:
                    g2 = np.exp(-0.5 * ((f_arr - cf2_hz) / sigma_hz) ** 2) * 0.7
                    mask[:, ti] = np.maximum(mask[:, ti], g2)

    # Smooth in time to fill gaps between contour points
    if n_time > 1:
        mask = gaussian_filter1d(mask, sigma=1.5, axis=1)

    return np.clip(mask, 0.0, 1.0)


def make_mask_tile(tidx):
    """Generate an RGBA mask tile: R=G=B=0, A=(1−mask)×255.

    When composited on top of the raw spectrogram at opacity `α` (crossfade):
      • Call regions (mask≈1): A≈0 → transparent → raw tile shows through.
      • Background (mask≈0): A≈255 → black at globalAlpha=α → dims the background.
    """
    with mask_tile_lock:
        if tidx in mask_tile_cache:
            return mask_tile_cache[tidx]

    if TILE_DIR:
        disk_path = os.path.join(TILE_DIR, f"mask_tile_{tidx:04d}.png")
        if os.path.exists(disk_path):
            with open(disk_path, "rb") as fh:
                data = fh.read()
            with mask_tile_lock:
                mask_tile_cache[tidx] = data
            return data
    else:
        disk_path = None

    # Compute mask (n_freq × n_time), row 0 = lowest freq
    mask_arr = _compute_call_mask(tidx)

    # Flip vertically so row 0 = highest freq (same orientation as make_tile)
    mask_flipped = mask_arr[::-1, :]

    # Build RGBA: R=G=B=0, A=(1-mask)*255
    h, w = mask_flipped.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = np.round((1.0 - mask_flipped) * 255).astype(np.uint8)

    pil  = Image.fromarray(rgba, mode="RGBA").resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    if disk_path:
        try:
            os.makedirs(TILE_DIR, exist_ok=True)
            with open(disk_path, "wb") as fh:
                fh.write(data)
        except Exception:
            pass

    with mask_tile_lock:
        mask_tile_cache[tidx] = data
    return data


def _flat_gain_db(f_hz):
    """Per-frequency gain correction (dB) to compensate for the EM258 capsule rolloff.

    The Avisoft EM258 (and similar CCP electret ultrasonic capsules) have an
    approximately first-order (6 dB/octave) rolloff starting around 40 kHz.
    This function returns the dB boost needed to flatten the response so that
    energy at higher frequencies is displayed at equivalent brightness to
    lower-frequency energy of the same physical amplitude.

    Model: first-order high-pass shelf starting at f_ref = 40 kHz.
      boost(f) = max(0,  20 · log10(f / f_ref))  dB
    Gives:  13 kHz →  0.0 dB
            40 kHz →  0.0 dB  (rolloff start)
            60 kHz → +3.5 dB
            80 kHz → +6.0 dB  (one octave above f_ref)
            96 kHz → +7.6 dB

    Consult the actual EM258 datasheet (Avisoft-Bioacoustics) to calibrate
    f_ref and the rolloff order for your specific unit.
    """
    f_ref = 40_000.0  # Hz — frequency where capsule rolloff begins
    boost = np.maximum(0.0, 20.0 * np.log10(np.maximum(f_hz, f_ref) / f_ref))
    return boost   # shape matches f_hz


def make_flat_tile(tidx):
    """Per-frequency-normalised spectrogram tile (the "Flat" view).

    Why not a mic-response gain boost?
    -----------------------------------
    Applying a frequency-dependent dB gain (e.g. +7.6 dB at 96 kHz to
    compensate EM258 rolloff) is mathematically identical to multiplying the
    linear power spectrum by a frequency-dependent constant.  Both operations
    lift the *noise floor* by the same factor as the signal, so the background
    becomes progressively brighter at high frequencies — the "pink gradient"
    the user noticed.  No form of linear gain (additive in dB, multiplicative
    in power, or IIR/FIR in the time domain) can keep the noise floor dark
    while boosting bat calls, because it is frequency-blind: it doesn't know
    whether a given pixel is a bat call or background noise.

    Per-frequency normalisation (spectral whitening)
    -------------------------------------------------
    For each frequency bin k, we compute its local noise floor (2nd percentile
    over time) and its local signal ceiling (99.9th percentile over time) from
    this tile's own data, then map:

        arr[k, t] = clip( (Sdb[k,t] − lo[k]) / (hi[k] − lo[k]),  0, 1 )

    Result: the noise appears dark at every frequency; energy that stands
    above the local floor (bat calls, harmonics) appears bright — regardless
    of the recording chain's absolute frequency response.

    Physically this removes any spectrally-coloured noise source (mic rolloff,
    preamp shape, narrowband interference) and shows *deviation from the local
    noise floor*.  It is not a calibrated amplitude display, but for visually
    identifying calls at all frequencies it is the right tool.

    Bat calls occupy ~1% of pixels per tile, so the 99.9th-percentile captures
    call energy while the 2nd-percentile sits stably in the noise floor.
    """
    with flat_tile_lock:
        if tidx in flat_tile_cache:
            return flat_tile_cache[tidx]

    if TILE_DIR:
        disk_path = os.path.join(TILE_DIR, f"flat_tile_{tidx:04d}.png")
        if os.path.exists(disk_path):
            with open(disk_path, "rb") as fh:
                data = fh.read()
            with flat_tile_lock:
                flat_tile_cache[tidx] = data
            return data
    else:
        disk_path = None

    sr  = finfo["sr"]
    dur = finfo["duration_s"]
    t0  = tidx * TILE_DURATION
    t1  = min(t0 + TILE_DURATION, dur)
    f0  = int(t0 * sr); f1 = int(t1 * sr)

    with audio_lock:
        audio_fh.seek(f0)
        audio = audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    f_s, _, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm  = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
    Sdb = 10 * np.log10(Sxx[bm, :] + 1e-12)   # (n_freq, n_time)

    # Per-frequency normalization using globally pre-computed constants so the
    # scale is time-invariant (same for every tile → no brightness jumps at
    # tile boundaries).  _global_vmin_f/_global_vmax_f are (n_freq,) arrays
    # computed from 30 sample tiles during startup.
    n_freq = Sdb.shape[0]
    if (_global_vmin_f is not None and _global_vmax_f is not None
            and len(_global_vmin_f) == n_freq):
        lo = _global_vmin_f[:, np.newaxis]   # (n_freq, 1) → broadcasts over time
        hi = _global_vmax_f[:, np.newaxis]
    else:
        # Fallback if global stats are missing or frequency-bin count changed
        lo = np.percentile(Sdb, 2.0,  axis=1, keepdims=True)
        hi = np.percentile(Sdb, 99.9, axis=1, keepdims=True)
    arr = np.clip((Sdb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)

    pil  = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()

    if disk_path:
        try:
            os.makedirs(TILE_DIR, exist_ok=True)
            with open(disk_path, "wb") as fh:
                fh.write(data)
        except Exception:
            pass

    with flat_tile_lock:
        flat_tile_cache[tidx] = data
    return data


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/info")
def api_info():
    return jsonify({
        **finfo,
        "tile_duration": TILE_DURATION,
        "tile_w": TILE_W, "tile_h": TILE_H,
        "freq_low": FREQ_LOW_K, "freq_high": FREQ_HIGH_K,
        "n_tiles": int(np.ceil(finfo["duration_s"] / TILE_DURATION)),
        "tile_version": TILE_NORM_VERSION,
        "colors": COLORS,
        "ready": calls_ready.is_set(),
        "progress": progress,
    })

@app.route("/api/status")
def api_status():
    return jsonify({"ready": calls_ready.is_set(), "progress": progress})

@app.route("/api/calls")
def api_calls():
    return jsonify({"ready": calls_ready.is_set(),
                    "calls": list(all_calls)})

@app.route("/api/profiles")
def api_profiles():
    """Return PROFILES list with all scholarly reference data (tuples → lists for JSON)."""
    out = []
    for p in PROFILES:
        entry = dict(p)
        entry["Fchar"] = list(p["Fchar"])
        entry["Fmin"]  = list(p["Fmin"])
        entry["dur"]   = list(p["dur"])
        entry["sweep"] = list(p["sweep"])
        entry["color"] = COLORS.get(p["name"], "#888888")
        out.append(entry)
    # Also include an Unclassified pseudo-profile
    out.append({
        "name": "Unclassified", "short": "????",
        "Fchar": None, "Fmin": None, "dur": None, "sweep": None,
        "common": "Unclassified",
        "call_type": "Does not strongly match any reference profile.",
        "desc": (
            "Calls that did not score ≥ 50% against any of the heuristic species profiles. "
            "May represent species not in the reference set, poor-quality detections, "
            "or edge-of-range calls that fall between profiles."
        ),
        "habitat": "—", "range": "—", "ipi_ms": "—", "refs": [],
        "color": COLORS.get("Unclassified", "#888888"),
    })
    return jsonify(out)

@app.route("/api/tile/<int:tidx>")
def api_tile(tidx):
    ntiles = int(np.ceil(finfo["duration_s"] / TILE_DURATION))
    if tidx < 0 or tidx >= ntiles:
        return "not found", 404
    data = make_tile(tidx)
    return send_file(io.BytesIO(data), mimetype="image/png",
                     max_age=3600)

@app.route("/api/tile_mask/<int:tidx>")
def api_tile_mask(tidx):
    ntiles = int(np.ceil(finfo["duration_s"] / TILE_DURATION))
    if tidx < 0 or tidx >= ntiles:
        return "not found", 404
    if not calls_ready.is_set():
        return "detection not ready", 503
    data = make_mask_tile(tidx)
    return send_file(io.BytesIO(data), mimetype="image/png",
                     max_age=3600)

@app.route("/api/tile_flat/<int:tidx>")
def api_tile_flat(tidx):
    ntiles = int(np.ceil(finfo["duration_s"] / TILE_DURATION))
    if tidx < 0 or tidx >= ntiles:
        return "not found", 404
    data = make_flat_tile(tidx)
    return send_file(io.BytesIO(data), mimetype="image/png",
                     max_age=3600)

@app.route("/api/conversation")
def api_conversation():
    """Return cleaned human/assistant turns from the Claude Code session log.

    Checks for a pre-exported conversation.json next to the script first
    (works on any server), then falls back to live JSONL files in
    ~/.claude/projects/ (works on the dev machine).
    """
    import pathlib

    script_dir   = pathlib.Path(__file__).parent
    bundled_path = script_dir / "conversation.json"

    # ── 1. Pre-exported bundle (committed to repo) ────────────────
    if bundled_path.exists():
        try:
            with open(bundled_path) as fh:
                data = json.load(fh)
            return jsonify({"messages": data.get("messages", []),
                            "source": str(bundled_path)})
        except Exception:
            pass

    # ── 2. Live JSONL (dev machine fallback) ─────────────────────
    project_dir = pathlib.Path.home() / ".claude" / "projects" / "-Users-brandon-claude"
    candidates  = sorted(project_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True)

    messages = []
    for jsonl_path in candidates:
        try:
            with open(jsonl_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj  = json.loads(line)
                    typ  = obj.get("type")
                    msg  = obj.get("message", {})
                    role = msg.get("role", "")
                    if typ not in ("user", "assistant") or role not in ("user", "assistant"):
                        continue
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        parts = [c.get("text", "") for c in content
                                 if isinstance(c, dict) and c.get("type") == "text"]
                        text = "\n".join(parts).strip()
                    else:
                        text = str(content).strip()
                    if (not text
                            or text.startswith("<system-reminder")
                            or text.startswith("<function_calls>")):
                        continue
                    messages.append({
                        "role": role,
                        "text": text,
                        "ts":   obj.get("timestamp", ""),
                    })
            if messages:
                break
        except Exception:
            continue

    return jsonify({"messages": messages,
                    "source": str(candidates[0]) if candidates else ""})

# ─────────────────────────────────────────────
# Frontend HTML + CSS + JS (embedded)
# ─────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Bat Spectrogram Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0e0e0e; color: #ddd; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

#header { padding: 8px 14px; background: #1a1a1a; border-bottom: 1px solid #2a2a2a; display: flex; align-items: center; gap: 20px; flex-shrink: 0; }
#header h1 { font-size: 14px; font-weight: 600; color: #eee; }
#header .meta { font-size: 11px; color: #777; }
.clf-active { background: #1a2a1a !important; color: #59a14f !important; }
#status-bar { font-size: 11px; color: #f28e2b; margin-left: auto; }

#main { display: flex; flex: 1; overflow: hidden; }

#canvas-col { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; }

#controls { padding: 5px 10px; background: #161616; border-bottom: 1px solid #222; display: flex; align-items: flex-start; gap: 10px; flex-shrink: 0; flex-wrap: nowrap; overflow-x: auto; }
#controls button { background: #2a2a2a; border: 1px solid #3a3a3a; color: #ccc; padding: 3px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; flex-shrink: 0; }
#controls button:hover { background: #383838; }
.ctrl-group { display: flex; flex-direction: column; gap: 2px; }
.ctrl-group-label { font-size: 9px; color: #555; text-transform: uppercase; letter-spacing: .07em; line-height: 1; }
.ctrl-group-body { display: flex; align-items: flex-start; gap: 8px; flex-wrap: wrap; row-gap: 4px; }
.ctrl-lbl { color: #aaa; font-size: 11px; display: flex; align-items: center; gap: 5px; white-space: nowrap; }
.ctrl-sep { width: 1px; height: 28px; background: #2a2a2a; flex-shrink: 0; }
/* Headerless sub-group: items stack in a column, used inside ctrl-group-body */
.ctrl-subgroup { display: flex; flex-direction: column; gap: 4px; flex-shrink: 0; }
/* 2-column grid variant: items distribute evenly 2 per row */
.ctrl-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; }
/* Inside a 2-col label: right-align left text, left-align right text, slider fills middle */
.ctrl-2col .ctrl-lbl .sl-l { width: 30px; text-align: right; flex-shrink: 0; }
.ctrl-2col .ctrl-lbl .sl-r { width: 36px; text-align: left;  flex-shrink: 0; }
.ctrl-2col .ctrl-lbl input[type=range] { flex: 1; min-width: 0; width: auto; }

/* ── Cross-browser range slider track fill ───────────────────────────── */
input[type=range] {
  -webkit-appearance: none; appearance: none;
  height: 4px; border-radius: 2px; outline: none; cursor: pointer;
  /* fill left of thumb; --fill and --pct are set by JS per slider */
  background: linear-gradient(to right, var(--fill,#888) var(--pct,0%), #3a3a3a var(--pct,0%));
  vertical-align: middle;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 13px; height: 13px; border-radius: 50%;
  background: var(--fill, #aaa); cursor: pointer;
  box-shadow: 0 0 0 2px #161616;
}
/* Firefox track / progress / thumb */
input[type=range]::-moz-range-track   { height: 4px; background: #3a3a3a; border-radius: 2px; }
input[type=range]::-moz-range-progress{ height: 4px; background: var(--fill,#888); border-radius: 2px; }
input[type=range]::-moz-range-thumb   {
  border: none; width: 13px; height: 13px; border-radius: 50%;
  background: var(--fill,#aaa); cursor: pointer;
  box-shadow: 0 0 0 2px #161616;
}
.time-display { color: #aaa; font-size: 11px; margin-left: auto; min-width: 13em; flex-shrink: 0; line-height: 1.6; }

#canvas-wrap { position: relative; flex: 1; overflow: hidden; display: flex; flex-direction: row; }
#mainCanvas { display: block; flex: 1; min-width: 0; cursor: crosshair; }

/* Frequency range scrollbar */
#freq-scrollbar { width: 20px; flex-shrink: 0; background: #111; border-left: 1px solid #222; position: relative; user-select: none; cursor: default; }
#freq-sb-track  { position: absolute; left: 4px; right: 4px; top: 0; bottom: 0; }
#freq-sb-fill   { position: absolute; left: 0; right: 0; background: #2d3d2d; border-radius: 2px; cursor: grab; }
#freq-sb-fill:active { cursor: grabbing; }
#freq-sb-fill:hover  { background: #3a4e3a; }
.freq-sb-handle { position: absolute; left: -2px; right: -2px; height: 7px; background: #f28e2b; border-radius: 3px; cursor: ns-resize; transform: translateY(-50%); }
.freq-sb-handle:hover { background: #ffba5a; }
/* Tick marks */
.freq-sb-tick { position: absolute; right: 0; width: 4px; height: 1px; background: #333; pointer-events: none; }
#tooltip { position: absolute; background: rgba(10,10,10,0.92); border: 1px solid #333; border-radius: 4px; padding: 8px 10px; font-size: 11px; line-height: 1.6; pointer-events: none; display: none; max-width: 220px; z-index: 10; }
#tooltip .sp-name { font-size: 12px; font-weight: 700; margin-bottom: 3px; }
#tooltip .param { color: #aaa; }
#tooltip .param span { color: #eee; }

#overview-wrap { flex-shrink: 0; height: 64px; background: #111; border-top: 1px solid #222; position: relative; }
#overviewCanvas { display: block; }

/* ── Right-panel accordion ── */
#detail { width: 260px; flex-shrink: 0; background: #131313; border-left: 1px solid #222; overflow: hidden; display: flex; flex-direction: column; }

/* Call accordion – top */
/* ── Accordion layout ──────────────────────────────────────── */
/* Call section — top */
#acc-call-wrap { flex: 0 0 auto; display: flex; flex-direction: column; min-height: 0; }
#acc-call-wrap.acc-open { flex: 1 1 0; }
.acc-section-header { display: flex; align-items: center; gap: 7px; padding: 9px 12px; cursor: pointer; border-bottom: 1px solid #222; user-select: none; background: #141414; flex-shrink: 0; }
.acc-section-header:hover { background: #1a1a1a; }
.acc-chevron { font-size: 10px; color: #555; width: 10px; flex-shrink: 0; }
.acc-section-title { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: .08em; flex: 1; }
.acc-section-meta { font-size: 11px; color: #aaa; font-weight: 600; max-width: 130px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
/* Body is hidden by default; grows + scrolls when its section has .acc-open */
.acc-body { display: none; overflow-y: auto; flex: 1 1 0; min-height: 0; padding: 12px; }
#acc-call-wrap.acc-open .acc-body { display: block; }
.acc-empty { color: #555; font-size: 12px; }
.acc-sp-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; color: #fff; margin-bottom: 8px; }
.acc-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.acc-table td { padding: 3px 0; color: #aaa; }
.acc-table td:last-child { color: #ddd; text-align: right; }
.acc-sub-header { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: .07em; border-top: 1px solid #1e1e1e; margin-top: 10px; padding-top: 8px; margin-bottom: 5px; display: flex; align-items: center; justify-content: space-between; }
.acc-zoom-btn { cursor: pointer; color: #f28e2b; font-size: 11px; text-transform: none; letter-spacing: 0; }

/* Species section — always fills remaining panel space */
#acc-species { flex: 1 1 0; display: flex; flex-direction: column; min-height: 0; border-top: 1px solid #222; }

/* Scrollable container — flex column so items stack; overflow hidden (body scrolls inside) */
#acc-sp-scroll { display: flex; flex-direction: column; flex: 1 1 0; min-height: 0; overflow: hidden; }

/* Each species item: fixed-height header by default; grows to fill remaining space when open */
.sp-acc-item { flex: 0 0 auto; border-top: 1px solid #1a1a1a; }
.sp-acc-item.acc-open { flex: 1 1 0; display: flex; flex-direction: column; min-height: 0; }
.sp-acc-header { display: flex; align-items: center; gap: 6px; padding: 7px 10px; cursor: pointer; user-select: none; flex-shrink: 0; }
.sp-acc-header:hover { background: #181818; }
.sp-acc-header.hidden-sp { opacity: 0.35; }
.sp-acc-header.acc-active { background: #1a1a1a; }
/* Body hidden by default; when item is open it fills remaining space and scrolls internally */
.sp-acc-body { display: none; padding: 12px; border-bottom: 1px solid #1c1c1c; }
.sp-acc-item.acc-open .sp-acc-body { display: block; overflow-y: auto; flex: 1 1 0; min-height: 0; }
.sp-acc-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.sp-acc-chk { cursor: pointer; accent-color: #f28e2b; flex-shrink: 0; }
.sp-acc-name { font-size: 11px; color: #aaa; flex: 1; }
.sp-acc-count { font-size: 10px; color: #555; }
.sp-acc-arrow { font-size: 9px; color: #444; margin-left: 3px; }
.sp-solo-btn { background: none; border: none; cursor: pointer; color: #3a3a3a; font-size: 13px; padding: 0 1px; line-height: 1; flex-shrink: 0; }
.sp-solo-btn:hover { color: #666; }
.sp-solo-btn.soloed { color: #59a14f; }

/* Shared content styles (used in both accordions) */
.sp-section { margin-bottom: 12px; }
.sp-section h4 { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 5px; }
.sp-section p { font-size: 11px; color: #888; line-height: 1.65; margin: 0; }
.sp-profile-row { display: flex; justify-content: space-between; font-size: 11px; padding: 2px 0; border-bottom: 1px solid #1a1a1a; }
.sp-profile-row .prl { color: #666; }
.sp-profile-row .prv { color: #bbb; }
.sp-stats-tbl { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 4px; }
.sp-stats-tbl th { font-size: 9px; color: #444; text-align: right; padding: 1px 3px; font-weight: normal; text-transform: uppercase; }
.sp-stats-tbl th:first-child { text-align: left; }
.sp-stats-tbl td { padding: 2px 3px; color: #888; text-align: right; }
.sp-stats-tbl td:first-child { color: #666; text-align: left; }
.ref-tag { font-size: 10px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 3px; padding: 1px 5px; color: #666; display: inline-block; margin: 2px 2px 2px 0; }

#progress-overlay { position: absolute; inset: 0; background: rgba(14,14,14,0.85); display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; z-index: 20; }
#progress-overlay p { color: #aaa; font-size: 13px; }
#pbar-wrap { width: 260px; height: 6px; background: #2a2a2a; border-radius: 3px; }
#pbar { height: 100%; background: #f28e2b; border-radius: 3px; width: 0%; transition: width 0.3s; }

/* ── Modal dialogs ── */
.modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.72); z-index: 100; display: none; align-items: center; justify-content: center; }
.modal-backdrop.open { display: flex; }
.modal { background: #181818; border: 1px solid #333; border-radius: 8px; display: flex; flex-direction: column; max-height: 88vh; box-shadow: 0 8px 40px rgba(0,0,0,0.6); }
.modal-header { padding: 14px 18px; border-bottom: 1px solid #2a2a2a; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
.modal-header h2 { font-size: 14px; font-weight: 600; color: #eee; flex: 1; }
.modal-close { background: none; border: none; color: #666; font-size: 18px; cursor: pointer; padding: 0 4px; line-height: 1; }
.modal-close:hover { color: #ccc; }
.modal-body { overflow-y: auto; padding: 18px; flex: 1; }

/* Session dialog */
#session-modal .modal { width: min(860px, 96vw); }
.conv-turn { margin-bottom: 18px; }
.conv-turn .role { font-size: 10px; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin-bottom: 5px; }
.conv-turn.user .role { color: #76b7b2; }
.conv-turn.assistant .role { color: #f28e2b; }
.conv-turn .bubble { background: #111; border: 1px solid #252525; border-radius: 6px; padding: 10px 14px; font-size: 12px; line-height: 1.7; color: #ccc; white-space: pre-wrap; word-break: break-word; }
.conv-turn.user .bubble { border-color: #1e3333; }
.conv-turn .ts { font-size: 10px; color: #444; margin-top: 3px; }

/* About dialog */
#about-modal .modal { width: min(560px, 96vw); }
.about-section { margin-bottom: 18px; }
.about-section h3 { font-size: 12px; color: #f28e2b; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 8px; }
.about-section p, .about-section li { font-size: 12px; color: #aaa; line-height: 1.7; }
.about-section a { color: #76b7b2; text-decoration: none; }
.about-section a:hover { text-decoration: underline; }
.about-section ul { padding-left: 18px; }
.apology { background: #1a1410; border: 1px solid #3a2a10; border-radius: 6px; padding: 12px 14px; margin-top: 8px; font-size: 12px; color: #aaa; line-height: 1.7; font-style: italic; }
</style>
</head>
<body>

<div id="header">
  <h1>Bat Spectrogram Viewer</h1>
  <span class="meta" id="file-meta">Loading…</span>
  <span id="status-bar"></span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:6px;">
    <span style="font-size:10px;color:#555;letter-spacing:.04em">CLASSIFIER</span>
    <div style="display:flex;border:1px solid #2a3a2a;border-radius:3px;overflow:hidden">
      <button id="clf-v1" onclick="setClassifier('v1')"
        style="background:#111;color:#555;border:none;padding:3px 8px;cursor:pointer;font-size:10px;font-family:inherit"
        title="v1: peak freq / Fmin / duration / sweep rate (4 criteria)">v1</button>
      <button id="clf-v2" onclick="setClassifier('v2')"
        class="clf-active"
        style="background:#111;color:#555;border:none;padding:3px 8px;cursor:pointer;font-size:10px;font-family:inherit;border-left:1px solid #2a3a2a"
        title="v2: + bandwidth + CF fraction (6 criteria)">v2</button>
    </div>
  </div>
  <button id="btn-session" onclick="openSession()" style="background:#1a2a2a;border:1px solid #2a3a3a;color:#76b7b2;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:11px;font-family:inherit;">Claude session ↗</button>
  <button id="btn-about" onclick="openAbout()" style="background:#1a1a2a;border:1px solid #2a2a3a;color:#888;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:11px;font-family:inherit;">About</button>
</div>

<!-- Session dialog -->
<div class="modal-backdrop" id="session-modal" onclick="closeModal('session-modal')">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-header">
      <h2>Claude Source Session</h2>
      <span style="font-size:11px;color:#555;flex:1">This viewer was built entirely through a conversation with Claude.</span>
      <button class="modal-close" onclick="closeModal('session-modal')">✕</button>
    </div>
    <div class="modal-body" id="session-body">
      <p style="color:#555;font-size:12px">Loading conversation…</p>
    </div>
  </div>
</div>

<!-- About dialog -->
<div class="modal-backdrop" id="about-modal" onclick="closeModal('about-modal')">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-header">
      <h2>About Bat Spectrogram Viewer</h2>
      <button class="modal-close" onclick="closeModal('about-modal')">✕</button>
    </div>
    <div class="modal-body">
      <div class="about-section">
        <h3>Detection Model</h3>
        <p>Bat call detection uses <strong>BatDetect2</strong>, a fully-convolutional neural network with self-attention, trained on 17 UK bat species and applied here for detection only (species classification uses separate California-tuned heuristic profiles).</p>
        <ul style="margin-top:8px">
          <li><a href="https://github.com/macaodha/batdetect2" target="_blank">github.com/macaodha/batdetect2</a></li>
          <li><a href="https://doi.org/10.1371/journal.pcbi.1011333" target="_blank">Mac Aodha et al. (2023), PLOS Computational Biology</a></li>
        </ul>
        <p style="margin-top:8px;font-size:11px;color:#666">Citation: Mac Aodha O, et al. "Towards a General Approach for Bat Echolocation Detection and Classification." <em>PLOS Computational Biology</em> 19(8): e1011333 (2023).</p>
      </div>
      <div class="about-section">
        <h3>Recording</h3>
        <p>Recorded 2025-05-28 at 19:42, Campbell CA, USA. Zoom F3 field recorder. 192 kHz / 24-bit stereo FLAC, 21 min 2 sec.</p>
      </div>
      <div class="about-section">
        <h3>A note on how this was made</h3>
        <div class="apology">
          <strong style="color:#c8a060;font-style:normal">An apology for vibe coding.</strong><br><br>
          This entire application — signal processing pipeline, Flask backend, spectrogram renderer, interactive UI, BatDetect2 integration — was written by Claude (Anthropic's AI assistant) in a single conversation, with the human providing direction but no code.<br><br>
          This is "vibe coding": steering a language model by feel rather than by careful engineering. The result works, but it carries all the hallmarks: inconsistent abstractions, accumulating technical debt with each iteration, decisions made by pattern-matching to training data rather than genuine understanding of your specific constraints.<br><br>
          If you're using this for scientific work, please review the detection thresholds, the species classification profiles, and the signal processing parameters critically. The neural net (BatDetect2) is peer-reviewed; the wrapper is vibes.<br><br>
          <span style="color:#665544">— Claude Sonnet 4.5, May 2026</span>
        </div>
      </div>
      <div class="about-section">
        <h3>Source</h3>
        <p><a href="https://github.com/bmechtley/xpabat" target="_blank">github.com/bmechtley/xpabat</a> · <button onclick="openSession();closeModal('about-modal')" style="background:none;border:none;color:#76b7b2;cursor:pointer;font-size:12px;padding:0;font-family:inherit">View full Claude session ↗</button></p>
      </div>
    </div>
  </div>
</div>

<div id="main">
  <div id="canvas-col">
    <div id="controls">
      <button id="btn-zoom-in">Zoom In</button>
      <button id="btn-zoom-out">Zoom Out</button>
      <button id="btn-fit">Fit All</button>
      <div class="ctrl-group">
        <div class="ctrl-group-label">Contours</div>
        <div class="ctrl-group-body">
          <div class="ctrl-subgroup">
            <label class="ctrl-lbl"><input type="checkbox" id="chk-contour" checked> Lines</label>
            <label class="ctrl-lbl"><input type="checkbox" id="chk-boxes"> Boxes</label>
          </div>
          <div class="ctrl-subgroup">
            <label class="ctrl-lbl" title="Contour line opacity">
              Opacity <input type="range" id="slider-contour-alpha" min="10" max="100" value="55" style="width:60px;--fill:#f28e2b"> <span id="contour-alpha-val">55%</span>
            </label>
            <label class="ctrl-lbl" title="Hover/click picking tolerance in pixels — distance from cursor to nearest call bounding box">
              Pick <input type="range" id="slider-pick-radius" min="0" max="80" value="20" style="width:60px;--fill:#f28e2b"> <span id="pick-radius-val">20</span>px
            </label>
          </div>
        </div>
      </div>
      <div class="ctrl-sep"></div>
      <div class="ctrl-group">
        <div class="ctrl-group-label">Spectrogram</div>
        <div class="ctrl-group-body ctrl-2col">
          <label class="ctrl-lbl" title="Crossfade: raw spectrogram ↔ call-isolated view">
            <span class="sl-l">Raw</span><input type="range" id="slider-crossfade" min="0" max="100" value="0" style="--fill:#59a14f"><span class="sl-r">Calls</span>
          </label>
          <label class="ctrl-lbl" title="Frequency compensation: raw ↔ mic-response-flattened">
            <span class="sl-l">Raw</span><input type="range" id="slider-flatness" min="0" max="100" value="0" style="--fill:#76b7b2"><span class="sl-r">Flat</span>
          </label>
          <label class="ctrl-lbl" title="Frequency axis: linear ↔ logarithmic">
            <span class="sl-l">Lin</span><input type="range" id="slider-log" min="0" max="100" value="0" style="--fill:#76b7b2"><span class="sl-r">Log</span>
          </label>
          <label class="ctrl-lbl" title="Spectrogram color saturation — reduce to make contours stand out against a grey background">
            <span class="sl-l">Gray</span><input type="range" id="slider-sat" min="0" max="100" value="100" style="--fill:#76b7b2"><span class="sl-r">Color</span>
          </label>
        </div>
      </div>
      <div class="time-display" id="time-display">–</div>
    </div>
    <div id="canvas-wrap">
      <canvas id="mainCanvas"></canvas>
      <div id="freq-scrollbar">
        <div id="freq-sb-track">
          <div id="freq-sb-fill">
            <div class="freq-sb-handle" id="freq-sb-top"></div>
            <div class="freq-sb-handle" id="freq-sb-bot" style="top:100%"></div>
          </div>
        </div>
      </div>
      <div id="tooltip"></div>
      <div id="progress-overlay">
        <p id="progress-msg">Opening audio file…</p>
        <div id="pbar-wrap"><div id="pbar"></div></div>
      </div>
    </div>
    <div id="overview-wrap">
      <canvas id="overviewCanvas"></canvas>
    </div>
  </div>

  <div id="detail">
    <!-- Call accordion — pinned top, grows when open -->
    <div id="acc-call-wrap">
      <div class="acc-section-header" onclick="toggleCallAcc()">
        <span class="acc-chevron" id="acc-call-chev">▸</span>
        <span class="acc-section-title">Selected Call</span>
        <span class="acc-section-meta" id="acc-call-meta"></span>
      </div>
      <div class="acc-body" id="acc-call-body">
        <span class="acc-empty">Click a call to inspect it</span>
      </div>
    </div>
    <!-- Species accordion — fills remaining panel space; inline accordion per row -->
    <div id="acc-species">
      <div id="acc-sp-scroll"></div>
    </div>
  </div>
</div>

<script>
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
  contourAlpha: 0.55,     // default contour opacity (0–1)
  crossfade: 0,           // 0 = raw spectrogram, 1 = call-isolated view
  flatness:  0,           // 0 = raw, 1 = mic-response-flattened spectrogram
  logScale: 0,            // 0 = linear, 1 = fully logarithmic
  saturation: 1.0,        // CSS saturate() applied to spectrogram tiles (1=full color, 0=grey)
  pickRadius: 20,         // hover/click tolerance: max px from cursor to call bounding box
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
};

// Fixed freq range of the server-rendered tile images (kHz)
let TILE_FREQ_LOW = 13, TILE_FREQ_HIGH = 96;

// ─── Canvas refs ─────────────────────────────────────────────
const canvasWrap = document.getElementById('canvas-wrap');
const canvas     = document.getElementById('mainCanvas');
const ctx        = canvas.getContext('2d');
const ovCanvas   = document.getElementById('overviewCanvas');
const octx       = ovCanvas.getContext('2d');

const YAXIS_W  = 52;   // px for freq axis
const SPEC_H   = () => canvas.height;
const OV_H     = 64;

// ─── Coordinate helpers ──────────────────────────────────────
function tToX(t) {
  return YAXIS_W + (t - S.viewStart) / S.viewDur * (canvas.width - YAXIS_W);
}
function xToT(x) {
  return S.viewStart + (x - YAXIS_W) / (canvas.width - YAXIS_W) * S.viewDur;
}

// Frequency → canvas Y.
// Blends: frac = (1-α)*linFrac + α*logFrac, then y = H*(1-frac)
// This is a direct closed-form computation.
function fToY(f) {
  const lo = S.freqLow, hi = S.freqHigh, a = S.logScale;
  const fc = Math.max(lo + 0.001, Math.min(hi, f));
  const linFrac = (fc - lo) / (hi - lo);
  const logFrac = Math.log(fc / lo) / Math.log(hi / lo);
  return SPEC_H() * (1 - ((1 - a) * linFrac + a * logFrac));
}

// Canvas Y → frequency. Inverts fToY via binary search (only ~40 iters, negligible).
function yToF(y) {
  const lo = S.freqLow, hi = S.freqHigh, a = S.logScale;
  const frac = 1 - y / SPEC_H();
  if (a === 0) return lo + frac * (hi - lo);
  if (a === 1) return lo * Math.exp(frac * Math.log(hi / lo));
  let fLo = lo, fHi = hi;
  for (let i = 0; i < 40; i++) {
    const mid = (fLo + fHi) / 2;
    const linF = (mid - lo) / (hi - lo);
    const logF = Math.log(mid / lo) / Math.log(hi / lo);
    ((1 - a) * linF + a * logF < frac) ? fLo = mid : fHi = mid;
  }
  return (fLo + fHi) / 2;
}


// ─── Tile warp cache ──────────────────────────────────────────
// Pre-warp each tile image into a detached HTMLCanvasElement at the current
// canvas height so each render only needs ONE drawImage per tile instead of
// ~200 band slices.  We use a plain <canvas> (not OffscreenCanvas) because
// Firefox does not GPU-accelerate OffscreenCanvas on the main thread, making
// drawImage from it as slow as software rendering.  Detached HTMLCanvasElements
// are hardware-accelerated in Chrome, Safari, and Firefox alike.
function _getWarpedTile(idx, img, H, warpCache = S.tileWarpCache) {
  const key = `${idx}-${H}-${S.logScale.toFixed(3)}-${S.freqLow.toFixed(1)}-${S.freqHigh.toFixed(1)}`;
  if (warpCache.has(key)) return warpCache.get(key);

  const osc  = document.createElement('canvas');
  osc.width  = img.naturalWidth;
  osc.height = H;
  const oc2  = osc.getContext('2d');

  const BANDS = Math.ceil(H / 2);
  for (let b = 0; b < BANDS; b++) {
    const cy  = b * 2;
    const f0  = yToF(cy);      // freq at top of this 2-px band
    const f1  = yToF(cy + 2); // freq at bottom
    const ty0 = (TILE_FREQ_HIGH - f0) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
    const ty1 = (TILE_FREQ_HIGH - f1) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
    if (ty0 < 0 || ty1 > 1.01 || ty1 <= ty0) continue;
    const imgY0 = ty0 * img.naturalHeight;
    const imgH  = Math.max(0.5, (ty1 - ty0) * img.naturalHeight);
    oc2.drawImage(img, 0, imgY0, img.naturalWidth, imgH,
                       0, cy,   img.naturalWidth, 2);
  }
  warpCache.set(key, osc);
  return osc;
}

// ─── Tile loading ─────────────────────────────────────────────
function loadTile(idx) {
  if (S.tileImgs.has(idx)) return;
  const img = new Image();
  S.tileImgs.set(idx, img);
  S.tileReady.set(idx, false);
  img.onload = () => {
    S.tileReady.set(idx, true);
    // Pre-warp immediately so the next render() only needs 1 drawImage per tile.
    // Doing it here (async, after network load) keeps the render loop cheap.
    const H = SPEC_H();
    if (H > 0) _getWarpedTile(idx, img, H);
    scheduleRender();
  };
  img.src = `/api/tile/${idx}?v=${S.tileVersion}`;
}

function loadMaskTile(idx) {
  if (S.maskTileImgs.has(idx)) return;
  const img = new Image();
  S.maskTileImgs.set(idx, img);
  S.maskTileReady.set(idx, false);
  img.onload = () => {
    S.maskTileReady.set(idx, true);
    const H = SPEC_H();
    if (H > 0) _getWarpedTile(idx, img, H, S.maskTileWarpCache);
    scheduleRender();
  };
  img.onerror = () => {
    S.maskTileImgs.delete(idx);
    S.maskTileReady.delete(idx);
    setTimeout(() => loadMaskTile(idx), 3000);
  };
  img.src = `/api/tile_mask/${idx}?v=${S.tileVersion}`;
}

function loadFlatTile(idx) {
  if (S.flatTileImgs.has(idx)) return;
  const img = new Image();
  S.flatTileImgs.set(idx, img);
  S.flatTileReady.set(idx, false);
  img.onload = () => {
    S.flatTileReady.set(idx, true);
    const H = SPEC_H();
    if (H > 0) _getWarpedTile(idx, img, H, S.flatTileWarpCache);
    scheduleRender();
  };
  img.src = `/api/tile_flat/${idx}?v=${S.tileVersion}`;
}

function ensureTiles() {
  const viewEnd = S.viewStart + S.viewDur;
  const first   = Math.max(0, Math.floor(S.viewStart / S.tileDur) - 1);
  const last    = Math.min(S.nTiles - 1, Math.ceil(viewEnd / S.tileDur));
  for (let i = first; i <= last; i++) {
    loadTile(i);
    if (S.crossfade > 0) loadMaskTile(i);
    if (S.flatness  > 0) loadFlatTile(i);
  }
  // Prefetch neighbours
  if (first > 0) {
    loadTile(first - 1);
    if (S.crossfade > 0) loadMaskTile(first - 1);
    if (S.flatness  > 0) loadFlatTile(first - 1);
  }
  if (last < S.nTiles - 1) {
    loadTile(last + 1);
    if (S.crossfade > 0) loadMaskTile(last + 1);
    if (S.flatness  > 0) loadFlatTile(last + 1);
  }
}

// ─── Rendering ───────────────────────────────────────────────
function scheduleRender() {
  if (S.renderPending) return;
  S.renderPending = true;
  requestAnimationFrame(() => { S.renderPending = false; render(); });
}

// ─── Classifier toggle ────────────────────────────────────────
// Copies the selected classifier's fields into the live c.species / c.color /
// c.short / c.conf fields so all rendering code works without modification.
function setClassifier(which) {
  S.classifier = which;
  const useV2 = (which === 'v2');
  for (const c of S.calls) {
    if (useV2) {
      c.species = c.species_v2 ?? c.species;
      c.conf    = c.conf_v2    ?? c.conf;
      c.color   = c.color_v2   ?? c.color;
      c.short   = c.short_v2   ?? c.short;
    } else {
      c.species = c.species_v1 ?? c.species;
      c.conf    = c.conf_v1    ?? c.conf;
      c.color   = c.color_v1   ?? c.color;
      c.short   = c.short_v1   ?? c.short;
    }
  }
  // Update toggle button appearance
  document.getElementById('clf-v1').classList.toggle('clf-active', !useV2);
  document.getElementById('clf-v2').classList.toggle('clf-active',  useV2);
  S.hiddenSpecies.clear();   // reset hide-state — species set may have changed
  buildLegend(S.colors);
  scheduleRender();
}

function render() {
  ensureTiles();
  const W = canvas.width, H = SPEC_H(), specW = W - YAXIS_W;
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);

  // ── Spectrogram tiles (frequency-warped) ──
  const viewEnd  = S.viewStart + S.viewDur;
  const first = Math.max(0, Math.floor(S.viewStart / S.tileDur));
  const last  = Math.min(S.nTiles - 1, Math.ceil(viewEnd / S.tileDur));

  // Apply saturation filter to all tile draws; reset before contours/axes so
  // those remain fully saturated regardless of the spectrogram setting.
  if (S.saturation < 1) ctx.filter = `saturate(${S.saturation})`;

  for (let i = first; i <= last; i++) {
    const img = S.tileImgs.get(i);
    const tS  = i * S.tileDur;
    const tE  = Math.min((i + 1) * S.tileDur, S.duration);
    const tileDurActual = tE - tS;

    if (!img || !S.tileReady.get(i)) {
      const x1 = Math.max(YAXIS_W, tToX(tS));
      const x2 = Math.min(W, tToX(tE));
      ctx.fillStyle = '#151515';
      ctx.fillRect(x1, 0, x2 - x1, H);
      ctx.fillStyle = '#2a2a2a';
      ctx.font = '11px monospace';
      ctx.fillText('loading…', x1 + 4, H / 2);
      continue;
    }

    // Source X slice (time axis, always linear in the tile image)
    const srcX0 = Math.max(0, (S.viewStart - tS) / tileDurActual * img.naturalWidth);
    const srcX1 = Math.min(img.naturalWidth, (viewEnd - tS) / tileDurActual * img.naturalWidth);
    if (srcX1 <= srcX0) continue;
    const dstX0 = Math.max(YAXIS_W, tToX(tS));
    const dstX1 = Math.min(W, tToX(tE));
    if (dstX1 <= dstX0) continue;
    const srcW = srcX1 - srcX0, dstW = dstX1 - dstX0;

    // Try the pre-warped tile (1 drawImage instead of ~200)
    const warped = _getWarpedTile(i, img, H);
    if (warped) {
      ctx.drawImage(warped, srcX0, 0, srcW, H, dstX0, 0, dstW, H);
    } else {
      // Fallback: per-band warp (OffscreenCanvas unavailable)
      const BANDS = Math.ceil(H / 2);
      for (let b = 0; b < BANDS; b++) {
        const f0  = yToF(b * 2), f1 = yToF(b * 2 + 2);
        const ty0 = (TILE_FREQ_HIGH - f0) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
        const ty1 = (TILE_FREQ_HIGH - f1) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
        if (ty0 < 0 || ty1 > 1.01 || ty1 <= ty0) continue;
        const imgY0 = ty0 * img.naturalHeight;
        const imgH  = Math.max(0.5, (ty1 - ty0) * img.naturalHeight);
        ctx.drawImage(img, srcX0, imgY0, srcW, imgH, dstX0, b*2, dstW, Math.min(2, H-b*2));
      }
    }

    // ── Flat tile overlay (frequency-compensated, source-over at S.flatness).
    // At flatness=0: invisible (raw). At flatness=1: full flat. Intermediate: blend.
    if (S.flatness > 0) {
      const fImg = S.flatTileImgs.get(i);
      if (fImg && S.flatTileReady.get(i)) {
        ctx.globalAlpha = S.flatness;
        const warpedFlat = _getWarpedTile(i, fImg, H, S.flatTileWarpCache);
        if (warpedFlat) {
          ctx.drawImage(warpedFlat, srcX0, 0, srcW, H, dstX0, 0, dstW, H);
        } else {
          const BANDS = Math.ceil(H / 2);
          for (let b = 0; b < BANDS; b++) {
            const f0  = yToF(b * 2), f1 = yToF(b * 2 + 2);
            const ty0 = (TILE_FREQ_HIGH - f0) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
            const ty1 = (TILE_FREQ_HIGH - f1) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
            if (ty0 < 0 || ty1 > 1.01 || ty1 <= ty0) continue;
            const imgY0 = ty0 * fImg.naturalHeight;
            const imgH  = Math.max(0.5, (ty1 - ty0) * fImg.naturalHeight);
            ctx.drawImage(fImg, srcX0, imgY0, srcW, imgH, dstX0, b*2, dstW, Math.min(2, H-b*2));
          }
        }
        ctx.globalAlpha = 1;
      } else {
        loadFlatTile(i);
      }
    }

    // ── Mask overlay (R=G=B=0, A=(1−mask)×255) at crossfade α.
    // Applied after flat/raw blend: background dims, call pixels unaffected.
    if (S.crossfade > 0) {
      const mImg = S.maskTileImgs.get(i);
      if (mImg && S.maskTileReady.get(i)) {
        ctx.globalAlpha = S.crossfade;
        const warpedMask = _getWarpedTile(i, mImg, H, S.maskTileWarpCache);
        if (warpedMask) {
          ctx.drawImage(warpedMask, srcX0, 0, srcW, H, dstX0, 0, dstW, H);
        } else {
          const BANDS = Math.ceil(H / 2);
          for (let b = 0; b < BANDS; b++) {
            const f0  = yToF(b * 2), f1 = yToF(b * 2 + 2);
            const ty0 = (TILE_FREQ_HIGH - f0) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
            const ty1 = (TILE_FREQ_HIGH - f1) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
            if (ty0 < 0 || ty1 > 1.01 || ty1 <= ty0) continue;
            const imgY0 = ty0 * mImg.naturalHeight;
            const imgH  = Math.max(0.5, (ty1 - ty0) * mImg.naturalHeight);
            ctx.drawImage(mImg, srcX0, imgY0, srcW, imgH, dstX0, b*2, dstW, Math.min(2, H-b*2));
          }
        }
        ctx.globalAlpha = 1;
      } else {
        loadMaskTile(i);
      }
    }
  }

  // Reset filter before drawing annotations so contours/axes stay fully saturated
  ctx.filter = 'none';

  // ── Grid lines (log-aware) ──
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth   = 1;
  const gridTicks = [14,16,18,20,25,30,35,40,50,60,70,80,90,96];
  for (const f of gridTicks) {
    if (f <= S.freqLow || f >= S.freqHigh) continue;
    const y = Math.round(fToY(f)) + 0.5;
    ctx.beginPath(); ctx.moveTo(YAXIS_W, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // ── Call overlays (always drawn: boxes appear on hover/select even when S.showBoxes is off) ──
  if (S.showBoxes || S.showContour || S.hoveredCall || S.selectedCall)
    drawCallOverlays(specW, H, viewEnd);

  // ── Freq axis ──
  drawFreqAxis(W, H);

  // ── Call density rug (above time axis) ──
  if (S.calls.length) drawCallRug(W, H, specW);

  // ── Time axis ──
  drawTimeAxis(W, H, specW);

  // ── Ruler (must be before crosshairs so crosshairs render on top) ──
  drawRuler(W, H);

  // ── Crosshairs ──
  drawCrosshairs(W, H);

  // ── Overview ──
  drawOverview();

  // ── Time display ──
  document.getElementById('time-display').innerHTML =
    `View: ${fmt(S.viewStart)} – ${fmt(S.viewStart + S.viewDur)}<br>Duration: ${S.viewDur.toFixed(1)}s`;
}

// Zoom-dependent base line width: thin when zoomed out (many calls, small pixels),
// progressively thicker when zoomed in.  Log2 of pixels-per-second keeps the
// growth gradual — doubles roughly every time the zoom doubles.
function _baseLineW() {
  const pxPerSec = (canvas.width - YAXIS_W) / Math.max(S.viewDur, 0.01);
  // ~0.8 px at 30 s view · ~1.5 px at 10 s · ~2.5 px at 3 s · ~3.5 px at 0.5 s
  return Math.max(0.5, Math.min(3.5, 1.0 + Math.log2(pxPerSec / 50) * 0.5));
}

function drawCall(c, specW, H) {
  const sel  = c === S.selectedCall;
  const hov  = c === S.hoveredCall;
  const col  = c.color;
  const base = _baseLineW();

  const x0 = tToX(c.t0),  x1 = tToX(c.t1);
  const y0 = fToY(c.Fmax), y1 = fToY(c.Fmin);
  const bw = x1 - x0,     bh = y1 - y0;

  // Bounding box: always visible when S.showBoxes is checked; otherwise only
  // on hover or selection so the spectrogram stays uncluttered by default.
  if (S.showBoxes || sel || hov) {
    ctx.globalAlpha = sel ? 0.45 : (hov ? 0.35 : 0.18);
    ctx.fillStyle   = col;
    ctx.fillRect(x0, y0, bw, bh);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = sel ? '#ffffff' : col;
    ctx.lineWidth   = sel ? Math.min(5, base * 2.2) : (hov ? Math.min(4, base * 1.6) : Math.max(0.5, base * 0.8));
    ctx.strokeRect(x0, y0, bw, bh);

    if (bw > 10) {
      ctx.font      = 'bold 10px monospace';
      ctx.fillStyle = sel ? '#ffffff' : col;
      const ly      = y0 > 14 ? y0 - 3 : y0 + bh + 11;
      ctx.fillText(c.short, x0 + 2, ly);
    }
  }

  if (S.showContour && c.contour && c.contour.length > 1) {
    ctx.beginPath();
    // Selected contour turns white to match the box border; hovered and normal
    // keep the species colour.
    ctx.strokeStyle = sel ? '#ffffff' : col;
    ctx.lineWidth   = sel ? Math.min(5, base * 2) : (hov ? Math.min(4, base * 1.6) : base);
    // Contour opacity: user-controlled slider; boosted on hover/select
    ctx.globalAlpha = sel ? Math.min(1, S.contourAlpha * 1.8)
                          : (hov ? Math.min(1, S.contourAlpha * 1.4)
                                 : S.contourAlpha);
    let first = true;
    for (const [ct, cf] of c.contour) {
      const cx = tToX(ct), cy = fToY(cf);
      if (first) { ctx.moveTo(cx, cy); first = false; }
      else        ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    if (sel || hov) {
      const pmid = c.contour[Math.floor(c.contour.length / 2)];
      ctx.beginPath();
      ctx.arc(tToX(pmid[0]), fToY(pmid[1]), Math.max(2, base * 0.9), 0, Math.PI * 2);
      ctx.fillStyle = sel ? '#ffffff' : col;
      ctx.fill();
    }
  }
}

function drawCrosshairs(W, H) {
  // Don't show crosshairs while actively drawing a ruler
  if (S.mouseX < 0 || S.isRuling) return;
  const mx = S.mouseX, my = S.mouseY;
  const t  = xToT(mx);
  const f  = yToF(my);

  ctx.save();
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = 'rgba(255,255,255,0.45)';
  ctx.lineWidth   = 1;
  // Vertical line (time)
  ctx.beginPath(); ctx.moveTo(mx, 0); ctx.lineTo(mx, H); ctx.stroke();
  // Horizontal line (frequency)
  ctx.beginPath(); ctx.moveTo(YAXIS_W, my); ctx.lineTo(W, my); ctx.stroke();
  ctx.setLineDash([]);

  ctx.font = '10px monospace';
  // Time label — just above the bottom time-axis strip (~20px from bottom)
  const tLabel = fmt(t);
  const tlw = ctx.measureText(tLabel).width;
  let tlx = mx + 4;
  if (tlx + tlw + 6 > W) tlx = mx - tlw - 8;
  ctx.fillStyle = 'rgba(0,0,0,0.78)';
  ctx.fillRect(tlx - 2, H - 32, tlw + 6, 14);
  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.fillText(tLabel, tlx, H - 21);

  // Frequency label — just right of the freq-axis column
  const fLabel = f.toFixed(1) + ' kHz';
  const flw = ctx.measureText(fLabel).width;
  let fly = my - 5;
  if (fly < 12) fly = my + 14;
  ctx.fillStyle = 'rgba(0,0,0,0.78)';
  ctx.fillRect(YAXIS_W + 5, fly - 12, flw + 6, 14);
  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.fillText(fLabel, YAXIS_W + 7, fly);

  ctx.restore();
}

function drawRuler(W, H) {
  if (!S.isRuling && !S.rulerFixed) return;
  const moved = Math.hypot(S.rulerX1 - S.rulerX0, S.rulerY1 - S.rulerY0);
  if (moved < 3) return;

  const x0 = Math.min(S.rulerX0, S.rulerX1);
  const x1 = Math.max(S.rulerX0, S.rulerX1);
  const y0 = Math.min(S.rulerY0, S.rulerY1);
  const y1 = Math.max(S.rulerY0, S.rulerY1);
  const rW = x1 - x0, rH = y1 - y0;

  const t0  = xToT(x0), t1 = xToT(x1);
  const fHi = yToF(y0), fLo = yToF(y1);   // y0=top=higher freq
  const dtMs = (t1 - t0) * 1000;
  const df   = fHi - fLo;

  ctx.save();

  // Translucent fill
  ctx.fillStyle   = 'rgba(242,142,43,0.08)';
  ctx.fillRect(x0, y0, rW, rH);

  // Dashed orange border
  ctx.setLineDash([5, 3]);
  ctx.strokeStyle = '#f28e2b';
  ctx.lineWidth   = 1.5;
  ctx.globalAlpha = 0.9;
  ctx.strokeRect(x0, y0, rW, rH);
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  // Corner dots
  ctx.fillStyle = '#f28e2b';
  for (const [cx, cy] of [[x0,y0],[x1,y0],[x0,y1],[x1,y1]]) {
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI*2); ctx.fill();
  }

  // Measurement label
  const dtStr = dtMs >= 1000 ? (dtMs/1000).toFixed(3)+'s' : dtMs.toFixed(1)+'ms';
  const lines = [
    `Δt   ${dtStr}`,
    `Δf   ${df.toFixed(1)} kHz`,
    `t   ${fmt(t0)} → ${fmt(t1)}`,
    `f   ${fLo.toFixed(1)} → ${fHi.toFixed(1)} kHz`,
  ];
  ctx.font = '11px monospace';
  const lw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 14;
  const lh = lines.length * 16 + 10;

  // Prefer label to the right of the box; fall back left if it would clip
  let lx = x1 + 8, ly = y0;
  if (lx + lw > W - 4)  lx = x0 - lw - 8;
  if (lx < YAXIS_W + 4) lx = x0 + 4;
  if (ly + lh > H - 4)  ly = y1 - lh;
  if (ly < 2)            ly = 2;

  ctx.fillStyle   = 'rgba(10,10,10,0.88)';
  ctx.fillRect(lx, ly, lw, lh);
  ctx.strokeStyle = '#f28e2b';
  ctx.lineWidth   = 1;
  ctx.strokeRect(lx, ly, lw, lh);
  ctx.fillStyle   = '#f28e2b';
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], lx + 7, ly + 16 + i * 16);
  }

  ctx.restore();
}

// Binary search: first index where calls[i].t0 >= target
function callsLowerBound(target) {
  let lo = 0, hi = S.calls.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (S.calls[mid].t0 < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function drawCallOverlays(specW, H, viewEnd) {
  // S.calls is sorted by t0.  Use binary search to skip the bulk of the array.
  // Back up 0.3 s from viewStart to catch calls that started just before the window.
  const startIdx = Math.max(0, callsLowerBound(S.viewStart - 0.3));
  const visible  = [];
  for (let i = startIdx; i < S.calls.length; i++) {
    const c = S.calls[i];
    if (c.t0 >= viewEnd) break;
    if (c.t1 > S.viewStart && !S.hiddenSpecies.has(c.species)) visible.push(c);
  }
  if (!visible.length) return;

  const SPARSE_THRESHOLD = 400;
  if (visible.length <= SPARSE_THRESHOLD) {
    for (const c of visible) drawCall(c, specW, H);
    return;
  }

  // Dense view: batched rects, then repaint selected/hovered individually on top
  drawCallsBatched(visible, specW, H);

  // O(1) visibility check via time range rather than array scan
  const sel = S.selectedCall;
  if (sel && sel.t0 < viewEnd && sel.t1 > S.viewStart)
    drawCall(sel, specW, H);
  const hov = S.hoveredCall;
  if (hov && hov !== sel && hov.t0 < viewEnd && hov.t1 > S.viewStart)
    drawCall(hov, specW, H);
}

function drawCallsBatched(visible, specW, H) {
  // Zoomed-out view: draw each call as a vertical tick at its centre time,
  // spanning Fmin→Fmax.  Tick width uses the same zoom-scaled _baseLineW() as
  // the individual contour renderer so there is no visible jump at the
  // sparse/dense threshold.
  const tickW = Math.max(1, Math.round(_baseLineW()));
  const half  = Math.floor(tickW / 2);

  const bySpecies = {};
  for (const c of visible) {
    if (!bySpecies[c.species]) bySpecies[c.species] = { col: c.color, calls: [] };
    bySpecies[c.species].calls.push(c);
  }

  for (const { col, calls } of Object.values(bySpecies)) {
    ctx.fillStyle   = col;
    ctx.globalAlpha = Math.min(1, S.contourAlpha * 1.3);
    ctx.beginPath();
    for (const c of calls) {
      const xc = Math.round(tToX((c.t0 + c.t1) / 2));
      const y0 = Math.floor(fToY(c.Fmax));
      const y1 = Math.ceil(fToY(c.Fmin));
      ctx.rect(xc - half, y0, tickW, Math.max(tickW, y1 - y0));
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawFreqAxis(W, H) {
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, YAXIS_W, H);
  ctx.strokeStyle = '#2a2a2a';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(YAXIS_W, 0); ctx.lineTo(YAXIS_W, H); ctx.stroke();

  ctx.fillStyle = '#777';
  ctx.font      = '10px monospace';
  ctx.textAlign = 'right';
  const ticks = [13,14,15,16,18,20,25,30,35,40,50,60,70,80,90,96];
  for (const f of ticks) {
    if (f < S.freqLow || f > S.freqHigh) continue;
    const y = Math.round(fToY(f));
    if (y < 0 || y > H) continue;
    ctx.fillStyle = '#666';
    ctx.fillText(`${f}k`, YAXIS_W - 5, y + 3);
    ctx.strokeStyle = '#2a2a2a';
    ctx.beginPath(); ctx.moveTo(YAXIS_W - 3, y + 0.5); ctx.lineTo(YAXIS_W, y + 0.5); ctx.stroke();
  }
  // Rotated label
  ctx.save();
  ctx.translate(10, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#444';
  ctx.font      = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Frequency (Hz)', 0, 0);
  ctx.restore();
  ctx.textAlign = 'left';

}

// Compact call-density rug drawn just above the time axis.
// Every visible call → 1-px vertical tick in its species colour.
// Gives an immediate sense of density and species composition when zoomed out.
const RUG_H = 11;
function drawCallRug(W, H, specW) {
  const rugTop = H - 14 - RUG_H - 2;   // 14px = time-axis height, 2px gap
  ctx.fillStyle = 'rgba(8,8,8,0.82)';
  ctx.fillRect(YAXIS_W, rugTop, specW, RUG_H);

  const viewEnd  = S.viewStart + S.viewDur;
  const startIdx = callsLowerBound(S.viewStart - 0.3);

  // Group by species for batched drawing
  const bySpecies = {};
  for (let i = startIdx; i < S.calls.length; i++) {
    const c = S.calls[i];
    if (c.t0 > viewEnd) break;
    if (S.hiddenSpecies.has(c.species)) continue;
    if (!bySpecies[c.species]) bySpecies[c.species] = { col: c.color, xs: [] };
    bySpecies[c.species].xs.push(Math.round(tToX((c.t0 + c.t1) / 2)));
  }

  for (const { col, xs } of Object.values(bySpecies)) {
    ctx.fillStyle   = col;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (const x of xs) ctx.rect(x, rugTop + 1, 1, RUG_H - 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Hairline border
  ctx.strokeStyle = '#1e1e1e';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(YAXIS_W, rugTop + 0.5);
  ctx.lineTo(YAXIS_W + specW, rugTop + 0.5);
  ctx.stroke();
}

function drawTimeAxis(W, H, specW) {
  const viewEnd = S.viewStart + S.viewDur;
  // Choose a sensible tick interval
  const targets = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60];
  const minPx   = 60;
  let interval  = targets.find(v => v / S.viewDur * specW >= minPx) || 60;
  const t0 = Math.ceil(S.viewStart / interval) * interval;
  ctx.fillStyle   = '#555';
  ctx.font        = '10px monospace';
  ctx.strokeStyle = '#2a2a2a';
  ctx.lineWidth   = 1;
  for (let t = t0; t <= viewEnd; t += interval) {
    const x = Math.round(tToX(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, H - 14); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillText(fmt(t), x + 2, H - 3);
  }
}

// Overview coordinate helpers (use S.ovStart/ovDur, not full S.duration)
function ovTX(t)  { return (t - S.ovStart) / S.ovDur * ovCanvas.width; }
function ovXT(ox) { return S.ovStart + ox / ovCanvas.width * S.ovDur; }

function drawOverview() {
  const OW = ovCanvas.width, OH = OV_H;
  const ovD = S.ovDur || S.duration || 1;   // guard against 0 before init
  octx.clearRect(0, 0, OW, OH);
  octx.fillStyle = '#0d0d0d';
  octx.fillRect(0, 0, OW, OH);

  // Individual call dots — y-position encodes peak frequency
  for (const c of S.calls) {
    const x = ovTX(c.t0);
    const w = Math.max(1, (c.t1 - c.t0) / ovD * OW);
    if (x + w < 0 || x > OW) continue;
    const fy = c.Fpeak >= S.freqLow && c.Fpeak <= S.freqHigh
               ? OH * (1 - (c.Fpeak - S.freqLow) / (S.freqHigh - S.freqLow))
               : OH / 2;
    octx.fillStyle   = c.color;
    octx.globalAlpha = 0.7;
    octx.fillRect(x, Math.max(0, fy - 2), w, 4);
  }
  octx.globalAlpha = 1;

  // Viewport box
  const vx0 = ovTX(S.viewStart);
  const vx1 = ovTX(S.viewStart + S.viewDur);
  const vw  = Math.max(2, vx1 - vx0);
  octx.fillStyle = 'rgba(255,255,255,0.07)';
  octx.fillRect(vx0, 0, vw, OH);
  octx.strokeStyle = 'rgba(255,255,255,0.28)';
  octx.lineWidth = 1;
  octx.strokeRect(vx0, 0, vw, OH);

  // Draggable edge handles — brighter vertical bars
  const hw = 4;
  octx.fillStyle = _ovDrag ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.45)';
  octx.fillRect(vx0,      0, hw, OH);
  octx.fillRect(vx1 - hw, 0, hw, OH);

  // When zoomed: draw a thin full-recording ruler at the bottom (3 px strip)
  const zoomed = ovD < S.duration * 0.99;
  if (zoomed) {
    const rH = 3, rY = OH - rH;
    octx.fillStyle = '#1e1e1e';
    octx.fillRect(0, rY, OW, rH);
    // overview window within full recording
    const rx0 = S.ovStart / S.duration * OW;
    const rx1 = (S.ovStart + ovD) / S.duration * OW;
    octx.fillStyle = 'rgba(255,255,255,0.18)';
    octx.fillRect(rx0, rY, Math.max(2, rx1 - rx0), rH);
    // viewport within full recording
    const rvx0 = S.viewStart / S.duration * OW;
    const rvx1 = (S.viewStart + S.viewDur) / S.duration * OW;
    octx.fillStyle = 'rgba(255,255,255,0.5)';
    octx.fillRect(rvx0, rY, Math.max(1, rvx1 - rvx0), rH);
  }

  // Border
  octx.strokeStyle = zoomed ? '#3a3a3a' : '#222';
  octx.lineWidth   = 1;
  octx.strokeRect(0, 0, OW, OH);
}

// ─── Resize ──────────────────────────────────────────────────
function resize() {
  const cr = canvas.getBoundingClientRect();
  canvas.width  = Math.max(1, Math.round(cr.width));
  canvas.height = Math.max(1, Math.round(cr.height));
  S.tileWarpCache.clear();      // height changed → pre-warped tiles are stale
  S.maskTileWarpCache.clear();
  S.flatTileWarpCache.clear();
  ovCanvas.width  = document.getElementById('overview-wrap').getBoundingClientRect().width;
  ovCanvas.height = OV_H;
  updateScrollbar();
  scheduleRender();
}

// ─── Events ──────────────────────────────────────────────────
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  // Normalise deltaY across deltaMode units, then cap at ±200 px-equivalents
  // so a single big trackpad flick doesn't teleport the view.
  let delta = e.deltaY;
  if (e.deltaMode === 1) delta *= 20;   // line mode → px
  if (e.deltaMode === 2) delta *= 400;  // page mode → px
  delta = Math.sign(delta) * Math.min(Math.abs(delta), 200);
  // Exponential zoom: 1.0025 per pixel gives ~1.65× per 200-px swipe — smooth.
  const factor  = Math.pow(1.0025, delta);
  const rect    = canvas.getBoundingClientRect();
  const relX    = (e.clientX - rect.left - YAXIS_W) / (canvas.width - YAXIS_W);
  const tCursor = S.viewStart + relX * S.viewDur;
  S.viewDur     = Math.max(0.5, Math.min(S.duration, S.viewDur * factor));
  S.viewStart   = Math.max(0, Math.min(S.duration - S.viewDur, tCursor - relX * S.viewDur));
  scheduleRender();
}, { passive: false });

canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  if (mx < YAXIS_W) return;       // click on freq-axis column: ignore
  S.isRuling   = true;
  S.rulerFixed = false;
  S.rulerX0 = S.rulerX1 = mx;
  S.rulerY0 = S.rulerY1 = Math.max(0, Math.min(canvas.height, my));
});

window.addEventListener('mousemove', e => {
  // Overview transport drag
  if (_ovDrag) {
    const OW  = ovCanvas.width;
    const dx  = e.clientX - _ovX0;
    const dt  = dx / OW * (S.ovDur || S.duration);  // time delta scaled to overview zoom
    const MIN = 0.5;
    if (_ovDrag === 'pan') {
      S.viewStart = Math.max(0, Math.min(S.duration - _ovVD0, _ovVS0 + dt));
      S.viewDur   = _ovVD0;
    } else if (_ovDrag === 'left') {
      const viewEnd  = _ovVS0 + _ovVD0;
      const newStart = Math.max(0, Math.min(viewEnd - MIN, _ovVS0 + dt));
      S.viewStart    = newStart;
      S.viewDur      = viewEnd - newStart;
    } else if (_ovDrag === 'right') {
      const newEnd = Math.max(_ovVS0 + MIN, Math.min(S.duration, _ovVS0 + _ovVD0 + dt));
      S.viewStart  = _ovVS0;
      S.viewDur    = newEnd - _ovVS0;
    }
    scheduleRender(); return;
  }

  // Ruler rubber-band drag (main canvas)
  if (S.isRuling) {
    const rect = canvas.getBoundingClientRect();
    S.rulerX1 = Math.max(YAXIS_W,     Math.min(canvas.width,  e.clientX - rect.left));
    S.rulerY1 = Math.max(0,            Math.min(canvas.height, e.clientY - rect.top));
    scheduleRender();
  }
  updateHover(e);
});

window.addEventListener('mouseup', e => {
  if (_ovDrag) {
    _ovDrag = null;
    ovCanvas.style.cursor = 'default';
    return;
  }
  if (!S.isRuling) return;
  S.isRuling = false;
  const moved = Math.hypot(S.rulerX1 - S.rulerX0, S.rulerY1 - S.rulerY0);
  if (moved < 5) {
    S.rulerFixed = false;   // tiny drag → treat as click, discard ruler
    handleClick(e);
  } else {
    S.rulerFixed = true;    // real drag → leave ruler on screen
  }
  canvas.style.cursor = S.hoveredCall ? 'pointer' : 'crosshair';
  scheduleRender();
});

canvas.addEventListener('mouseleave', () => {
  S.mouseX = -1; S.mouseY = -1;
  if (S.hoveredCall) { S.hoveredCall = null; hideTooltip(); }
  scheduleRender();
});

// ─── Hit-testing ─────────────────────────────────────────────
// Returns the call whose bounding box is closest to (mx, my) in canvas pixels,
// as long as that distance is ≤ S.pickRadius.  Uses binary search on t0 to
// avoid scanning all 10k+ calls on every mousemove.
function _hitTest(mx, my) {
  const N      = S.pickRadius;
  const specW  = canvas.width - YAXIS_W;
  // Convert N pixels → seconds so we can bound the binary search window.
  const tol_t  = N * S.viewDur / Math.max(specW, 1);
  const t      = xToT(mx);

  let found     = null;
  let foundDist = N + 1;   // one beyond threshold — any dist ≤ N beats this

  // Start a little before (t - tol_t) to catch long calls that started earlier.
  const si = Math.max(0, callsLowerBound(t - tol_t - 0.15));
  for (let i = si; i < S.calls.length; i++) {
    const c = S.calls[i];
    // c.t0 in pixel space is > mx + N → can't be within N px → stop.
    if (c.t0 > t + tol_t + 0.01) break;
    if (S.hiddenSpecies.has(c.species)) continue;

    // Bounding box in canvas pixels
    const cx0 = tToX(c.t0), cx1 = tToX(c.t1);
    const cy0 = fToY(c.Fmax), cy1 = fToY(c.Fmin);  // y0=top (high freq)

    // Distance from point to axis-aligned rect: 0 if inside, else Euclidean
    // to nearest edge.  max(a, 0, b) = positive part of the outside gap.
    const dx   = Math.max(cx0 - mx, 0, mx - cx1);
    const dy   = Math.max(cy0 - my, 0, my - cy1);
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < foundDist) { found = c; foundDist = dist; }
  }
  return found;
}

function updateHover(e) {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;

  if (mx < YAXIS_W || mx > canvas.width || my < 0 || my > SPEC_H()) {
    S.mouseX = -1; S.mouseY = -1;
    if (S.hoveredCall) { S.hoveredCall = null; hideTooltip(); }
    scheduleRender();
    return;
  }

  S.mouseX = mx; S.mouseY = my;

  const found = _hitTest(mx, my);
  if (found !== S.hoveredCall) {
    S.hoveredCall = found;
    if (!S.isRuling) canvas.style.cursor = found ? 'pointer' : 'crosshair';
    if (found) showTooltip(found, e.clientX, e.clientY);
    else hideTooltip();
  }
  scheduleRender();
}

function handleClick(e) {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;
  const found = _hitTest(mx, my);
  S.selectedCall = (found === S.selectedCall) ? null : found;
  renderDetail(S.selectedCall);
  scheduleRender();
}

// ─── Overview transport drag ──────────────────────────────────
// All positions are in the overview's own fixed coordinate system
// (ox / OW * duration = time), fully independent of viewStart/viewDur.
let _ovDrag = null;   // 'left' | 'right' | 'pan' | 'jump' | null
let _ovX0 = 0, _ovVS0 = 0, _ovVD0 = 0;
const OV_EDGE_PX = 7;  // px grab zone for each edge handle

function ovHitTest(ox) {
  const vx0 = ovTX(S.viewStart);
  const vx1 = ovTX(S.viewStart + S.viewDur);
  if (Math.abs(ox - vx0) <= OV_EDGE_PX) return 'left';
  if (Math.abs(ox - vx1) <= OV_EDGE_PX) return 'right';
  if (ox > vx0 && ox < vx1)             return 'pan';
  return 'jump';
}

ovCanvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  e.preventDefault();
  const ox  = e.clientX - ovCanvas.getBoundingClientRect().left;
  _ovDrag   = ovHitTest(ox);
  _ovX0     = e.clientX;
  _ovVS0    = S.viewStart;
  _ovVD0    = S.viewDur;
  if (_ovDrag === 'jump') {
    const t = ovXT(ox);
    S.viewStart = Math.max(0, Math.min(S.duration - S.viewDur, t - S.viewDur / 2));
    _ovDrag = 'pan';
    _ovVS0  = S.viewStart;
    scheduleRender();
  }
  ovCanvas.style.cursor = _ovDrag === 'pan' ? 'grabbing' : 'ew-resize';
});

ovCanvas.addEventListener('mousemove', e => {
  if (_ovDrag) return;  // cursor already set
  const ox  = e.clientX - ovCanvas.getBoundingClientRect().left;
  const hit = ovHitTest(ox);
  ovCanvas.style.cursor = (hit === 'left' || hit === 'right') ? 'ew-resize'
                        : hit === 'pan' ? 'grab' : 'default';
});

// Keyboard
window.addEventListener('keydown', e => {
  const step = S.viewDur * 0.25;
  if (e.key === 'ArrowRight') S.viewStart = Math.min(S.duration - S.viewDur, S.viewStart + step);
  if (e.key === 'ArrowLeft')  S.viewStart = Math.max(0, S.viewStart - step);
  if (e.key === '+'||e.key==='=') zoomBy(0.7);
  if (e.key === '-')           zoomBy(1.4);
  if (e.key === 'Escape')      { S.rulerFixed = false; }
  scheduleRender();
});

// ─── Overview: scroll-wheel zoom + double-click reset ────────────────
ovCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  let delta = e.deltaY;
  if (e.deltaMode === 1) delta *= 20;
  if (e.deltaMode === 2) delta *= 400;
  delta = Math.sign(delta) * Math.min(Math.abs(delta), 200);
  const factor  = Math.pow(1.0025, delta);
  const ox      = e.clientX - ovCanvas.getBoundingClientRect().left;
  const tCursor = ovXT(ox);
  const relX    = ox / ovCanvas.width;
  S.ovDur   = Math.max(2.0, Math.min(S.duration, (S.ovDur || S.duration) * factor));
  S.ovStart = Math.max(0, Math.min(S.duration - S.ovDur, tCursor - relX * S.ovDur));
  scheduleRender();
}, { passive: false });

ovCanvas.addEventListener('dblclick', () => {
  // Double-click resets the overview zoom to the full recording
  S.ovStart = 0;
  S.ovDur   = S.duration;
  scheduleRender();
});

document.getElementById('btn-zoom-in').onclick  = () => zoomBy(0.6);
document.getElementById('btn-zoom-out').onclick = () => zoomBy(1.6);
document.getElementById('btn-fit').onclick      = () => { S.viewStart = 0; S.viewDur = S.duration; scheduleRender(); };
document.getElementById('chk-contour').onchange = e => { S.showContour = e.target.checked; scheduleRender(); };
document.getElementById('chk-boxes').onchange   = e => { S.showBoxes   = e.target.checked; scheduleRender(); };
document.getElementById('slider-contour-alpha').oninput = e => {
  S.contourAlpha = e.target.value / 100;
  document.getElementById('contour-alpha-val').textContent = e.target.value + '%';
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-crossfade').oninput = e => {
  const wasZero = S.crossfade === 0;
  S.crossfade = e.target.value / 100;
  if (wasZero && S.crossfade > 0) ensureTiles();
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-flatness').oninput = e => {
  const wasZero = S.flatness === 0;
  S.flatness = e.target.value / 100;
  if (wasZero && S.flatness > 0) ensureTiles();
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-log').oninput = e => {
  S.logScale = e.target.value / 100;
  S.tileWarpCache.clear();
  S.maskTileWarpCache.clear();
  S.flatTileWarpCache.clear();
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-sat').oninput = e => {
  S.saturation = e.target.value / 100;
  updateTrack(e.target);
  scheduleRender();
};
document.getElementById('slider-pick-radius').oninput = e => {
  S.pickRadius = parseInt(e.target.value);
  document.getElementById('pick-radius-val').textContent = e.target.value;
  updateTrack(e.target);
};

// ─── Frequency scrollbar ──────────────────────────────────────
// Coordinate system: y=0 = Nyquist (top), y=trackH = 0 Hz (bottom).
// This is INDEPENDENT of S.freqLow/freqHigh, fixing the feedback-loop bug.
const sbTrack  = document.getElementById('freq-sb-track');
const sbFill   = document.getElementById('freq-sb-fill');
const sbTop    = document.getElementById('freq-sb-top');
const sbBot    = document.getElementById('freq-sb-bot');

let _sbDrag = null;  // null | 'top' | 'bot' | 'pan'
let _sbY0 = 0, _sbHi0 = 0, _sbLo0 = 0;

function sbTrackH() { return sbTrack.getBoundingClientRect().height; }

function updateScrollbar() {
  const h   = sbTrackH();
  if (h === 0) return;
  const ny  = S.nyquist;
  const top = (1 - S.freqHigh / ny) * h;   // y of max-freq handle
  const bot = (1 - S.freqLow  / ny) * h;   // y of min-freq handle
  sbFill.style.top    = top + 'px';
  sbFill.style.height = (bot - top) + 'px';
  sbTop.style.top     = '0px';   // relative to fill
  sbBot.style.top     = '100%';

  // Tick marks — rebuild only if nyquist changed (rare)
  if (!sbTrack._ticked) {
    sbTrack._ticked = true;
    [10,20,30,40,50,60,70,80,90].forEach(f => {
      if (f >= ny) return;
      const d = document.createElement('div');
      d.className = 'freq-sb-tick';
      d.style.top = ((1 - f / ny) * 100) + '%';
      d.title     = f + ' kHz';
      sbTrack.appendChild(d);
    });
  }
}

function sbStartDrag(type, e) {
  e.preventDefault(); e.stopPropagation();
  _sbDrag  = type;
  _sbY0    = e.clientY;
  _sbHi0   = S.freqHigh;
  _sbLo0   = S.freqLow;
}
sbTop.addEventListener('mousedown',  e => sbStartDrag('top', e));
sbBot.addEventListener('mousedown',  e => sbStartDrag('bot', e));
sbFill.addEventListener('mousedown', e => sbStartDrag('pan', e));

window.addEventListener('mousemove', e => {
  if (!_sbDrag) return;
  const h   = sbTrackH();
  const ny  = S.nyquist;
  const dy  = e.clientY - _sbY0;
  const df  = -dy / h * ny;           // upward mouse = higher freq
  const MIN_SPAN = 2;                  // kHz minimum window
  if (_sbDrag === 'top') {
    S.freqHigh = Math.max(_sbLo0 + MIN_SPAN, Math.min(ny, _sbHi0 + df));
  } else if (_sbDrag === 'bot') {
    S.freqLow  = Math.min(_sbHi0 - MIN_SPAN, Math.max(0,  _sbLo0 + df));
  } else {                             // pan: move both, keep span
    const span = _sbHi0 - _sbLo0;
    S.freqHigh = Math.min(ny,    Math.max(span, _sbHi0 + df));
    S.freqLow  = S.freqHigh - span;
  }
  S.tileWarpCache.clear();      // freq range changed → pre-warped tiles are stale
  S.maskTileWarpCache.clear();
  S.flatTileWarpCache.clear();
  updateScrollbar();
  scheduleRender();
});

window.addEventListener('mouseup', () => { _sbDrag = null; });

function zoomBy(factor) {
  const mid   = S.viewStart + S.viewDur / 2;
  S.viewDur   = Math.max(0.5, Math.min(S.duration, S.viewDur * factor));
  S.viewStart = Math.max(0, Math.min(S.duration - S.viewDur, mid - S.viewDur / 2));
}

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
  const body = document.getElementById('acc-call-body');
  const meta = document.getElementById('acc-call-meta');
  if (!c) {
    body.innerHTML = '<span class="acc-empty">Click a call to inspect it</span>';
    meta.textContent = '';
    return;
  }
  meta.textContent = `#${c.id} · ${c.short}`;
  // Show classifier disagreement as a comparison badge
  const v1sp = c.species_v1 ?? c.species;
  const v2sp = c.species_v2 ?? c.species;
  const disagrees = v1sp !== v2sp;
  const cmpRow = disagrees
    ? `<tr style="color:#aaa;font-size:10px"><td>v1 says</td>
         <td><span style="background:${c.color_v1};color:#fff;padding:1px 4px;border-radius:2px;font-size:9px">${c.short_v1}</span> ${(c.conf_v1*100).toFixed(0)}%</td></tr>`
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
  document.getElementById('btn-zoom-call').onclick = () => {
    const pad = Math.max(0.4, (S.selectedCall.t1 - S.selectedCall.t0) * 1.5);
    S.viewStart = Math.max(0, S.selectedCall.t0 - pad);
    const vEnd  = Math.min(S.duration, S.selectedCall.t1 + pad);
    S.viewDur   = vEnd - S.viewStart;
    scheduleRender();
  };
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
      ${prof.refs.map(r => `<span class="ref-tag">${r}</span>`).join('')}
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
  updateScrollbar();
  try { _profiles = await (await fetch('/api/profiles')).json(); } catch {}
  S.colors = info.colors;
  buildLegend(S.colors);

  document.getElementById('file-meta').textContent =
    `${(info.duration_s / 60).toFixed(1)} min  ·  ${(info.sr / 1000).toFixed(0)} kHz  ·  ${info.n_tiles} tiles`;

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

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Modal helpers ────────────────────────────────────────────
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
}

function openAbout() {
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
</script>
</body>
</html>"""

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def trim_call_contour(c):
    """Clean a call's frequency contour and tighten Fmin/Fmax.

    Three passes:

    Pass 1 — Floor filter
        Discard any contour point below 20 kHz (below which no western-NA bat
        species emits echolocation energy in this recording).

    Pass 2 — Harmonic separation
        Look for a large frequency gap (> 10 kHz) that splits the remaining
        points into a lower cluster (fundamental) and an upper cluster
        (harmonic).  If such a gap exists AND the lower cluster accounts for
        ≥ 15% of the post-floor points, keep only the lower cluster.

        Rationale: the raw argmax often jumps to the first harmonic (at ~2×
        the fundamental) when the harmonic is momentarily louder.  This is
        common in LACI calls where the harmonic can dominate.  The gap between
        fundamental and harmonic is always ≥ one fundamental-width (typically
        15–40 kHz) and is reliably larger than any legitimate intra-call FM
        sweep seen in this dataset.

    Pass 3 — Bounding box
        Recompute Fmin / Fmax from the cleaned contour points.
    """
    cnt = c.get("contour")
    if not cnt or len(cnt) < 2:
        return
    cnt.sort(key=lambda pt: pt[0])   # guarantee time-monotone order (in-place)
    freqs = np.array([pt[1] for pt in cnt])

    # ── Pass 1: floor filter ─────────────────────────────────────
    mask    = freqs >= 20.0
    trimmed_pts   = [pt for pt, m in zip(cnt, mask) if m]
    trimmed_freqs = freqs[mask]

    if len(trimmed_pts) == 0:
        return   # all floor-noise; leave untouched
    if len(trimmed_pts) == 1:
        f = float(trimmed_freqs[0])
        c["contour"] = [[cnt[0][0], f], [cnt[-1][0], f]]
        c["Fmin"]    = round(f - 2.0, 2)
        c["Fmax"]    = round(f + 2.0, 2)
        return

    # ── Pass 2: harmonic separation ──────────────────────────────
    # Criteria for a true fundamental/harmonic split:
    #   (a) gap ≥ 7 kHz between the two clusters, AND
    #   (b) upper-cluster centre / lower-cluster centre ≥ 1.55
    #       (≈ 2× fundamental; real harmonics land at 1.8–2.2×, FM-sweep
    #        intra-call components land at 1.2–1.4× and must NOT be split)
    HARMONIC_GAP_KHZ  = 7.0    # minimum gap that might indicate a harmonic split
    HARMONIC_RATIO    = 1.55   # minimum freq-ratio upper/lower to confirm harmonic
    MIN_FUND_FRACTION = 0.15   # lower cluster must have ≥ 15% of points to keep it

    sorted_f = np.sort(trimmed_freqs)
    gaps     = np.diff(sorted_f)
    if gaps.max() >= HARMONIC_GAP_KHZ:
        split_idx     = int(gaps.argmax())
        lower_cluster = sorted_f[:split_idx + 1]
        upper_cluster = sorted_f[split_idx + 1:]
        n_total       = len(sorted_f)
        lower_centre  = float(np.median(lower_cluster))
        upper_centre  = float(np.median(upper_cluster))
        ratio         = upper_centre / lower_centre if lower_centre > 0 else 0.0
        if (ratio >= HARMONIC_RATIO
                and len(lower_cluster) >= max(2, MIN_FUND_FRACTION * n_total)):
            # Keep only points in the lower (fundamental) cluster
            fund_ceil     = float(lower_cluster.max()) + 2.0
            keep          = trimmed_freqs <= fund_ceil
            trimmed_pts   = [pt for pt, k in zip(trimmed_pts, keep) if k]
            trimmed_freqs = trimmed_freqs[keep]

    if len(trimmed_pts) == 0:
        return
    if len(trimmed_pts) == 1:
        f = float(trimmed_freqs[0])
        c["contour"] = [[cnt[0][0], f], [cnt[-1][0], f]]
        c["Fmin"]    = round(f - 2.0, 2)
        c["Fmax"]    = round(f + 2.0, 2)
        return

    # ── Pass 3: update contour, bounding box, and derived features ──
    c["contour"] = trimmed_pts
    new_lo = float(trimmed_freqs.min())
    new_hi = float(trimmed_freqs.max())
    if new_hi - new_lo < 1.0:
        pad     = (1.0 - (new_hi - new_lo)) / 2
        new_lo -= pad;  new_hi += pad
    c["Fmin"] = round(new_lo, 2)
    c["Fmax"] = round(new_hi, 2)

    # Bandwidth: total frequency span of the cleaned contour (kHz)
    c["bw"] = round(new_hi - new_lo, 2)

    # CF fraction: proportion of contour frames within 2 kHz of the median
    # frequency.  High → mostly constant-frequency (CF); low → steep FM sweep.
    med_f   = float(np.median(trimmed_freqs))
    c["cf_frac"] = round(float(np.mean(np.abs(trimmed_freqs - med_f) <= 2.0)), 3)



def reclassify_calls(calls):
    """Run both classifiers on every call in-place using the current PROFILES.

    Stores v1 (4-criterion) and v2 (6-criterion with bw+cf_frac) results.
    c["species"] / c["color"] / c["short"] / c["conf"] always reflect v2 so
    the rest of the code works unchanged.  v1 results live in c["species_v1"]
    etc. for the UI comparison toggle.
    """
    counts_v1, counts_v2 = {}, {}
    short_map = {p["name"]: p["short"] for p in PROFILES}
    for c in calls:
        sp1, cf1 = classify_v1(c)
        sp2, cf2 = classify_v2(c)
        # v1
        c["species_v1"] = sp1
        c["conf_v1"]    = cf1
        c["color_v1"]   = COLORS.get(sp1, "#888888")
        c["short_v1"]   = short_map.get(sp1, "????")
        counts_v1[sp1]  = counts_v1.get(sp1, 0) + 1
        # v2 (default / active)
        c["species"] = sp2
        c["conf"]    = cf2
        c["color"]   = COLORS.get(sp2, "#888888")
        c["short"]   = short_map.get(sp2, "????")
        counts_v2[sp2]  = counts_v2.get(sp2, 0) + 1

    def _summary(counts):
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    print(f"reclassify v1: {_summary(counts_v1)}")
    print(f"reclassify v2: {_summary(counts_v2)}")


def try_load_cache():
    """Return True if valid cached results were loaded, False if detection must run."""
    global all_calls
    if not os.path.exists(CACHE_FILE):
        return False
    try:
        with open(CACHE_FILE) as fh:
            cache = json.load(fh)
        # Invalidate if detection threshold changed
        if cache.get("bd2_thresh") != BD2_THRESH:
            print("Cache is stale (BD2_THRESH changed) — re-detecting.")
            return False
        all_calls.extend(cache["calls"])
        det = cache.get("detector", "cached")
        # Trim contour outliers in case the cache pre-dates the freq-gating fix
        for c in all_calls:
            trim_call_contour(c)
        # Re-run classifier so profile changes / priors take effect without re-detecting
        reclassify_calls(all_calls)
        progress["status"] = f"Loaded from cache — {len(all_calls)} calls  [{det}]"
        calls_ready.set()
        print(progress["status"])
        return True
    except Exception as exc:
        print(f"Cache load failed ({exc}) — re-detecting.")
        return False

def startup(redetect=False):
    global audio_fh, TILE_DIR
    print(f"Opening {AUDIO_FILE} …")
    audio_fh = sf.SoundFile(AUDIO_FILE)
    finfo.update({
        "sr":         audio_fh.samplerate,
        "nframes":    audio_fh.frames,
        "channels":   audio_fh.channels,
        "duration_s": audio_fh.frames / audio_fh.samplerate,
    })
    print(f"  {finfo['duration_s']:.1f} s  ·  {finfo['sr']:,} Hz  ·  {finfo['channels']} ch")

    TILE_DIR = os.path.splitext(AUDIO_FILE)[0] + "_tiles"
    os.makedirs(TILE_DIR, exist_ok=True)
    print(f"  Tile cache → {TILE_DIR}")

    # Compute (or load) global spectrogram normalisation before any tiles are made
    _init_tile_norm()

    if not redetect and try_load_cache():
        # Detection loaded from cache — pre-generate tiles right away
        threading.Thread(target=_pregenerate_tiles, daemon=True).start()
        return

    progress["status"] = "Detection starting…"
    t = threading.Thread(target=run_detection, daemon=True)
    t.start()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Bat echolocation spectrogram viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file", nargs="?", default=AUDIO_FILE,
        help="Path to FLAC/WAV bat recording",
    )
    parser.add_argument(
        "--port", type=int, default=5001,
        help="HTTP port to listen on",
    )
    parser.add_argument(
        "--redetect", action="store_true",
        help="Ignore cached detections and re-run BatDetect2",
    )
    args = parser.parse_args()

    # Override module-level path constants with CLI values
    AUDIO_FILE = args.file
    CACHE_FILE = os.path.splitext(AUDIO_FILE)[0] + ".calls.json"

    if args.redetect:
        print("--redetect: ignoring cache, re-running detection.")

    startup(redetect=args.redetect)
    print(f"\nStarting server → http://localhost:{args.port}  (Ctrl-C to stop)\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True, use_reloader=False)
