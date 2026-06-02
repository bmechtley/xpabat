"""Per-file state registry.

Each audio file gets a stable 8-char ID (sha256 of its absolute path).
All server state lives in FileEntry objects so multiple users can view
different files simultaneously without interference.
"""

import hashlib, os, threading
from collections import OrderedDict


# ── LRU tile cache ────────────────────────────────────────────────────────────
# Tile PNGs are persisted to disk; the in-memory cache is a read-through
# optimisation for the current viewport.  Unbounded dicts caused OOM on the
# 2 GB server (756 MB per fully-scrolled file × 5 cache types × 4 files ≈ 3 GB).
# Capping at TILE_CACHE_SIZE keeps each cache type at ≤ 30 × ~700 KB ≈ 21 MB
# per file, ~420 MB total across 4 files and 5 cache types worst-case.
TILE_CACHE_SIZE = 30   # tiles kept in RAM per (file, cache-type)


class _LRUDict:
    """Least-recently-used dict with a fixed capacity.

    Thread-unsafe — callers must hold an external lock (the tile_lock that
    already wraps every cache access in tiles.py).
    """

    __slots__ = ("_d", "_maxsize")

    def __init__(self, maxsize: int = TILE_CACHE_SIZE):
        self._d       = OrderedDict()
        self._maxsize = maxsize

    def __contains__(self, key):
        return key in self._d

    def __getitem__(self, key):
        self._d.move_to_end(key)
        return self._d[key]

    def __setitem__(self, key, value):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        while len(self._d) > self._maxsize:
            self._d.popitem(last=False)   # evict oldest

    def clear(self):
        self._d.clear()


class FileEntry:
    """All mutable server state for one audio file."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path)
        self.fid  = hashlib.sha256(self.path.encode()).hexdigest()[:8]

        import gen_paths as _gp
        self.generated_dir    = _gp.generated_dir(self.path)
        self.spectrograms_dir = _gp.spectrograms_dir(self.path)
        self.calls_dir        = _gp.calls_dir(self.path)
        # tile_dir kept as alias for spectrograms_dir so legacy callers still work
        self.tile_dir         = self.spectrograms_dir
        # cache_file points to new batdetect2.json; try_load_cache falls back
        # to the old .calls.json if the new path doesn't exist yet.
        self.cache_file       = _gp.calls_meta_path(self.path, "batdetect2")

        self.audio_lock = threading.Lock()
        self.audio_fh   = None
        self.finfo      = {}

        # Which per-contour-type files have been merged into all_calls.
        # Populated by startup.ensure_contour_loaded().
        self.contour_loaded = {}   # method_name → True
        self.contour_lock   = threading.Lock()

        # ── Per-detector call storage ─────────────────────────────
        # calls_by_detector["batdetect2"] / ["tadarida"] → list of call dicts
        # Backward-compat: all_calls / calls_ready / detection_progress still
        # point at the "batdetect2" slot (the original default detector).
        self.calls_by_detector   = {}            # detector_key → call list
        self.ready_by_detector   = {}            # detector_key → threading.Event
        self.progress_by_detector = {}           # detector_key → progress dict

        # Legacy aliases pointing at batdetect2 slot (used throughout detect.py)
        self.all_calls          = []             # reference: same list as calls_by_detector["batdetect2"]
        self.calls_ready        = threading.Event()
        self.detection_progress = {"done": 0, "total": 1, "status": "Loading audio…"}
        self.stop_event         = threading.Event()

        # Initialise the default (batdetect2) slot to the legacy objects
        self.calls_by_detector["batdetect2"]    = self.all_calls
        self.ready_by_detector["batdetect2"]    = self.calls_ready
        self.progress_by_detector["batdetect2"] = self.detection_progress

        self.vmin   = -100.0
        self.vmax   =  -30.0
        self.vmin_f = None
        self.vmax_f = None
        self.reass_norm_max   = None
        self.reass_norm_max_f = None
        self._reass_norm_done = False
        self.psd_p01 = -120.0   # 1st-percentile dB across all display-range bins
        self.psd_p99 =  -40.0   # 99th-percentile dB across all display-range bins

        self.tile_cache      = _LRUDict()
        self.tile_lock       = threading.Lock()
        self.flat_tile_cache = _LRUDict()
        self.flat_tile_lock  = threading.Lock()
        self.mask_tile_cache = _LRUDict()
        self.mask_tile_lock  = threading.Lock()
        self.reassigned_tile_cache      = _LRUDict()
        self.reassigned_tile_lock       = threading.Lock()
        self.flat_reassigned_tile_cache = _LRUDict()
        self.flat_reassigned_tile_lock  = threading.Lock()

        self.mask_progress = {"done": 0, "total": 0, "status": "idle"}


_entries: dict = {}   # fid  → FileEntry
_by_path: dict = {}   # path → FileEntry
_default_fid   = None
_lock          = threading.Lock()


def register(path: str) -> FileEntry:
    """Return (or create) the FileEntry for path.  Thread-safe."""
    abspath = os.path.abspath(path)
    fid = hashlib.sha256(abspath.encode()).hexdigest()[:8]
    with _lock:
        if fid in _entries:
            return _entries[fid]
        entry = FileEntry(abspath)
        _entries[fid]     = entry
        _by_path[abspath] = entry
        return entry


def get(fid: str):
    return _entries.get(fid)


def get_by_path(path: str):
    return _by_path.get(os.path.abspath(path))


def get_or_default(fid=None):
    if fid:
        e = _entries.get(fid)
        if e:
            return e
    if _default_fid:
        return _entries.get(_default_fid)
    if _entries:
        return next(iter(_entries.values()))
    return None


def set_default(path: str) -> None:
    global _default_fid
    abspath = os.path.abspath(path)
    fid = hashlib.sha256(abspath.encode()).hexdigest()[:8]
    _default_fid = fid


def all_entries() -> list:
    return list(_entries.values())
