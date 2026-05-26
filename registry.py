"""Per-file state registry.

Each audio file gets a stable 8-char ID (sha256 of its absolute path).
All server state lives in FileEntry objects so multiple users can view
different files simultaneously without interference.
"""

import hashlib, os, threading


class FileEntry:
    """All mutable server state for one audio file."""

    def __init__(self, path: str):
        self.path       = os.path.abspath(path)
        self.name       = os.path.basename(self.path)
        self.fid        = hashlib.sha256(self.path.encode()).hexdigest()[:8]
        self.tile_dir   = os.path.splitext(self.path)[0] + "_tiles"
        self.cache_file = os.path.splitext(self.path)[0] + ".calls.json"

        self.audio_lock = threading.Lock()
        self.audio_fh   = None
        self.finfo      = {}

        self.all_calls          = []
        self.calls_ready        = threading.Event()
        self.detection_progress = {"done": 0, "total": 1, "status": "starting"}
        self.stop_event         = threading.Event()

        self.vmin   = -100.0
        self.vmax   =  -30.0
        self.vmin_f = None
        self.vmax_f = None

        self.tile_cache      = {}
        self.tile_lock       = threading.Lock()
        self.flat_tile_cache = {}
        self.flat_tile_lock  = threading.Lock()
        self.mask_tile_cache = {}
        self.mask_tile_lock  = threading.Lock()

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
