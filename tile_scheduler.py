"""
Priority-based background tile scheduler.

Renders raw and flat PNG tiles for every registered audio file.  Workers
never stop; file switches change priorities rather than interrupting work.

Heap entries: (-priority, sequence, path, tile_type, tidx)
Lazy deletion: boosting pushes a new higher-priority entry; when a worker
pops an old entry whose (path, tile_type, tidx) is already in _done, it
silently skips it.
"""

import heapq, io, json, os, threading
import numpy as np
from scipy import signal
from PIL import Image

from config import (
    TILE_DURATION, TILE_W, TILE_H,
    FREQ_LOW, FREQ_HIGH,
    D_NPERSEG, D_NOVERLAP,
    TILE_NORM_VERSION,
)

PRIO_BG       =      1   # idle file, not currently viewed
PRIO_ACTIVE   =    100   # currently viewed file
PRIO_VIEWPORT = 10_000   # tiles visible right now in viewport
N_WORKERS     =      1   # server has 1 CPU; two workers just context-switch


# ── Standalone render functions ────────────────────────────────────────────────
# These take explicit arguments so they work for any file, not just state.*.

def _spectrogram_db(mono, sr):
    f_s, _, Sxx = signal.spectrogram(mono, fs=sr, nperseg=D_NPERSEG,
                                      noverlap=D_NOVERLAP, window="hann")
    bm = (f_s >= FREQ_LOW) & (f_s <= FREQ_HIGH)
    return 10 * np.log10(Sxx[bm, :] + 1e-12)   # (n_freq, n_time)


def _png_from_arr(arr):
    from state import _inferno
    rgb = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)
    pil = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _read_tile_mono(fh, tidx, sr, dur):
    t0 = tidx * TILE_DURATION
    t1 = min(t0 + TILE_DURATION, dur)
    f0 = int(t0 * sr)
    f1 = int(t1 * sr)
    fh.seek(f0)
    audio = fh.read(f1 - f0, dtype="float32", always_2d=True)
    return audio.mean(axis=1) if audio.ndim > 1 else audio.ravel()


def render_raw_tile(fh, tidx, sr, dur, vmin, vmax):
    mono = _read_tile_mono(fh, tidx, sr, dur)
    Sdb  = _spectrogram_db(mono, sr)
    arr  = np.clip((Sdb - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return _png_from_arr(arr)


def render_flat_tile(fh, tidx, sr, dur, vmin_f, vmax_f):
    mono   = _read_tile_mono(fh, tidx, sr, dur)
    Sdb    = _spectrogram_db(mono, sr)
    n_freq = Sdb.shape[0]
    if vmin_f is not None and vmax_f is not None and len(vmin_f) == n_freq:
        lo = vmin_f[:, np.newaxis]
        hi = vmax_f[:, np.newaxis]
    else:
        lo = np.percentile(Sdb, 2.0,  axis=1, keepdims=True)
        hi = np.percentile(Sdb, 99.9, axis=1, keepdims=True)
    arr = np.clip((Sdb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return _png_from_arr(arr)


# ── Per-file context ───────────────────────────────────────────────────────────

class _FileCtx:
    """Holds per-file state needed by scheduler workers."""

    def __init__(self, path):
        self.path     = path
        self.tile_dir = os.path.splitext(path)[0] + "_tiles"
        self._lock    = threading.Lock()
        self.ntiles   = 0
        self.progress = {
            "raw":  {"done": 0, "total": 0, "status": "idle"},
            "flat": {"done": 0, "total": 0, "status": "idle"},
        }
        self._finfo  = None
        self._norms  = None   # (vmin, vmax, vmin_f, vmax_f) once loaded

    # ── finfo / norms — lazy, thread-safe ─────────────────────────

    def finfo(self):
        if self._finfo is None:
            from startup import _open_audio
            fh = _open_audio(self.path)
            self._finfo = {"sr": fh.samplerate,
                           "duration_s": fh.frames / fh.samplerate}
            fh.close()
        return self._finfo

    def norms(self):
        with self._lock:
            if self._norms is None:
                self._norms = self._load_norms()
        return self._norms

    def _load_norms(self):
        norm_path = os.path.join(self.tile_dir, "norm.json")
        if os.path.exists(norm_path):
            try:
                with open(norm_path) as f:
                    d = json.load(f)
                if d.get("version") == TILE_NORM_VERSION:
                    vf = np.array(d["vmin_f"]) if "vmin_f" in d else None
                    wf = np.array(d["vmax_f"]) if "vmax_f" in d else None
                    return (d["vmin"], d["vmax"], vf, wf)
            except Exception:
                pass
        return self._compute_norms(norm_path)

    def _compute_norms(self, norm_path):
        from startup import _open_audio
        fi  = self.finfo()
        sr, dur = fi["sr"], fi["duration_s"]
        n   = int(np.ceil(dur / TILE_DURATION))
        idxs = np.linspace(0, n - 1, min(30, n), dtype=int)
        vmins, vmaxs, plos, phis = [], [], [], []
        fh = _open_audio(self.path)
        try:
            for i in idxs:
                try:
                    mono = _read_tile_mono(fh, i, sr, dur)
                    Sdb  = _spectrogram_db(mono, sr)
                    vmins.append(float(np.percentile(Sdb, 2.0)))
                    vmaxs.append(float(np.percentile(Sdb, 99.9)))
                    plos.append(np.percentile(Sdb, 2.0,  axis=1))
                    phis.append(np.percentile(Sdb, 99.9, axis=1))
                except Exception:
                    pass
        finally:
            fh.close()
        vmin   = float(np.percentile(vmins, 50)) if vmins else -100.0
        vmax   = float(np.percentile(vmaxs, 75)) if vmaxs else  -30.0
        vmin_f = np.percentile(np.vstack(plos), 50, axis=0) if plos else None
        vmax_f = np.percentile(np.vstack(phis), 75, axis=0) if phis else None
        try:
            os.makedirs(self.tile_dir, exist_ok=True)
            d = {"version": TILE_NORM_VERSION, "mode": "global",
                 "vmin": vmin, "vmax": vmax}
            if vmin_f is not None:
                d["vmin_f"] = vmin_f.tolist()
                d["vmax_f"] = vmax_f.tolist()
            with open(norm_path, "w") as f:
                json.dump(d, f)
        except Exception:
            pass
        return (vmin, vmax, vmin_f, vmax_f)

    # ── disk helpers ──────────────────────────────────────────────

    def disk_path(self, tile_type, tidx):
        prefix = "tile" if tile_type == "raw" else "flat_tile"
        return os.path.join(self.tile_dir, f"{prefix}_{tidx:04d}.png")

    def on_disk(self, tile_type, tidx):
        return os.path.exists(self.disk_path(tile_type, tidx))


# ── Scheduler ─────────────────────────────────────────────────────────────────

class TileScheduler:

    def __init__(self):
        self._heap   = []
        self._lock   = threading.Lock()
        self._cond   = threading.Condition(self._lock)
        self._seq    = 0
        self._files  = {}      # path → _FileCtx
        self._done   = set()   # (path, tile_type, tidx) already on disk
        self._active = None    # path of currently-viewed file

    def start(self, n=N_WORKERS):
        for _ in range(n):
            threading.Thread(target=self._worker, daemon=True).start()

    # ── Public API ────────────────────────────────────────────────

    def register_file(self, path):
        """Enqueue all missing tiles for `path` at background priority.
        No-op if the file is already registered."""
        with self._lock:
            if path in self._files:
                return
            ctx = _FileCtx(path)
            self._files[path] = ctx

        # finfo() may open the file briefly — do this outside the lock
        fi = ctx.finfo()
        n  = int(np.ceil(fi["duration_s"] / TILE_DURATION))
        ctx.ntiles = n
        prio = PRIO_ACTIVE if path == self._active else PRIO_BG

        with self._lock:
            for tt in ("raw", "flat"):
                missing = 0
                for i in range(n):
                    key = (path, tt, i)
                    if key in self._done or ctx.on_disk(tt, i):
                        self._done.add(key)
                    else:
                        self._push(path, tt, i, prio)
                        missing += 1
                done = n - missing
                ctx.progress[tt] = {
                    "done": done, "total": n,
                    "status": "done" if missing == 0 else
                              ("running" if prio >= PRIO_ACTIVE else "idle"),
                }
            self._cond.notify_all()

    def set_active(self, path):
        """Make `path` the active file; boost all its tiles to PRIO_ACTIVE."""
        with self._lock:
            self._active = path
            ctx = self._files.get(path)
            if ctx is None:
                return
            for tt in ("raw", "flat"):
                for i in range(ctx.ntiles):
                    if (path, tt, i) not in self._done:
                        self._push(path, tt, i, PRIO_ACTIVE)
                if ctx.progress[tt]["status"] == "idle":
                    ctx.progress[tt]["status"] = "running"
            self._cond.notify_all()

    def boost_viewport(self, path, t0, t1):
        """Boost tiles covering [t0, t1] to PRIO_VIEWPORT."""
        with self._lock:
            ctx = self._files.get(path)
            if ctx is None:
                return
            i0 = max(0, int(t0 / TILE_DURATION))
            i1 = min(ctx.ntiles - 1, int(np.ceil(t1 / TILE_DURATION)))
            for tt in ("raw", "flat"):
                for i in range(i0, i1 + 1):
                    if (path, tt, i) not in self._done:
                        self._push(path, tt, i, PRIO_VIEWPORT)
            self._cond.notify_all()

    def get_progress(self, path=None):
        """Return tile_progress dict for a path (default: the active file).
        raw/flat come from the scheduler's _FileCtx; mask comes from the registry entry."""
        import registry
        p = path or self._active
        result = {
            "raw":  {"done": 0, "total": 0, "status": "idle"},
            "flat": {"done": 0, "total": 0, "status": "idle"},
            "mask": {"done": 0, "total": 0, "status": "idle"},
        }
        if p:
            ctx = self._files.get(p)
            if ctx:
                result["raw"]  = dict(ctx.progress["raw"])
                result["flat"] = dict(ctx.progress["flat"])
            entry = registry.get_by_path(p)
            if entry:
                result["mask"] = dict(entry.mask_progress)
        return result

    # ── Internals ─────────────────────────────────────────────────

    def _push(self, path, tt, i, prio):
        """Push a task entry. Caller must hold self._lock."""
        self._seq += 1
        heapq.heappush(self._heap, (-prio, self._seq, path, tt, i))

    def _worker(self):
        """Long-running worker thread; maintains its own per-file audio handles."""
        handles = {}   # path → (fh, sr, dur)

        while True:
            with self._cond:
                while not self._heap:
                    self._cond.wait()
                neg_p, _seq, path, tt, tidx = heapq.heappop(self._heap)

            key = (path, tt, tidx)
            with self._lock:
                if key in self._done:
                    continue

            ctx = self._files.get(path)
            if ctx is None:
                continue

            # Already on disk — just mark done and update counter
            if ctx.on_disk(tt, tidx):
                with self._lock:
                    self._done.add(key)
                p = ctx.progress[tt]
                p["done"] = min(p["done"] + 1, p["total"])
                if p["done"] >= p["total"]:
                    p["status"] = "done"
                continue

            # Open a per-worker audio handle for this file (persistent)
            if path not in handles:
                from startup import _open_audio
                try:
                    fh = _open_audio(path)
                    fi = ctx.finfo()
                    handles[path] = (fh, fi["sr"], fi["duration_s"])
                except Exception as exc:
                    print(f"[sched] cannot open {os.path.basename(path)}: {exc}")
                    continue
            fh, sr, dur = handles[path]

            # Load norms (lazy, cached per _FileCtx, thread-safe)
            try:
                vmin, vmax, vmin_f, vmax_f = ctx.norms()
            except Exception as exc:
                print(f"[sched] norms failed {os.path.basename(path)}: {exc}")
                continue

            # Render and write to disk — hold the bg semaphore only during
            # compute so request handlers can always run between tiles.
            try:
                from tiles import _bg_sem
                with _bg_sem:
                    if tt == "raw":
                        data = render_raw_tile(fh, tidx, sr, dur, vmin, vmax)
                    else:
                        data = render_flat_tile(fh, tidx, sr, dur, vmin_f, vmax_f)

                disk = ctx.disk_path(tt, tidx)
                os.makedirs(ctx.tile_dir, exist_ok=True)
                with open(disk, "wb") as f:
                    f.write(data)

                with self._lock:
                    self._done.add(key)
                p = ctx.progress[tt]
                p["done"] = min(p["done"] + 1, p["total"])
                if p["done"] >= p["total"]:
                    p["status"] = "done"
                    print(f"[sched] {tt} done: {os.path.basename(path)}", flush=True)

            except Exception as exc:
                print(f"[sched] {tt}[{tidx}] {os.path.basename(path)}: {exc}")
