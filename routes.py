import io, json, os
from pathlib import Path
import numpy as np
from flask import jsonify, send_file, render_template, request

from state import app
import state
import registry as reg
import config
from config import (
    TILE_DURATION, TILE_W, TILE_H,
    FREQ_LOW_K, FREQ_HIGH_K, FREQ_LOW, FREQ_HIGH,
    D_NPERSEG, D_NOVERLAP,
    TILE_NORM_VERSION,
)
from tiles import make_tile, make_mask_tile, make_flat_tile
from species import PROFILES, COLORS
from scipy import signal


def _entry_or_404(fid=None):
    """Look up the FileEntry for the given fid (or the default).
    Returns (entry, None) on success, (None, error_response) on failure."""
    entry = reg.get_or_default(fid)
    if not entry:
        return None, (jsonify({"error": "file not found"}), 404)
    return entry, None


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def api_info():
    entry, err = _entry_or_404(request.args.get('f'))
    if err:
        return err
    if not entry.finfo:
        return jsonify({"error": "not ready"}), 503
    return jsonify({
        **entry.finfo,
        "fid":             entry.fid,
        "tile_duration":   TILE_DURATION,
        "tile_w":          TILE_W,
        "tile_h":          TILE_H,
        "freq_low":        FREQ_LOW_K,
        "freq_high":       FREQ_HIGH_K,
        "n_tiles":         int(np.ceil(entry.finfo["duration_s"] / TILE_DURATION)),
        "tile_version":    f"{TILE_NORM_VERSION}_{Path(entry.path).stem}",
        "colors":          COLORS,
        "ready":           entry.calls_ready.is_set(),
        "progress":        entry.detection_progress,
        "bit_depth":       entry.finfo.get("bit_depth", ""),
        "recording_start": entry.finfo.get("recording_start"),
        "filename":        Path(entry.audio_fh.name).name if entry.audio_fh else "",
    })


@app.route("/api/status")
def api_status():
    entry, err = _entry_or_404(request.args.get('f'))
    if err:
        return err
    tp = (state.scheduler.get_progress(entry.path)
          if state.scheduler else {
              "raw":  {"done": 0, "total": 0, "status": "idle"},
              "flat": {"done": 0, "total": 0, "status": "idle"},
              "mask": dict(entry.mask_progress),
          })
    return jsonify({"ready":         entry.calls_ready.is_set(),
                    "progress":      entry.detection_progress,
                    "tile_progress": tp})


@app.route("/api/boost", methods=["POST"])
def api_boost():
    """Boost tile generation priority for the viewport of the requested file."""
    entry, err = _entry_or_404(request.args.get('f'))
    if not err and state.scheduler:
        data = request.get_json(force=True) or {}
        t0   = float(data.get("t0", 0))
        t1   = float(data.get("t1", t0 + 30))
        state.scheduler.boost_viewport(entry.path, t0, t1)
    return jsonify({"ok": True})


@app.route("/api/calls")
def api_calls():
    entry, err = _entry_or_404(request.args.get('f'))
    if err:
        return err
    return jsonify({"ready": entry.calls_ready.is_set(),
                    "calls": list(entry.all_calls)})


@app.route("/api/psd")
def api_psd():
    """Average power-spectrum for the requested time window (kHz + normalised power)."""
    entry, err = _entry_or_404(request.args.get('f'))
    if err:
        return err
    if not entry.finfo:
        return jsonify({"freqs": [], "powers": []}), 503
    dur = float(entry.finfo["duration_s"])
    sr  = entry.finfo["sr"]
    t0  = max(0.0, min(dur, float(request.args.get("t0", 0))))
    t1  = max(t0 + 0.01, min(dur, float(request.args.get("t1", dur))))
    # Cap to 4 s (centred on window) so computation stays fast
    MAX_S = 4.0
    if t1 - t0 > MAX_S:
        mid = (t0 + t1) / 2.0
        t0  = max(0.0, mid - MAX_S / 2)
        t1  = min(dur,  t0  + MAX_S)
    f0 = int(t0 * sr)
    f1 = min(int(dur * sr), int(t1 * sr))
    with entry.audio_lock:
        entry.audio_fh.seek(f0)
        audio = entry.audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio.ravel()
    if len(mono) < D_NPERSEG:
        return jsonify({"freqs": [], "powers": []})
    f_arr, _, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm   = (f_arr >= FREQ_LOW) & (f_arr <= FREQ_HIGH)
    Sdb  = 10 * np.log10(Sxx[bm, :].mean(axis=1) + 1e-12)
    # Don't clip — JS re-normalises to the visible peak anyway
    norm = (Sdb - entry.vmin) / max(entry.vmax - entry.vmin, 1e-6)
    return jsonify({"freqs":  (f_arr[bm] / 1000).tolist(),
                    "powers": norm.tolist(),
                    "vmin":   entry.vmin,
                    "vmax":   entry.vmax})


@app.route("/api/profiles")
def api_profiles():
    """Return PROFILES list with all scholarly reference data (tuples → lists for JSON)."""
    out = []
    for p in PROFILES:
        ep = dict(p)
        ep["Fchar"] = list(p["Fchar"])
        ep["Fmin"]  = list(p["Fmin"])
        ep["dur"]   = list(p["dur"])
        ep["sweep"] = list(p["sweep"])
        ep["color"] = COLORS.get(p["name"], "#888888")
        out.append(ep)
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
    entry, err = _entry_or_404(request.args.get('f'))
    if err:
        return err
    if not entry.finfo:
        return "not ready", 503
    ntiles = int(np.ceil(entry.finfo["duration_s"] / TILE_DURATION))
    if tidx < 0 or tidx >= ntiles:
        return "not found", 404
    data = make_tile(entry, tidx)
    return send_file(io.BytesIO(data), mimetype="image/png", max_age=3600)


@app.route("/api/tile_mask/<int:tidx>")
def api_tile_mask(tidx):
    entry, err = _entry_or_404(request.args.get('f'))
    if err:
        return err
    if not entry.finfo:
        return "not ready", 503
    ntiles = int(np.ceil(entry.finfo["duration_s"] / TILE_DURATION))
    if tidx < 0 or tidx >= ntiles:
        return "not found", 404
    if not entry.calls_ready.is_set():
        return "detection not ready", 503
    data = make_mask_tile(entry, tidx)
    return send_file(io.BytesIO(data), mimetype="image/png", max_age=3600)


@app.route("/api/tile_flat/<int:tidx>")
def api_tile_flat(tidx):
    entry, err = _entry_or_404(request.args.get('f'))
    if err:
        return err
    if not entry.finfo:
        return "not ready", 503
    ntiles = int(np.ceil(entry.finfo["duration_s"] / TILE_DURATION))
    if tidx < 0 or tidx >= ntiles:
        return "not found", 404
    data = make_flat_tile(entry, tidx)
    return send_file(io.BytesIO(data), mimetype="image/png", max_age=3600)


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


@app.route("/api/files")
def api_files():
    """List all registered audio files with their stable IDs."""
    entries = sorted(reg.all_entries(), key=lambda e: e.name)
    fid     = request.args.get('f')
    current = reg.get_or_default(fid)
    return jsonify({
        "files":   [{"fid": e.fid, "name": e.name} for e in entries],
        "current": current.fid if current else None,
    })
