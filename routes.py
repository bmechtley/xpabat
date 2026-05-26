import io, json, os
import numpy as np
from flask import jsonify, send_file, render_template, request

from state import app, finfo, all_calls, calls_ready, progress
from state import _global_vmin, _global_vmax
import state
from config import (
    TILE_DURATION, TILE_W, TILE_H,
    FREQ_LOW_K, FREQ_HIGH_K, FREQ_LOW, FREQ_HIGH,
    D_NPERSEG, D_NOVERLAP,
    TILE_NORM_VERSION,
)
from tiles import make_tile, make_mask_tile, make_flat_tile
from species import PROFILES, COLORS
from scipy import signal


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

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
        "bit_depth":       finfo.get("bit_depth", ""),
        "recording_start": finfo.get("recording_start"),
    })

@app.route("/api/status")
def api_status():
    return jsonify({"ready": calls_ready.is_set(), "progress": progress})

@app.route("/api/calls")
def api_calls():
    return jsonify({"ready": calls_ready.is_set(),
                    "calls": list(all_calls)})

@app.route("/api/psd")
def api_psd():
    """Average power-spectrum for the requested time window (kHz + normalised power)."""
    dur = float(finfo["duration_s"])
    sr  = finfo["sr"]
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
    with state.audio_lock:
        state.audio_fh.seek(f0)
        audio = state.audio_fh.read(f1 - f0, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio.ravel()
    if len(mono) < D_NPERSEG:
        return jsonify({"freqs": [], "powers": []})
    f_arr, _, Sxx = signal.spectrogram(
        mono, fs=sr, nperseg=D_NPERSEG, noverlap=D_NOVERLAP, window="hann")
    bm   = (f_arr >= FREQ_LOW) & (f_arr <= FREQ_HIGH)
    Sdb  = 10 * np.log10(Sxx[bm, :].mean(axis=1) + 1e-12)
    norm = np.clip((Sdb - state._global_vmin) / max(state._global_vmax - state._global_vmin, 1e-6), 0, 1)
    return jsonify({"freqs": (f_arr[bm] / 1000).tolist(), "powers": norm.tolist(),
                    "vmin": state._global_vmin, "vmax": state._global_vmax})

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
