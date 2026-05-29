import io, json, os
from pathlib import Path
import numpy as np
from flask import jsonify, send_file, render_template, request, redirect

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

@app.before_request
def _force_https():
    """Redirect HTTP → HTTPS when running behind a reverse proxy (piku/nginx).
    nginx sets X-Forwarded-Proto; the header is absent on localhost dev runs,
    so this redirect never fires during local development."""
    if request.headers.get('X-Forwarded-Proto') == 'http':
        return redirect(request.url.replace('http://', 'https://', 1), code=301)

@app.after_request
def _coop_coep(response):
    """SharedArrayBuffer requires both COOP and COEP to be set."""
    response.headers['Cross-Origin-Opener-Policy']   = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    return response


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
    # Return raw dB values; the client normalises against its own expanding scale.
    return jsonify({"freqs":   (f_arr[bm] / 1000).tolist(),
                    "dbs":     Sdb.tolist(),
                    "psd_p01": entry.psd_p01,
                    "psd_p99": entry.psd_p99})


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


def _tool_summary(name, inp):
    """Short human-readable label for a tool call's input."""
    import pathlib as _pl
    if not isinstance(inp, dict):
        return name
    if name == "Bash":
        desc = (inp.get("description") or "").strip()
        cmd  = (inp.get("command")     or "").strip()
        if desc:
            return desc[:120] + ("…" if len(desc) > 120 else "")
        cmd_flat = " ".join(cmd.split())
        return cmd_flat[:100] + ("…" if len(cmd_flat) > 100 else "")
    if name in ("Read", "Edit", "Write"):
        p = inp.get("file_path") or ""
        return _pl.Path(p).name or p[:60] or name
    if name == "WebFetch":
        url = inp.get("url") or ""
        return url[:120] + ("…" if len(url) > 120 else "")
    if name == "WebSearch":
        q = inp.get("query") or ""
        return q[:120] + ("…" if len(q) > 120 else "")
    if name == "Agent":
        d = inp.get("description") or ""
        return d[:120] + ("…" if len(d) > 120 else "")
    if name == "Skill":
        return inp.get("skill") or name
    # Fallback: first non-empty string value
    for v in inp.values():
        if isinstance(v, str) and v.strip():
            s = v.strip()
            return s[:100] + ("…" if len(s) > 100 else "")
    return name


_NOTE_PREFIXES = (
    "This session is being continued from a previous conversation",
    "[IMPORTANT: Read this context",
)

# Matches a message whose entire content is a slash-command injection block,
# e.g. <create-pr-command>…</create-pr-command> → the user only typed /create-pr
import re as _re
_SLASH_CMD_BLOCK_RE = _re.compile(
    r'^\s*<([a-z][a-z0-9-]+-command)>.*?</\1>\s*$', _re.DOTALL)

def _slash_cmd_label(text):
    """If text is entirely a <xxx-command> injection, return the '/cmd' label; else None."""
    m = _SLASH_CMD_BLOCK_RE.match(text)
    if m:
        tag = m.group(1)               # e.g. "create-pr-command"
        cmd = tag[:-len('-command')]   # e.g. "create-pr"
        return f'/{cmd}'
    return None


def _tool_detail(name, inp):
    """Full expandable detail string for a tool call (raw command / path / etc.)."""
    if not isinstance(inp, dict):
        return ""
    if name == "Bash":
        return (inp.get("command") or "").strip()
    if name in ("Read", "Edit", "Write"):
        return (inp.get("file_path") or "").strip()
    if name == "WebFetch":
        return (inp.get("url") or "").strip()
    if name == "Agent":
        d = (inp.get("description") or "").strip()
        p = (inp.get("prompt")      or "").strip()
        return (d + ("\n\n" + p[:500] if p else "")).strip()
    return ""


def _parse_ts(ts_str):
    """Parse an ISO-8601 timestamp string to a datetime, or return None."""
    if not ts_str:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_jsonl_messages(raw_lines):
    """Parse raw JSONL dicts into a flat list of message dicts.

    Returns dicts with role in {"user", "assistant", "note", "tool"}.

    Text messages:
      {"role": "user"|"assistant"|"note", "text": "...", "ts": "...",
       "stats": {"output_tokens": N, "input_tokens": N,
                 "duration_s": N, "has_thinking": bool}}   # stats on assistant only

    Tool messages:
      {"role": "tool", "name": "...", "summary": "...", "detail": "...",
       "result": "...", "is_error": bool, "duration_s": N|None, "ts": "..."}
    """
    # ── Pass 1: map tool_use_id → {content, is_error, ts} ────────
    result_map = {}
    for obj in raw_lines:
        ts      = obj.get("timestamp", "")
        content = obj.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            uid = block.get("tool_use_id", "")
            if not uid:
                continue
            raw = block.get("content", "")
            if isinstance(raw, list):
                raw = "\n".join(
                    b.get("text", "") for b in raw
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            result_map[uid] = {
                "content":  str(raw),
                "is_error": bool(block.get("is_error", False)),
                "ts":       ts,   # timestamp of the tool_result (for exec-time calc)
            }

    # ── Pass 2: emit in order, tracking timing state ──────────────
    messages        = []
    _last_human_ts  = None   # ts of the last real human message
    _saw_thinking   = False  # did a thinking block appear in this API call?

    for obj in raw_lines:
        typ  = obj.get("type")
        msg  = obj.get("message", {})
        role = msg.get("role", "")
        ts   = obj.get("timestamp", "")

        if typ not in ("user", "assistant") or role not in ("user", "assistant"):
            continue

        content = msg.get("content", "")
        usage   = msg.get("usage") or {}

        # ── String content (legacy / summary entries) ─────────────
        if isinstance(content, str):
            text = content.strip()
            if (text
                    and not text.startswith("<system-reminder")
                    and not text.startswith("<function_calls>")):
                label = _slash_cmd_label(text)
                if label:
                    text = label
                emit_role = role
                if role == "user" and any(text.startswith(p) for p in _NOTE_PREFIXES):
                    emit_role = "note"
                else:
                    # Real human message: update timing state
                    _last_human_ts = ts
                    _saw_thinking  = False
                messages.append({"role": emit_role, "text": text, "ts": ts})
            continue

        if not isinstance(content, list):
            continue

        # ── Determine if this user entry is a real human message ──
        # Each JSONL entry has exactly one content block.  If the block is
        # tool_result the human didn't type anything; anything else is human text.
        is_tool_result_entry = (
            role == "user"
            and len(content) > 0
            and isinstance(content[0], dict)
            and content[0].get("type") == "tool_result"
        )
        if role == "user" and not is_tool_result_entry:
            _last_human_ts = ts
            _saw_thinking  = False

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")

            # ── Thinking blocks: track flag, don't emit ───────────
            if btype == "thinking":
                _saw_thinking = True
                continue

            # ── Text blocks ───────────────────────────────────────
            if btype == "text":
                text = block.get("text", "").strip()
                if (not text
                        or text.startswith("<system-reminder")
                        or text.startswith("<function_calls>")):
                    continue
                label = _slash_cmd_label(text)
                if label:
                    text = label
                emit_role = role
                if role == "user" and any(text.startswith(p) for p in _NOTE_PREFIXES):
                    emit_role = "note"

                out = {"role": emit_role, "text": text, "ts": ts}

                # Attach timing / token stats to assistant responses
                if role == "assistant":
                    dur_s = None
                    if _last_human_ts and ts:
                        t0 = _parse_ts(_last_human_ts)
                        t1 = _parse_ts(ts)
                        if t0 and t1:
                            dur_s = round((t1 - t0).total_seconds(), 1)
                    out["stats"] = {
                        "output_tokens": usage.get("output_tokens"),
                        "input_tokens":  (usage.get("input_tokens", 0)
                                          + usage.get("cache_read_input_tokens", 0)),
                        "duration_s":    dur_s,
                        "has_thinking":  _saw_thinking,
                    }
                messages.append(out)

            # ── Tool-use blocks ───────────────────────────────────
            elif btype == "tool_use":
                uid    = block.get("id", "")
                name   = block.get("name", "")
                inp    = block.get("input") or {}
                summ   = _tool_summary(name, inp)
                detail = _tool_detail(name, inp)
                res    = result_map.get(uid, {})
                raw_r  = res.get("content", "")
                first_line = (raw_r.split("\n")[0] if raw_r else "")[:120]

                # Tool execution time: tool_use ts → tool_result ts
                dur_s = None
                res_ts = res.get("ts")
                if res_ts and ts:
                    t0 = _parse_ts(ts)
                    t1 = _parse_ts(res_ts)
                    if t0 and t1:
                        dur_s = round((t1 - t0).total_seconds(), 1)

                messages.append({
                    "role":       "tool",
                    "name":       name,
                    "summary":    summ,
                    "detail":     detail,
                    "result":     first_line,
                    "is_error":   res.get("is_error", False),
                    "duration_s": dur_s,
                    "ts":         ts,
                })
            # tool_result blocks consumed via result_map; skip here.

    return messages


@app.route("/api/conversation")
def api_conversation():
    """Return human/assistant turns and tool calls from the Claude Code session log.

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
            raw_lines = []
            with open(jsonl_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            raw_lines.append(json.loads(line))
                        except Exception:
                            pass
            messages = _parse_jsonl_messages(raw_lines)
            if messages:
                break
        except Exception:
            continue

    return jsonify({"messages": messages,
                    "source": str(candidates[0]) if candidates else ""})


@app.route("/api/audio_chunk")
def api_audio_chunk():
    """Serve a slice of raw mono float32 PCM for the audio playback worker.

    Query params:
      f      — file ID (optional; uses default if omitted)
      frame  — first source frame to return
      n      — number of frames to return (capped at 192 000 / 1 s at max SR)
    Response: application/octet-stream, little-endian float32 samples.
    """
    entry, err = _entry_or_404(request.args.get('f'))
    if err:
        return err
    if not entry.finfo:
        return "not ready", 503

    sr    = entry.finfo['sr']
    dur   = entry.finfo['duration_s']
    total = int(dur * sr)
    frame = max(0, min(total - 1, int(request.args.get('frame', 0))))
    n     = min(max(0, int(request.args.get('n', sr))), sr)   # cap at 1 s
    n     = min(n, total - frame)
    if n <= 0:
        return b'', 200, {'Content-Type': 'application/octet-stream',
                          'Cross-Origin-Resource-Policy': 'same-origin'}

    with entry.audio_lock:
        entry.audio_fh.seek(frame)
        audio = entry.audio_fh.read(n, dtype='float32', always_2d=True)

    mono = audio.mean(axis=1) if audio.ndim > 1 and audio.shape[1] > 1 else audio.ravel()
    return mono.astype(np.float32).tobytes(), 200, {
        'Content-Type':                 'application/octet-stream',
        'Cross-Origin-Resource-Policy': 'same-origin',
        'Cache-Control':                'no-store',
    }


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
