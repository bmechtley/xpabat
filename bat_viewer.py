#!/usr/bin/env python3
"""
Bat Spectrogram Viewer — interactive web UI
Run:  python3 bat_viewer.py
Open: http://localhost:5000
"""

import io, json, os, threading, warnings
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, send_file, render_template_string, request
from scipy import signal
from scipy.ndimage import label, binary_dilation, binary_erosion
import matplotlib
matplotlib.use("Agg")
from matplotlib.cm import get_cmap
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
AUDIO_FILE    = "/Users/brandon/claude/2025-05-28 1942 bats on campbell 1.flac"
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
SEQ_GAP       = 3.0        # s  — gap larger than this starts a new call sequence / bout
CHUNK_SECS    = 10.0

# ─────────────────────────────────────────────
# Species
# ─────────────────────────────────────────────
PROFILES = [
    {"name":"Eptesicus fuscus",      "short":"EPFU","Fchar":(22,32),"Fmin":(18,28),"dur":(8,20), "sweep":(0.5,3.5)},
    {"name":"Lasiurus cinereus",     "short":"LACI","Fchar":(16,22),"Fmin":(13,20),"dur":(8,25), "sweep":(1.0,5.0)},
    {"name":"Tadarida brasiliensis", "short":"TABR","Fchar":(20,28),"Fmin":(18,25),"dur":(8,25), "sweep":(0.2,1.5)},
    {"name":"Lasiurus borealis",     "short":"LABO","Fchar":(35,50),"Fmin":(25,40),"dur":(8,20), "sweep":(1.5,5.0)},
    {"name":"Antrozous pallidus",    "short":"ANPA","Fchar":(35,50),"Fmin":(28,40),"dur":(3,12), "sweep":(2.0,8.0)},
    {"name":"Myotis (medium)",       "short":"MYLU","Fchar":(40,60),"Fmin":(30,50),"dur":(2,8),  "sweep":(3.0,15.0)},
    {"name":"Myotis (small)",        "short":"MYCA","Fchar":(45,70),"Fmin":(32,55),"dur":(1.5,6),"sweep":(5.0,20.0)},
]
COLORS = {
    "Eptesicus fuscus":      "#4e79a7",
    "Lasiurus cinereus":     "#f28e2b",
    "Tadarida brasiliensis": "#59a14f",
    "Lasiurus borealis":     "#e15759",
    "Antrozous pallidus":    "#b07aa1",
    "Myotis (medium)":       "#76b7b2",
    "Myotis (small)":        "#ff9da7",
    "Unclassified":          "#888888",
}

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
app          = Flask(__name__)
audio_lock   = threading.Lock()
audio_fh     = None
finfo        = {}
all_calls    = []
all_seqs     = []   # per-sequence summary objects
calls_ready  = threading.Event()
progress     = {"done": 0, "total": 1, "status": "starting"}
tile_cache   = {}
tile_lock    = threading.Lock()
_inferno     = get_cmap("inferno")

# ─────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────
def score(call, p):
    return sum([
        p["Fchar"][0] <= call["Fpeak"] <= p["Fchar"][1],
        p["Fmin"][0]  <= call["Fmin"]  <= p["Fmin"][1],
        p["dur"][0]   <= call["dur"]   <= p["dur"][1],
        p["sweep"][0] <= call["sweep"] <= p["sweep"][1],
    ]) / 4

def classify(call):
    scores = {p["name"]: score(call, p) for p in PROFILES}
    best   = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= 0.5 else ("Unclassified", 0.0)

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
        else:
            out.append(dict(c))
    return out

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
                    seg      = Sb[:, i0:i1]
                    prows    = np.argmax(seg, axis=0)
                    fc_hz    = fb[prows]
                    fc_t     = t[i0:i1] + chunk_offset_s
                    Fmax_k   = fc_hz.max() / 1000
                    Fmin_k   = fc_hz.min() / 1000
                    fpeak    = fb[seg.mean(axis=1).argmax()] / 1000
                    tms      = np.linspace(0, dur_s * 1000, len(fc_hz))
                    swp      = (abs(np.polyfit(tms, fc_hz / 1000, 1)[0])
                                if len(fc_hz) > 2 else 0.0)
                    contour  = [[float(ct), float(cf / 1000)]
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
                prows  = np.argmax(seg, axis=0)
                fc_hz  = fb[prows]
                fc_t   = t[i0:i1+1] + chunk_offset_s
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

    merged = merge(raw)
    for idx, c in enumerate(merged):
        sp, conf    = classify(c)
        c["id"]     = idx
        c["species"] = sp
        c["conf"]   = round(conf, 2)
        c["color"]  = COLORS.get(sp, "#888888")
        c["short"]  = next((p["short"] for p in PROFILES if p["name"] == sp), "????")

    # ── Sequence (bout) detection ─────────────────────────────
    # Calls are already sorted by t0 (merge() sorts them).
    # A new sequence starts when the gap to the previous call exceeds SEQ_GAP.
    seq_id = 0
    if merged:
        merged[0]["seq_id"] = 0
        for i in range(1, len(merged)):
            if merged[i]["t0"] - merged[i-1]["t1"] > SEQ_GAP:
                seq_id += 1
            merged[i]["seq_id"] = seq_id

    # Per-sequence stats: count, t_start, t_end, dominant species, mean IPI
    seqs = {}
    for c in merged:
        sid = c["seq_id"]
        if sid not in seqs:
            seqs[sid] = {"t0": c["t0"], "t1": c["t1"], "calls": [], "species_counts": {}}
        seqs[sid]["t1"] = max(seqs[sid]["t1"], c["t1"])
        seqs[sid]["calls"].append(c)
        sp = c["species"]
        seqs[sid]["species_counts"][sp] = seqs[sid]["species_counts"].get(sp, 0) + 1

    for sid, s in seqs.items():
        s["n"]            = len(s["calls"])
        s["dom_species"]  = max(s["species_counts"], key=s["species_counts"].get)
        s["dom_color"]    = COLORS.get(s["dom_species"], "#888888")
        ipis = []
        for i in range(1, len(s["calls"])):
            ipis.append(s["calls"][i]["t0"] - s["calls"][i-1]["t1"])
        s["mean_ipi_ms"] = round(float(np.mean(ipis)) * 1000, 1) if ipis else 0.0
        s["dur_s"]       = round(s["t1"] - s["t0"], 2)
        # Remove the bulky 'calls' list before sending to frontend
        del s["calls"]
        del s["species_counts"]

    # Attach per-sequence stats to each call and build the seqs list
    for c in merged:
        c["seq_n"]       = seqs[c["seq_id"]]["n"]
        c["seq_t0"]      = seqs[c["seq_id"]]["t0"]
        c["seq_t1"]      = seqs[c["seq_id"]]["t1"]

    all_calls.extend(merged)
    # Make seqs JSON-serialisable (int keys → must be list)
    all_seqs.extend(
        {"seq_id": sid, **s} for sid, s in sorted(seqs.items())
    )
    progress["status"] = (f"Done — {len(all_calls)} calls in {len(all_seqs)} sequences"
                          f"  [{detector_label}]")
    calls_ready.set()
    print(progress["status"])

# ─────────────────────────────────────────────
# Tile generation
# ─────────────────────────────────────────────
def make_tile(tidx):
    with tile_lock:
        if tidx in tile_cache:
            return tile_cache[tidx]

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

    vmin = np.percentile(Sdb, 20)
    vmax = np.percentile(Sdb, 99.5)
    arr  = np.clip((Sdb - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb  = (_inferno(arr[::-1, :])[:, :, :3] * 255).astype(np.uint8)

    img  = Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.LANCZOS)
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    with tile_lock:
        if len(tile_cache) > 80:
            del tile_cache[next(iter(tile_cache))]
        tile_cache[tidx] = data
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
                    "calls": list(all_calls),
                    "seqs":  list(all_seqs)})

@app.route("/api/tile/<int:tidx>")
def api_tile(tidx):
    ntiles = int(np.ceil(finfo["duration_s"] / TILE_DURATION))
    if tidx < 0 or tidx >= ntiles:
        return "not found", 404
    data = make_tile(tidx)
    return send_file(io.BytesIO(data), mimetype="image/png",
                     max_age=3600)

@app.route("/api/conversation")
def api_conversation():
    """Return cleaned human/assistant turns from the Claude Code session log."""
    import glob, pathlib
    # Try the session that lives alongside the known project directory
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
                    # Skip injected system reminders, empty, or tool XML blocks
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

    return jsonify({"messages": messages, "source": str(candidates[0]) if candidates else ""})

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
#status-bar { font-size: 11px; color: #f28e2b; margin-left: auto; }

#main { display: flex; flex: 1; overflow: hidden; }

#canvas-col { flex: 1; display: flex; flex-direction: column; overflow: hidden; position: relative; }

#controls { padding: 5px 10px; background: #161616; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
#controls button { background: #2a2a2a; border: 1px solid #3a3a3a; color: #ccc; padding: 3px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; }
#controls button:hover { background: #383838; }
#controls .time-display { color: #aaa; font-size: 11px; margin-left: auto; }

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

#detail { width: 260px; flex-shrink: 0; background: #131313; border-left: 1px solid #222; padding: 14px 12px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
#detail h2 { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid #222; padding-bottom: 6px; }
#detail .empty { color: #555; font-size: 12px; margin-top: 10px; }
#detail .sp-badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: 700; color: #fff; margin-bottom: 8px; }
#detail table { width: 100%; border-collapse: collapse; font-size: 12px; }
#detail td { padding: 3px 0; color: #aaa; }
#detail td:last-child { color: #ddd; text-align: right; }
#legend { margin-top: auto; }
#legend h3 { font-size: 11px; color: #666; margin-bottom: 6px; text-transform: uppercase; }
.leg-row { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #aaa; margin-bottom: 3px; }
.leg-swatch { width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; }

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
  <button id="btn-session" onclick="openSession()" style="margin-left:auto;background:#1a2a2a;border:1px solid #2a3a3a;color:#76b7b2;padding:3px 10px;border-radius:3px;cursor:pointer;font-size:11px;font-family:inherit;">Claude session ↗</button>
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
      <label style="color:#aaa;font-size:11px;display:flex;align-items:center;gap:6px;">
        <input type="checkbox" id="chk-contour" checked> Show contour
      </label>
      <label style="color:#aaa;font-size:11px;display:flex;align-items:center;gap:6px;">
        <input type="checkbox" id="chk-boxes" checked> Show boxes
      </label>
      <label style="color:#aaa;font-size:11px;display:flex;align-items:center;gap:6px;">
        Darken <input type="range" id="slider-darken" min="0" max="90" value="0" style="width:80px;accent-color:#f28e2b"> <span id="darken-val">0%</span>
      </label>
      <label style="color:#aaa;font-size:11px;display:flex;align-items:center;gap:6px;">
        Lin<input type="range" id="slider-log" min="0" max="100" value="0" style="width:80px;accent-color:#76b7b2">Log
      </label>
      <span class="time-display" id="time-display">–</span>
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
    <h2>Call Detail</h2>
    <div id="detail-body"><span class="empty">Click a call to inspect it</span></div>
    <div id="legend">
      <h3>Species</h3>
      <div id="legend-list"></div>
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
  showContour: true,
  showBoxes: true,
  darken: 0,
  logScale: 0,        // 0 = linear, 1 = fully logarithmic
  nyquist: 96,        // kHz — full scrollbar range (set from server)
  seqs: [],           // sequence summary objects from server
  selectedSeqId: null,
  renderPending: false,
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


// ─── Tile loading ─────────────────────────────────────────────
function loadTile(idx) {
  if (S.tileImgs.has(idx)) return;
  const img = new Image();
  S.tileImgs.set(idx, img);
  S.tileReady.set(idx, false);
  img.onload = () => { S.tileReady.set(idx, true); scheduleRender(); };
  img.src = `/api/tile/${idx}`;
}

function ensureTiles() {
  const viewEnd = S.viewStart + S.viewDur;
  const first   = Math.max(0, Math.floor(S.viewStart / S.tileDur) - 1);
  const last    = Math.min(S.nTiles - 1, Math.ceil(viewEnd / S.tileDur));
  for (let i = first; i <= last; i++) loadTile(i);
  // Prefetch neighbours
  if (first > 0) loadTile(first - 1);
  if (last < S.nTiles - 1) loadTile(last + 1);
}

// ─── Rendering ───────────────────────────────────────────────
function scheduleRender() {
  if (S.renderPending) return;
  S.renderPending = true;
  requestAnimationFrame(() => { S.renderPending = false; render(); });
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

  // Precompute per-band tile Y: for each 2-px canvas band, find the source tile Y.
  const BANDS   = Math.ceil(H / 2);
  const bandTileY = new Float32Array(BANDS + 1);
  for (let b = 0; b <= BANDS; b++) {
    const cy  = b * 2;
    const f   = yToF(cy);  // physical frequency at this canvas row
    // Map f back into the tile's LINEAR frequency axis
    const tileFrac = (TILE_FREQ_HIGH - f) / (TILE_FREQ_HIGH - TILE_FREQ_LOW);
    bandTileY[b]   = tileFrac;  // 0=top of tile (high freq), 1=bottom (low freq)
  }

  for (let i = first; i <= last; i++) {
    const img = S.tileImgs.get(i);
    const tS = i * S.tileDur;
    const tE = Math.min((i + 1) * S.tileDur, S.duration);
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

    // X: source slice in image (time axis, always linear)
    const srcX0 = Math.max(0, (S.viewStart - tS) / tileDurActual * img.naturalWidth);
    const srcX1 = Math.min(img.naturalWidth, (viewEnd   - tS) / tileDurActual * img.naturalWidth);
    if (srcX1 <= srcX0) continue;
    const dstX0 = Math.max(YAXIS_W, tToX(tS));
    const dstX1 = Math.min(W, tToX(tE));
    if (dstX1 <= dstX0) continue;
    const srcW = srcX1 - srcX0, dstW = dstX1 - dstX0;

    // Y: draw band by band, warping frequency axis
    for (let b = 0; b < BANDS; b++) {
      const ty0 = bandTileY[b],   ty1 = bandTileY[b + 1];
      if (ty0 < 0 || ty1 > 1 || ty1 <= ty0) continue;
      const imgY0  = ty0 * img.naturalHeight;
      const imgH   = Math.max(0.5, (ty1 - ty0) * img.naturalHeight);
      const canY   = b * 2;
      const canH   = Math.min(2, H - canY);
      ctx.drawImage(img, srcX0, imgY0, srcW, imgH, dstX0, canY, dstW, canH);
    }
  }

  // ── Darken overlay ──
  if (S.darken > 0) {
    ctx.fillStyle   = `rgba(0,0,0,${S.darken})`;
    ctx.fillRect(YAXIS_W, 0, W - YAXIS_W, H);
  }

  // ── Grid lines (log-aware) ──
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth   = 1;
  const gridTicks = [14,16,18,20,25,30,35,40,50,60,70,80,90,96];
  for (const f of gridTicks) {
    if (f <= S.freqLow || f >= S.freqHigh) continue;
    const y = Math.round(fToY(f)) + 0.5;
    ctx.beginPath(); ctx.moveTo(YAXIS_W, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // ── Sequence span bracket ──
  drawSequenceSpans(specW, H);

  // ── Call overlays ──
  if (S.showBoxes || S.showContour) {
    // Draw dimmed / out-of-sequence calls first, then in-sequence on top
    const inGroup = c => S.selectedSeqId !== null && c.seq_id === S.selectedSeqId;
    for (const c of S.calls) {
      if (c.t1 < S.viewStart || c.t0 > viewEnd) continue;
      if (!inGroup(c)) drawCall(c, specW, H);
    }
    for (const c of S.calls) {
      if (c.t1 < S.viewStart || c.t0 > viewEnd) continue;
      if (inGroup(c))  drawCall(c, specW, H);
    }
  }

  // ── Freq axis ──
  drawFreqAxis(W, H);

  // ── Time axis ──
  drawTimeAxis(W, H, specW);

  // ── Ruler (must be before crosshairs so crosshairs render on top) ──
  drawRuler(W, H);

  // ── Crosshairs ──
  drawCrosshairs(W, H);

  // ── Overview ──
  drawOverview();

  // ── Time display ──
  const mid = S.viewStart + S.viewDur / 2;
  document.getElementById('time-display').textContent =
    `View: ${fmt(S.viewStart)} – ${fmt(S.viewStart + S.viewDur)}  |  Duration: ${S.viewDur.toFixed(1)}s`;
}

function drawCall(c, specW, H) {
  const sel    = c === S.selectedCall;
  const inSeq  = S.selectedSeqId !== null && c.seq_id === S.selectedSeqId;
  const hov    = c === S.hoveredCall;
  const col    = c.color;
  // When a sequence is selected, dim everything outside it
  const dimmed = S.selectedSeqId !== null && !inSeq;

  const x0 = tToX(c.t0),  x1 = tToX(c.t1);
  const y0 = fToY(c.Fmax), y1 = fToY(c.Fmin);
  const bw = x1 - x0,     bh = y1 - y0;

  if (S.showBoxes) {
    ctx.globalAlpha = dimmed ? 0.12 : (sel ? 0.45 : (inSeq ? 0.3 : (hov ? 0.35 : 0.18)));
    ctx.fillStyle   = col;
    ctx.fillRect(x0, y0, bw, bh);
    ctx.globalAlpha = dimmed ? 0.25 : 1;
    ctx.strokeStyle = sel ? '#ffffff' : (inSeq ? col : col);
    ctx.lineWidth   = sel ? 2.5 : (inSeq ? 1.8 : (hov ? 1.8 : 0.9));
    ctx.strokeRect(x0, y0, bw, bh);
    ctx.globalAlpha = 1;

    if (!dimmed && bw > 10) {
      ctx.font      = 'bold 10px monospace';
      ctx.fillStyle = col;
      const ly      = y0 > 14 ? y0 - 3 : y0 + bh + 11;
      ctx.fillText(c.short, x0 + 2, ly);
    }
  }

  if (!dimmed && S.showContour && c.contour && c.contour.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth   = sel ? 2 : 1.2;
    ctx.globalAlpha = sel ? 0.95 : (inSeq ? 0.75 : (hov ? 0.85 : 0.45));
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
      ctx.arc(tToX(pmid[0]), fToY(pmid[1]), 3, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
    }
  }
}

function drawSequenceSpans(specW, H) {
  // Draw a subtle bracket for the selected sequence
  if (S.selectedSeqId === null) return;
  const seqObj = S.seqs.find(s => s.seq_id === S.selectedSeqId);
  if (!seqObj) return;
  const viewEnd = S.viewStart + S.viewDur;
  if (seqObj.t1 < S.viewStart || seqObj.t0 > viewEnd) return;

  const x0 = Math.max(YAXIS_W, tToX(seqObj.t0));
  const x1 = Math.min(canvas.width, tToX(seqObj.t1));
  if (x1 <= x0) return;

  const col = seqObj.dom_color;
  // Top and bottom bracket lines
  ctx.strokeStyle = col;
  ctx.lineWidth   = 1.5;
  ctx.globalAlpha = 0.5;
  const bY = 3, bH = H - 6;
  // Top bar
  ctx.beginPath(); ctx.moveTo(x0, bY); ctx.lineTo(x1, bY); ctx.stroke();
  // Bottom bar
  ctx.beginPath(); ctx.moveTo(x0, bY + bH); ctx.lineTo(x1, bY + bH); ctx.stroke();
  // Left tick
  ctx.beginPath(); ctx.moveTo(x0, bY); ctx.lineTo(x0, bY + 10); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x0, bY + bH); ctx.lineTo(x0, bY + bH - 10); ctx.stroke();
  // Right tick
  ctx.beginPath(); ctx.moveTo(x1, bY); ctx.lineTo(x1, bY + 10); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, bY + bH); ctx.lineTo(x1, bY + bH - 10); ctx.stroke();
  ctx.globalAlpha = 1;

  // Label
  ctx.font      = 'bold 10px monospace';
  ctx.fillStyle = col;
  ctx.globalAlpha = 0.8;
  const label = `Seq ${seqObj.seq_id + 1}  (${seqObj.n} calls · ${seqObj.dur_s.toFixed(1)}s · IPI ${seqObj.mean_ipi_ms}ms)`;
  const lx = Math.min(x0 + 4, canvas.width - ctx.measureText(label).width - 4);
  ctx.fillText(label, lx, bY + 13);
  ctx.globalAlpha = 1;
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

function drawOverview() {
  const OW = ovCanvas.width, OH = OV_H;
  octx.clearRect(0, 0, OW, OH);
  octx.fillStyle = '#0d0d0d';
  octx.fillRect(0, 0, OW, OH);

  // Sequence spans as coloured blocks (bottom strip)
  const seqH = Math.round(OH * 0.28);
  for (const seq of S.seqs) {
    const x  = seq.t0 / S.duration * OW;
    const w  = Math.max(2, (seq.t1 - seq.t0) / S.duration * OW);
    const hi = seq.seq_id === S.selectedSeqId;
    octx.fillStyle   = seq.dom_color;
    octx.globalAlpha = hi ? 0.85 : 0.35;
    octx.fillRect(x, OH - seqH, w, seqH);
  }
  octx.globalAlpha = 1;

  // Individual call dots (upper strip)
  const dotH = OH - seqH - 2;
  for (const c of S.calls) {
    const x   = c.t0 / S.duration * OW;
    const w   = Math.max(1, (c.t1 - c.t0) / S.duration * OW);
    const fy  = c.Fpeak >= S.freqLow && c.Fpeak <= S.freqHigh
                ? dotH * (1 - (c.Fpeak - S.freqLow) / (S.freqHigh - S.freqLow))
                : dotH / 2;
    const inS = c.seq_id === S.selectedSeqId;
    octx.fillStyle   = c.color;
    octx.globalAlpha = S.selectedSeqId === null ? 0.7 : (inS ? 1.0 : 0.2);
    octx.fillRect(x, Math.max(0, fy - 2), w, 4);
  }
  octx.globalAlpha = 1;

  // Divider between dot zone and sequence zone
  octx.strokeStyle = '#333';
  octx.lineWidth   = 1;
  octx.beginPath(); octx.moveTo(0, dotH + 1); octx.lineTo(OW, dotH + 1); octx.stroke();

  // Viewport box
  const vx0 = S.viewStart / S.duration * OW;
  const vx1 = (S.viewStart + S.viewDur) / S.duration * OW;
  octx.fillStyle = 'rgba(255,255,255,0.07)';
  octx.fillRect(vx0, 0, vx1 - vx0, OH);
  octx.strokeStyle = 'rgba(255,255,255,0.28)';
  octx.lineWidth = 1;
  octx.strokeRect(vx0, 0, vx1 - vx0, OH);

  // Draggable edge handles — brighter vertical bars
  const hw = 4;
  octx.fillStyle = _ovDrag ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.45)';
  octx.fillRect(vx0,           0, hw, OH);   // left edge
  octx.fillRect(vx1 - hw,      0, hw, OH);   // right edge

  // Border
  octx.strokeStyle = '#222';
  octx.lineWidth   = 1;
  octx.strokeRect(0, 0, OW, OH);
}

// ─── Resize ──────────────────────────────────────────────────
function resize() {
  const cr = canvas.getBoundingClientRect();
  canvas.width  = Math.max(1, Math.round(cr.width));
  canvas.height = Math.max(1, Math.round(cr.height));
  ovCanvas.width  = document.getElementById('overview-wrap').getBoundingClientRect().width;
  ovCanvas.height = OV_H;
  updateScrollbar();
  scheduleRender();
}

// ─── Events ──────────────────────────────────────────────────
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor  = e.deltaY > 0 ? 1.25 : 0.8;
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
    const dt  = dx / OW * S.duration;   // time delta for this pixel delta
    const MIN = 0.5;
    if (_ovDrag === 'pan') {
      S.viewStart = Math.max(0, Math.min(S.duration - _ovVD0, _ovVS0 + dt));
      S.viewDur   = _ovVD0;
    } else if (_ovDrag === 'left') {
      // Left edge: moves viewStart, keeps viewEnd fixed
      const viewEnd  = _ovVS0 + _ovVD0;
      const newStart = Math.max(0, Math.min(viewEnd - MIN, _ovVS0 + dt));
      S.viewStart    = newStart;
      S.viewDur      = viewEnd - newStart;
    } else if (_ovDrag === 'right') {
      // Right edge: keeps viewStart fixed, moves viewEnd
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

  // Always track mouse for crosshair drawing
  S.mouseX = mx; S.mouseY = my;

  const t  = xToT(mx);
  const f  = yToF(my);
  let found = null;
  for (const c of S.calls) {
    if (t >= c.t0 && t <= c.t1 && f >= c.Fmin && f <= c.Fmax) { found = c; break; }
  }
  if (found !== S.hoveredCall) {
    S.hoveredCall = found;
    if (!S.isRuling) canvas.style.cursor = found ? 'pointer' : 'crosshair';
  }
  if (found) showTooltip(found, e.clientX, e.clientY);
  else hideTooltip();
  scheduleRender();
}

function handleClick(e) {
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;

  const t  = xToT(mx);
  const f  = yToF(my);
  let found = null;
  for (const c of S.calls) {
    if (t >= c.t0 && t <= c.t1 && f >= c.Fmin && f <= c.Fmax) { found = c; break; }
  }
  if (found === S.selectedCall) {
    S.selectedCall  = null;
    S.selectedSeqId = null;
  } else {
    S.selectedCall  = found;
    S.selectedSeqId = found ? found.seq_id : null;
  }
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
  const OW  = ovCanvas.width;
  const vx0 = S.viewStart / S.duration * OW;
  const vx1 = (S.viewStart + S.viewDur) / S.duration * OW;
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
    const t = ox / ovCanvas.width * S.duration;
    S.viewStart = Math.max(0, Math.min(S.duration - S.viewDur, t - S.viewDur / 2));
    _ovDrag = 'pan';  // subsequent move pans
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

document.getElementById('btn-zoom-in').onclick  = () => zoomBy(0.6);
document.getElementById('btn-zoom-out').onclick = () => zoomBy(1.6);
document.getElementById('btn-fit').onclick      = () => { S.viewStart = 0; S.viewDur = S.duration; scheduleRender(); };
document.getElementById('chk-contour').onchange = e => { S.showContour = e.target.checked; scheduleRender(); };
document.getElementById('chk-boxes').onchange   = e => { S.showBoxes   = e.target.checked; scheduleRender(); };
document.getElementById('slider-darken').oninput = e => {
  S.darken = e.target.value / 100;
  document.getElementById('darken-val').textContent = e.target.value + '%';
  scheduleRender();
};
document.getElementById('slider-log').oninput = e => {
  S.logScale = e.target.value / 100;
  scheduleRender();
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
  updateScrollbar();
  scheduleRender();
});

window.addEventListener('mouseup', () => { _sbDrag = null; });

function zoomBy(factor) {
  const mid   = S.viewStart + S.viewDur / 2;
  S.viewDur   = Math.max(0.5, Math.min(S.duration, S.viewDur * factor));
  S.viewStart = Math.max(0, Math.min(S.duration - S.viewDur, mid - S.viewDur / 2));
}

function zoomToSeq(seqId) {
  const seq = S.seqs.find(s => s.seq_id === seqId);
  if (!seq) return;
  const pad = Math.max(1.0, (seq.t1 - seq.t0) * 0.2);
  S.viewStart = Math.max(0, seq.t0 - pad);
  S.viewDur   = Math.min(S.duration, (seq.t1 - seq.t0) + 2 * pad);
  scheduleRender();
}

// ─── Tooltip ─────────────────────────────────────────────────
function showTooltip(c, cx, cy) {
  const tt = document.getElementById('tooltip');
  tt.innerHTML = `
    <div class="sp-name" style="color:${c.color}">${c.species}</div>
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

// ─── Detail panel ─────────────────────────────────────────────
function renderDetail(c) {
  const el = document.getElementById('detail-body');
  if (!c) { el.innerHTML = '<span class="empty">Click a call to inspect it</span>'; return; }
  const seq = S.seqs.find(s => s.seq_id === c.seq_id);
  el.innerHTML = `
    <div class="sp-badge" style="background:${c.color}">${c.short} — ${c.species}</div>
    <table>
      <tr><td>Confidence</td><td>${(c.conf*100).toFixed(0)}%</td></tr>
      <tr><td>Time</td><td>${fmt(c.t0)} – ${fmt(c.t1)}</td></tr>
      <tr><td>Duration</td><td>${c.dur.toFixed(1)} ms</td></tr>
      <tr><td>Fmax (sweep start)</td><td>${c.Fmax.toFixed(1)} kHz</td></tr>
      <tr><td>Fpeak (energy)</td><td>${c.Fpeak.toFixed(1)} kHz</td></tr>
      <tr><td>Fmin (sweep end)</td><td>${c.Fmin.toFixed(1)} kHz</td></tr>
      <tr><td>Bandwidth</td><td>${(c.Fmax - c.Fmin).toFixed(1)} kHz</td></tr>
      <tr><td>Sweep rate</td><td>${c.sweep.toFixed(2)} kHz/ms</td></tr>
    </table>
    ${seq ? `
    <div style="margin-top:10px;border-top:1px solid #222;padding-top:8px;">
      <div style="font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px;">
        Sequence ${c.seq_id + 1}
        <span style="float:right;cursor:pointer;color:#f28e2b" onclick="zoomToSeq(${c.seq_id})">zoom ▶</span>
      </div>
      <table>
        <tr><td>Calls in sequence</td><td>${seq.n}</td></tr>
        <tr><td>Sequence start</td><td>${fmt(seq.t0)}</td></tr>
        <tr><td>Sequence end</td><td>${fmt(seq.t1)}</td></tr>
        <tr><td>Sequence duration</td><td>${seq.dur_s.toFixed(1)} s</td></tr>
        <tr><td>Mean IPI</td><td>${seq.mean_ipi_ms} ms</td></tr>
        <tr><td>Dom. species</td><td>${seq.dom_species.split(' ').slice(0,2).join(' ')}</td></tr>
      </table>
    </div>` : ''}
  `;
}

// ─── Legend ──────────────────────────────────────────────────
function buildLegend(colors) {
  const el = document.getElementById('legend-list');
  el.innerHTML = '';
  for (const [sp, col] of Object.entries(colors)) {
    const row = document.createElement('div');
    row.className = 'leg-row';
    row.innerHTML = `<div class="leg-swatch" style="background:${col}"></div><span>${sp}</span>`;
    el.appendChild(row);
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
  S.duration  = info.duration_s;
  S.freqLow   = info.freq_low;
  S.freqHigh  = info.freq_high;
  S.tileDur   = info.tile_duration;
  S.nTiles    = info.n_tiles;
  S.colors    = info.colors;
  S.viewDur   = Math.min(30, S.duration);
  TILE_FREQ_LOW  = info.freq_low;
  TILE_FREQ_HIGH = info.freq_high;
  S.nyquist      = info.freq_high;  // sr/2 in kHz
  updateScrollbar();
  buildLegend(info.colors);

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
  S.calls    = res.calls;
  S.seqs     = res.seqs || [];
  document.getElementById('status-bar').textContent =
    `${S.calls.length} calls · ${S.seqs.length} sequences`;
  scheduleRender();
}

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
</script>
</body>
</html>"""

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def startup():
    global audio_fh
    print(f"Opening {AUDIO_FILE} …")
    audio_fh = sf.SoundFile(AUDIO_FILE)
    finfo.update({
        "sr":         audio_fh.samplerate,
        "nframes":    audio_fh.frames,
        "channels":   audio_fh.channels,
        "duration_s": audio_fh.frames / audio_fh.samplerate,
    })
    print(f"  {finfo['duration_s']:.1f} s  ·  {finfo['sr']:,} Hz  ·  {finfo['channels']} ch")
    progress["status"] = "Detection starting…"
    t = threading.Thread(target=run_detection, daemon=True)
    t.start()

if __name__ == "__main__":
    startup()
    print("Starting server → http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
