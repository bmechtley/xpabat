# xpabat — Bat Echolocation Spectrogram Viewer

Interactive web-based viewer for ultrasonic bat recordings. Built entirely through a conversation with [Claude](https://claude.ai) (Anthropic). See the **Claude Session** button in the app for the full source conversation.

> **Note:** This README was also written by Claude, not the human author.

## Features

- Scrollable, zoomable spectrogram (192 kHz / 96 kHz Nyquist)
- **BatDetect2** neural-net call detection
- Per-call frequency contour overlay with harmonic separation
- Species classification (heuristic profiles for western North America, v1 and v2 classifiers)
- Crosshair cursor with time + frequency readout
- Frequency response flattening (mic-response compensation)
- Raw ↔ call-isolated crossfade view
- Log/linear frequency scale blend
- PSD transport panel with interactive frequency-range navigation
- Overview transport with draggable viewport
- Species accordion legend with per-species show/hide and solo
- Smooth exponential mousewheel zoom, prev/next call navigation
- URL state sync (`?t`, `?vd`, `?fl`, `?fh`, `?call`, `?modal`) for shareable links
- Multi-file support via `?f=` URL parameter
- WavPack (`.wv`) and other ffmpeg-decodable formats supported
- Claude session conversation log with tool call details, timing, and token stats

## Requirements

- Python 3.9+
- macOS with Apple Silicon recommended (MPS GPU acceleration for BatDetect2)
- A high-sample-rate FLAC, WAV, WavPack, or other ffmpeg-decodable bat recording

```
pip install -r requirements.txt
```

## Usage

1. Edit `AUDIO_FILE` in `config.py` to point to your recording.
2. Run:
   ```
   python3 bat_viewer.py [audio_file]
   ```
3. Open **http://localhost:5001** in your browser.
4. Wait for BatDetect2 detection to finish, then explore.

The audio file path can be set in `config.py` or passed as a command-line argument. Detection results and pre-rendered spectrogram tiles are cached to disk on first run; subsequent starts load instantly.

## Detection

Detection uses [BatDetect2](https://github.com/macaodha/batdetect2) (Mac Aodha et al., 2022, bioRxiv). The default model was trained on UK species; it is used here for **detection only**. Species labels are assigned via separate heuristic profiles tuned for western North American species (TABR, EPFU, LACI, LABO, ANPA, Myotis spp.).

**Citation:**
> Mac Aodha O, et al. "Towards a General Approach for Bat Echolocation Detection and Classification." *bioRxiv* (2022). doi:10.1101/2022.12.14.520490

### Contour tracking

Each detected call gets a frequency contour via a continuity-constrained argmax (`track_fundamental`): the tracked frequency is limited to jumps of at most ~15 kHz per STFT frame, preventing the tracker from leaping to a harmonic. Post-detection, `trim_call_contour` applies a two-stage filter:

1. **Floor filter** — drops contour points below 20 kHz (below any western-NA echolocation energy).
2. **Harmonic separation** — splits a bimodal contour if the gap between clusters is ≥ 7 kHz *and* the upper/lower cluster frequency ratio is ≥ 1.55 (indicating a true harmonic at ~2× the fundamental, not just intra-call FM sweep components).

## Deployment

Clone the repo and copy your audio file alongside it, or point `config.py` at the file path. On a server without ffmpeg/ffprobe installed, pre-generate the `.f32raw` decode cache and `.f32meta` sidecar on your development machine and rsync them alongside the audio file; the server will then open the file without needing ffmpeg.

## Files

| File | Description |
|------|-------------|
| `bat_viewer.py` | Entry point — argument parsing, Flask startup |
| `config.py` | Tunable parameters (audio file path, tile size, frequency range, detection threshold) |
| `startup.py` | Audio file loading, cache management, per-file initialisation |
| `detect.py` | BatDetect2 detection pipeline |
| `classify.py` | Heuristic species classifiers (v1 and v2) |
| `species.py` | Species reference profiles and colours |
| `tiles.py` | Spectrogram tile generation (raw, flat, mask) |
| `tile_scheduler.py` | Background tile pre-generation with viewport priority |
| `registry.py` | Per-file state registry for multi-file support |
| `state.py` | Shared Flask app and global state |
| `routes.py` | Flask API routes |
| `analyze_bats.py` | Standalone energy-threshold detector and frequency statistics |
| `analyze_bats_species.py` | Standalone species-level analysis with frequency sweep characterisation |

## A note on vibe coding

This codebase was generated entirely through AI-assisted "vibe coding" — steering Claude by feel rather than careful engineering. It works, but carries the usual hallmarks: layers of iterative fixes, heuristics tuned to one recording, no tests. Treat classification results (detection counts, frequency measurements, species assignments) with appropriate scepticism.
