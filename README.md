# xpabat — Bat Echolocation Spectrogram Viewer

Interactive web-based viewer for ultrasonic bat recordings. Built entirely through a conversation with [Claude](https://claude.ai) (Anthropic). See the **Claude Session** button in the app for the full source conversation.

![Bat spectrogram viewer](https://img.shields.io/badge/bat-calls-detected-orange)

## Features

- Scrollable, zoomable spectrogram (192 kHz / 96 kHz Nyquist)
- **BatDetect2** neural-net call detection (10,000+ calls on a 21-min recording)
- Per-call frequency contour overlay with harmonic separation
- Species classification (heuristic profiles for western North America)
- Call sequence / bout grouping with inter-pulse interval stats
- Crosshair cursor with time + frequency readout
- Click-and-drag measurement ruler (Δt, Δf)
- Log/linear frequency scale blend
- Frequency range scrollbar
- Overview transport with draggable viewport
- Species accordion legend with per-species show/hide
- Smooth exponential mousewheel zoom

## Requirements

- Python 3.9+
- macOS with Apple Silicon recommended (MPS GPU acceleration for BatDetect2)
- A high-sample-rate FLAC or WAV bat recording

```
pip install -r requirements.txt
```

## Usage

1. Edit `AUDIO_FILE` at the top of `bat_viewer.py` to point to your recording.
2. Run:
   ```
   python3 bat_viewer.py
   ```
3. Open **http://localhost:5001** in your browser.
4. Wait ~7 minutes for BatDetect2 detection to finish (Apple GPU), then explore.

Detection results and pre-warped spectrogram tiles are cached to disk on first run. Subsequent starts load instantly.

## Detection

Detection uses [BatDetect2](https://github.com/macaodha/batdetect2) (Mac Aodha et al., 2023, PLOS Computational Biology). The default model was trained on UK species; it is used here for **detection only**. Species labels are assigned via separate heuristic profiles tuned for western North American species (TABR, EPFU, LACI, LABO, ANPA, Myotis spp.).

**Citation:**
> Mac Aodha O, et al. "Towards a General Approach for Bat Echolocation Detection and Classification." *PLOS Computational Biology* 19(8): e1011333 (2023). https://doi.org/10.1371/journal.pcbi.1011333

### Contour tracking

Each detected call gets a frequency contour via a continuity-constrained argmax (`track_fundamental`): the tracked frequency is limited to jumps of at most ~15 kHz per STFT frame, preventing the tracker from leaping to a harmonic. Post-detection, `trim_call_contour` applies a two-stage filter:

1. **Floor filter** — drops contour points below 20 kHz (below any western-NA echolocation energy).
2. **Harmonic separation** — splits a bimodal contour if the gap between clusters is ≥ 7 kHz *and* the upper/lower cluster frequency ratio is ≥ 1.55 (indicating a true harmonic at ~2× the fundamental, not just intra-call FM sweep components).

### Sequence (bout) grouping

Calls are grouped into sequences using a 0.5 s inter-call gap threshold (`SEQ_GAP`). Sequence assignments are recomputed on every cache load, so changing the threshold takes effect immediately without re-detecting.

## Deployment

Copy the audio file, `.calls.json` cache, and `_tiles/` directory alongside `bat_viewer.py`. The cache is validated by filename only (not modification time), so copying files across machines works without re-detection.

## Files

| File | Description |
|------|-------------|
| `bat_viewer.py` | Main application (Flask backend + embedded HTML/JS frontend) |
| `analyze_bats.py` | Simple energy-threshold detector and frequency statistics |
| `analyze_bats_species.py` | Species-level analysis with frequency sweep characterisation |

## A note on vibe coding

This codebase was generated entirely through AI-assisted "vibe coding" — steering Claude by feel rather than careful engineering. It works, but carries the usual hallmarks: layers of iterative fixes, heuristics tuned to one recording, no tests. Use the scientific outputs (detection counts, frequency measurements) with appropriate scepticism.
