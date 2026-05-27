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

This codebase was generated entirely through AI-assisted "vibe coding" — steering Claude by feel rather than careful engineering. The UI works well enough, but carries the usual hallmarks: layers of iterative fixes, heuristics tuned to only a few recordings, no tests. Treat species information and classification results (detection counts, frequency measurements, species assignments) with intense scepticism.

## References

### Software

- Mac Aodha O, et al. "Towards a General Approach for Bat Echolocation Detection and Classification." *bioRxiv* (2022). doi:[10.1101/2022.12.14.520490](https://doi.org/10.1101/2022.12.14.520490) — **BatDetect2** neural-net detector used for call detection.

- Cannam C, Landone C, Sandler M. "Sonic Visualiser: An Open Source Application for Viewing, Analysing, and Annotating Music Audio Files." *Proceedings of the 18th ACM International Conference on Multimedia* (2010). doi:[10.1145/1873951.1874248](https://doi.org/10.1145/1873951.1874248) — UI design reference.

### Species accounts

References used in the species classification profiles (`species.py`), listed chronologically.

- Williams et al. (1973) Anim Behav 21:302–321. [Google Scholar](https://scholar.google.com/scholar?q=Williams+1973+bat+pursuit+echolocation+Animal+Behaviour) — TABR

- Watkins (1977) Mammalian Species 80:1–6. [Google Scholar](https://scholar.google.com/scholar?q=Watkins+1977+Myotis+velifer+cave+myotis+Mammalian+Species) — MYVE

- Simmons & Stein (1980) J Comp Physiol 135:335–353. [Google Scholar](https://scholar.google.com/scholar?q=Simmons+Stein+1980+acoustic+interference+bat+echolocation+Journal+Comparative+Physiology) — TABR

- Fenton & Bell (1981) J Mammal 62:317–324. [Google Scholar](https://scholar.google.com/scholar?q=Fenton+Bell+1981+recognition+insectivorous+bats+echolocation+Journal+Mammalogy) — EPFU, MYVE, MYYU, MYCA

- Bell (1982) Behav Ecol Sociobiol 10:1–6. [Google Scholar](https://scholar.google.com/scholar?q=Bell+1982+Antrozous+pallidus+pallid+bat+prey+Behavioral+Ecology+Sociobiology) — ANPA

- Czaplewski (1983) Mammalian Species 199:1–5. [Google Scholar](https://scholar.google.com/scholar?q=Czaplewski+1983+Parastrellus+hesperus+western+pipistrelle+Mammalian+Species) — PEHE

- Hermanson & O'Shea (1983) Mammalian Species 213:1–8. [Google Scholar](https://scholar.google.com/scholar?q=Hermanson+O%27Shea+1983+Antrozous+pallidus+Mammalian+Species) — ANPA

- Hoffmeister (1986) Mammals of Arizona, Univ. of Arizona Press. [Google Scholar](https://scholar.google.com/scholar?q=Hoffmeister+1986+Mammals+Arizona+University+Arizona+Press) — MYYU, MYCA, PEHE

- Betts (1998) J Mammal 79:1098–1105. [Google Scholar](https://scholar.google.com/scholar?q=Betts+1998+Lasiurus+cinereus+hoary+bat+habitat+Journal+Mammalogy) — LACI

- Cryan (2003) J Mammal 84:1020–1028. [Google Scholar](https://scholar.google.com/scholar?q=Cryan+2003+seasonal+distribution+migratory+tree+bats+Lasiurus+Journal+Mammalogy) — LACI

- Hoofer & Van Den Bussche (2003) J Mammal 84:698–707. [Google Scholar](https://scholar.google.com/scholar?q=Hoofer+Van+Den+Bussche+2003+molecular+phylogenetics+Pipistrellus+Journal+Mammalogy) — PEHE

- O'Shea & Bogan (2003) Monitoring Trends in Bat Populations of the US and Territories, USGS. [Google Scholar](https://scholar.google.com/scholar?q=O%27Shea+Bogan+2003+monitoring+trends+bat+populations+United+States+territories+USGS) — TABR, MYVE

- Whitaker (2004) J Mammal 85:1–13. [Google Scholar](https://scholar.google.com/scholar?q=Whitaker+2004+food+habits+big+brown+bat+Eptesicus+fuscus+Journal+Mammalogy) — EPFU

- Simmons (2005) Mammal Species of the World, 3rd ed. [Google Scholar](https://scholar.google.com/scholar?q=Simmons+2005+Mammal+Species+of+the+World+Wilson+Reeder) — EPFU, LACI, LBOS, ANPA, MYYU, MYCA

- Hoofer et al. (2006) J Mammal 87:252–257. [Google Scholar](https://scholar.google.com/scholar?q=Hoofer+2006+molecular+systematics+Lasiurus+red+bat+Journal+Mammalogy) — LBOS

- Valdez & Cryan (2009) J Mammal 90:1308–1320. [Google Scholar](https://scholar.google.com/scholar?q=Valdez+Cryan+2009+Lasiurus+blossevillii+western+red+bat+Journal+Mammalogy) — LBOS
