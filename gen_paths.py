"""
Path helpers for generated data.

All generated data lives under a ``generated/`` tree next to the audio files:

    generated/
      {audio_stem}/
        calls/
          batdetect2.json        -- call metadata only (no contour arrays)
          tadarida.json          -- Tadarida-D metadata
          hilbert.json           -- base Hilbert contour arrays
          cwt.json               -- CWT contour arrays
          stft.json
          chirp.json
          sharp.json
        spectrograms/
          norm.json
          reass_norm.json
          raw/tile_NNNN.png
          flat/tile_NNNN.png
          reassigned/tile_NNNN.png
          flat_reassigned/tile_NNNN.png
          mask/tile_NNNN.png

Contour files are plain JSON arrays parallel to the calls array:
    [ [[t, f], ...], null, [[t, f], ...], ... ]
Index i = contour for call i; null = no contour for that call.

Legacy paths (old flat layout next to audio files) are returned by
old_* helpers so code can fall back to them during migration.
"""

import os

# ── Constants ─────────────────────────────────────────────────────────────────

# Subdirectory names for each spectrogram tile type
SPEC_SUBDIRS = {
    "raw":             "raw",
    "flat":            "flat",
    "reassigned":      "reassigned",
    "flat_reassigned": "flat_reassigned",
    "mask":            "mask",
}

# Tile filename prefixes (legacy naming kept inside subdirs)
TILE_PREFIX = {
    "raw":             "tile",
    "flat":            "flat_tile",
    "reassigned":      "reassigned_tile",
    "flat_reassigned": "flat_reassigned_tile",
    "mask":            "mask_tile",
}

# Contour method → file name
CONTOUR_FILES = {
    "hilbert": "hilbert.json",
    "cwt":     "cwt.json",
    "stft":    "stft.json",
    "chirp":   "chirp.json",
    "sharp":   "sharp.json",
}

# Contour method → key in call dict
CONTOUR_KEY = {
    "hilbert": "contour",
    "cwt":     "contour_cwt",
    "stft":    "contour_stft",
    "chirp":   "contour_chirp",
    "sharp":   "contour_sharp",
}

# Reverse: call dict key → contour method
KEY_CONTOUR = {v: k for k, v in CONTOUR_KEY.items()}


# ── New-format path helpers ───────────────────────────────────────────────────

def stem(audio_path: str) -> str:
    return os.path.splitext(os.path.basename(audio_path))[0]


def generated_dir(audio_path: str) -> str:
    """Root generated dir for this audio file."""
    parent = os.path.dirname(os.path.abspath(audio_path))
    return os.path.join(parent, "generated", stem(audio_path))


def calls_dir(audio_path: str) -> str:
    return os.path.join(generated_dir(audio_path), "calls")


def spectrograms_dir(audio_path: str) -> str:
    return os.path.join(generated_dir(audio_path), "spectrograms")


def calls_meta_path(audio_path: str, detector: str = "batdetect2") -> str:
    """Path to the base call metadata file (no contour arrays)."""
    return os.path.join(calls_dir(audio_path), f"{detector}.json")


def contour_path(audio_path: str, method: str) -> str:
    """Path to a per-type contour array file."""
    fname = CONTOUR_FILES.get(method)
    if fname is None:
        raise ValueError(f"Unknown contour method: {method!r}")
    return os.path.join(calls_dir(audio_path), fname)


def tadarida_calls_path(audio_path: str) -> str:
    return os.path.join(calls_dir(audio_path), "tadarida.json")


def norm_json_path(audio_path: str) -> str:
    return os.path.join(spectrograms_dir(audio_path), "norm.json")


def reass_norm_json_path(audio_path: str) -> str:
    return os.path.join(spectrograms_dir(audio_path), "reass_norm.json")


def tile_subdir(audio_path: str, tile_type: str) -> str:
    """Directory for one spectrogram tile type."""
    subdir = SPEC_SUBDIRS.get(tile_type, tile_type)
    return os.path.join(spectrograms_dir(audio_path), subdir)


def tile_path(audio_path: str, tile_type: str, tidx: int) -> str:
    prefix = TILE_PREFIX.get(tile_type, tile_type)
    return os.path.join(tile_subdir(audio_path, tile_type),
                        f"{prefix}_{tidx:04d}.png")


# ── Legacy (old) path helpers — used for migration fallback ──────────────────

def old_tile_dir(audio_path: str) -> str:
    return os.path.splitext(audio_path)[0] + "_tiles"


def old_tile_path(audio_path: str, tile_type: str, tidx: int) -> str:
    prefix = TILE_PREFIX.get(tile_type, tile_type)
    return os.path.join(old_tile_dir(audio_path), f"{prefix}_{tidx:04d}.png")


def old_norm_json_path(audio_path: str) -> str:
    return os.path.join(old_tile_dir(audio_path), "norm.json")


def old_reass_norm_json_path(audio_path: str) -> str:
    return os.path.join(old_tile_dir(audio_path), "reass_norm.json")


def old_calls_json(audio_path: str) -> str:
    return os.path.splitext(audio_path)[0] + ".calls.json"


def old_tadarida_json(audio_path: str) -> str:
    return os.path.splitext(audio_path)[0] + ".tadarida.calls.json"


# ── Backwards-compat tile resolver ───────────────────────────────────────────

def resolve_tile_path(audio_path: str, tile_type: str, tidx: int) -> str:
    """Return the path to use for a tile.

    Checks the new generated path first, then falls back to the old
    ``_tiles/`` path.  Always returns the *new* path when neither exists
    (so newly rendered tiles land in the right place).
    """
    new = tile_path(audio_path, tile_type, tidx)
    if os.path.exists(new):
        return new
    old = old_tile_path(audio_path, tile_type, tidx)
    if os.path.exists(old):
        return old
    return new  # neither exists — caller will create at new path


def resolve_norm_json(audio_path: str) -> str:
    new = norm_json_path(audio_path)
    if os.path.exists(new):
        return new
    old = old_norm_json_path(audio_path)
    if os.path.exists(old):
        return old
    return new


def resolve_reass_norm_json(audio_path: str) -> str:
    new = reass_norm_json_path(audio_path)
    if os.path.exists(new):
        return new
    old = old_reass_norm_json_path(audio_path)
    if os.path.exists(old):
        return old
    return new
