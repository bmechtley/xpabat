#!/usr/bin/env python3
"""
One-shot migration: old flat layout → new generated/ tree.

Old layout (next to each audio file):
    <stem>.calls.json                 monolithic calls + all contour types
    <stem>.tadarida.calls.json        tadarida calls
    <stem>_tiles/tile_NNNN.png        raw tiles
    <stem>_tiles/flat_tile_NNNN.png
    <stem>_tiles/mask_tile_NNNN.png
    <stem>_tiles/reassigned_tile_NNNN.png
    <stem>_tiles/flat_reassigned_tile_NNNN.png
    <stem>_tiles/norm.json
    <stem>_tiles/reass_norm.json

New layout:
    generated/<stem>/calls/batdetect2.json   metadata only (no contours)
    generated/<stem>/calls/<method>.json     per-type contour arrays
    generated/<stem>/calls/tadarida.json
    generated/<stem>/spectrograms/norm.json
    generated/<stem>/spectrograms/reass_norm.json
    generated/<stem>/spectrograms/raw/tile_NNNN.png
    generated/<stem>/spectrograms/flat/flat_tile_NNNN.png
    generated/<stem>/spectrograms/mask/mask_tile_NNNN.png
    generated/<stem>/spectrograms/reassigned/reassigned_tile_NNNN.png
    generated/<stem>/spectrograms/flat_reassigned/flat_reassigned_tile_NNNN.png

Usage:
    python3 migrate.py [audio_dir] [--move|--copy] [--json-max-mb N]

  audio_dir       directory containing audio files (default: current dir)
  --move          move tile PNGs (default; frees disk immediately)
  --copy          copy tile PNGs instead of moving
  --json-max-mb   skip JSON splitting for files larger than this (default 600);
                  those need a re-detect to produce split files (avoids OOM).

Tiles are always migrated (cheap rename, no json.load).  The calls JSON is only
split for files under --json-max-mb; bigger ones are left in place and the app
falls back to loading them the old way (or you re-detect to get split files).
"""

import json, os, shutil, sys

import gen_paths as _gp

AUDIO_EXTS = {'.flac', '.wav', '.wv', '.mp3', '.ogg', '.aif', '.aiff'}

# Old tile-prefix → new tile_type
_PREFIX_TYPE = {
    "tile":                  "raw",
    "flat_tile":             "flat",
    "mask_tile":             "mask",
    "reassigned_tile":       "reassigned",
    "flat_reassigned_tile":  "flat_reassigned",
}


def _iter_audio(audio_dir):
    for fn in sorted(os.listdir(audio_dir)):
        p = os.path.join(audio_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
            yield p


def migrate_tiles(audio_path, move=True):
    """Move/copy all tile PNGs + norm JSONs into the new layout."""
    old_dir = _gp.old_tile_dir(audio_path)
    if not os.path.isdir(old_dir):
        return 0
    op = shutil.move if move else shutil.copy2
    n = 0
    for fn in os.listdir(old_dir):
        src = os.path.join(old_dir, fn)
        if not os.path.isfile(src):
            continue
        if fn == "norm.json":
            dst = _gp.norm_json_path(audio_path)
        elif fn == "reass_norm.json":
            dst = _gp.reass_norm_json_path(audio_path)
        elif fn.endswith(".png"):
            # Split "<prefix>_NNNN.png" → prefix + index
            base = fn[:-4]
            idx_str = base.rsplit("_", 1)[-1]
            prefix  = base[:-(len(idx_str) + 1)]
            tile_type = _PREFIX_TYPE.get(prefix)
            if tile_type is None:
                continue
            dst = os.path.join(_gp.tile_subdir(audio_path, tile_type, 3), fn)
        else:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        op(src, dst)
        n += 1
    print(f"    tiles: {n} files {'moved' if move else 'copied'}")
    return n


def migrate_calls(audio_path, json_max_mb=600):
    """Split the old monolithic calls JSON into the new v6 split format."""
    old = _gp.old_calls_json(audio_path)
    if not os.path.exists(old):
        return False
    size_mb = os.path.getsize(old) / 1e6
    if size_mb > json_max_mb:
        print(f"    calls: SKIP split ({size_mb:.0f} MB > {json_max_mb} MB) "
              f"— re-detect to produce split files")
        return False

    print(f"    calls: loading {size_mb:.0f} MB…", flush=True)
    with open(old) as fh:
        cache = json.load(fh)
    calls = cache.get("calls", [])

    out_dir = _gp.calls_dir(audio_path)
    os.makedirs(out_dir, exist_ok=True)

    # Metadata file (strip contour keys)
    meta_calls = []
    for c in calls:
        d = {k: v for k, v in c.items() if k not in _gp.CONTOUR_KEY.values()}
        meta_calls.append(d)
    meta = {
        "version":     6,
        "audio_file":  cache.get("audio_file", audio_path),
        "audio_mtime": cache.get("audio_mtime", os.path.getmtime(audio_path)),
        "detector":    cache.get("detector", "cached"),
        "bd2_thresh":  cache.get("bd2_thresh"),
        "calls":       meta_calls,
    }
    with open(_gp.calls_meta_path(audio_path), "w") as fh:
        json.dump(meta, fh)

    # Per-type contour arrays (compact even-encoding via startup.encode_contour)
    from startup import encode_contour
    for method, key in _gp.CONTOUR_KEY.items():
        arr = []
        any_present = False
        for c in calls:
            enc = encode_contour(c.get(key))
            arr.append(enc)
            if enc is not None:
                any_present = True
        if any_present:
            with open(_gp.contour_path(audio_path, method), "w") as fh:
                json.dump(arr, fh)

    print(f"    calls: split {len(calls)} calls into "
          f"batdetect2.json + {len(_gp.CONTOUR_KEY)} contour files")
    return True


def migrate_tadarida(audio_path, move=True):
    old = _gp.old_tadarida_json(audio_path)
    if not os.path.exists(old):
        return False
    dst = _gp.tadarida_calls_path(audio_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    (shutil.move if move else shutil.copy2)(old, dst)
    print(f"    tadarida: {'moved' if move else 'copied'}")
    return True


def reencode_split(audio_path):
    """Re-encode existing split contour files in place to the compact form.

    Reads each generated/<stem>/calls/<method>.json, decodes every contour
    (handles both pairs and already-compact forms), re-encodes with the
    even-spacing optimisation, and rewrites.  Idempotent.  Processes one
    contour file at a time to keep memory bounded.
    """
    from startup import encode_contour, decode_contour
    stem = _gp.stem(audio_path)
    any_done = False
    for method in _gp.CONTOUR_KEY:
        p = _gp.contour_path(audio_path, method)
        if not os.path.exists(p):
            continue
        before = os.path.getsize(p) / 1e6
        with open(p) as fh:
            arr = json.load(fh)
        out = []
        for enc in arr:
            dec = decode_contour(enc) if enc else None
            out.append(encode_contour(dec) if dec is not None else None)
        with open(p, "w") as fh:
            json.dump(out, fh)
        after = os.path.getsize(p) / 1e6
        print(f"    {method:8s}: {before:6.1f} MB → {after:6.1f} MB "
              f"({100*(1-after/before):+.0f}%)")
        any_done = True
    if not any_done:
        print(f"    (no split contour files found for {stem})")


def main():
    args = sys.argv[1:]
    reencode = "--reencode" in args
    move = "--copy" not in args
    json_max_mb = 600
    audio_dir = "."
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--json-max-mb":
            json_max_mb = float(args[i + 1]); i += 2; continue
        if a in ("--move", "--copy", "--reencode"):
            i += 1; continue
        audio_dir = a; i += 1
    audio_dir = audio_dir or "."

    if reencode:
        print(f"Re-encoding split contour files in {os.path.abspath(audio_dir)}\n")
        for audio_path in _iter_audio(audio_dir):
            print(f"  {os.path.basename(audio_path)}")
            reencode_split(audio_path)
            print()
        print("Done.")
        return

    print(f"Migrating audio in {os.path.abspath(audio_dir)}  "
          f"(mode={'move' if move else 'copy'}, json_max_mb={json_max_mb})\n")

    for audio_path in _iter_audio(audio_dir):
        print(f"  {os.path.basename(audio_path)}")
        migrate_calls(audio_path, json_max_mb=json_max_mb)
        migrate_tadarida(audio_path, move=move)
        migrate_tiles(audio_path, move=move)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
