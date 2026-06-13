#!/usr/bin/env python3
"""
Bat Spectrogram Viewer — interactive web UI
Run:  python3 bat_viewer.py [audio_file]
Open: http://localhost:5001
"""
import os, argparse
import config

def main():
    parser = argparse.ArgumentParser(
        description="Bat echolocation spectrogram viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", nargs="?", default=config.AUDIO_FILE,
        help="Path to FLAC/WAV bat recording")
    parser.add_argument("--port", type=int, default=os.environ.get('PORT', 80),
        help="HTTP port to listen on")
    parser.add_argument("--redetect", action="store_true",
        help="Ignore cached detections and re-run BatDetect2")
    args = parser.parse_args()

    # Override config before other modules see it
    config.AUDIO_FILE = args.file
    config.CACHE_FILE = os.path.splitext(args.file)[0] + ".calls.json"

    if args.redetect:
        print("--redetect: ignoring cache, re-running detection.")

    # Import routes to register them with the Flask app (must be after config override)
    import routes  # noqa: F401
    from state import app
    from startup import startup

    startup(redetect=args.redetect)
    print(f"\nStarting server → http://localhost:{args.port}  (Ctrl-C to stop)\n")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
