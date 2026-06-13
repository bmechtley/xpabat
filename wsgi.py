"""
WSGI entry point for gunicorn.

Called by:  gunicorn wsgi:app --bind 0.0.0.0:$PORT

Does everything app.py's main() does *except* call app.run() —
gunicorn handles the HTTP server.
"""
import os
import config

# Allow overriding the audio file via environment variable (for production).
# Falls back to the hardcoded default in config.py.
if os.environ.get("AUDIO_FILE"):
    config.AUDIO_FILE = os.environ["AUDIO_FILE"]
    config.CACHE_FILE = os.path.splitext(config.AUDIO_FILE)[0] + ".calls.json"

# Import routes to register all @app.route(...) handlers with the Flask app.
# Must happen before startup() so the routes exist when the scheduler fires.
import routes  # noqa: E402, F401

from state import app      # noqa: E402
from startup import startup  # noqa: E402

# Run the full startup: open audio, load/cache detections, start tile scheduler.
# --redetect is only for local dev; never force re-detection in production.
startup(redetect=False)
