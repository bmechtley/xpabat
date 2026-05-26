from matplotlib.cm import get_cmap
from flask import Flask

app      = Flask(__name__)
_inferno = get_cmap("inferno")

# Priority tile scheduler — created once in startup(), shared across all files.
scheduler = None
