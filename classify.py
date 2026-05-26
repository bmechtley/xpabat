from config import MERGE_GAP
from species import PROFILES


# ─────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────
def score_v1(call, p):
    """v1: fraction of 4 basic criteria met (Fpeak, Fmin, dur, sweep)."""
    return sum([
        p["Fchar"][0] <= call["Fpeak"] <= p["Fchar"][1],
        p["Fmin"][0]  <= call["Fmin"]  <= p["Fmin"][1],
        p["dur"][0]   <= call["dur"]   <= p["dur"][1],
        p["sweep"][0] <= call["sweep"] <= p["sweep"][1],
    ]) / 4

def score_v2(call, p):
    """v2: fraction of 6 criteria met — adds bandwidth and CF fraction.

    bw (Fmax−Fmin) distinguishes narrow-band CF bats (TABR ~2–15 kHz) from
    broadband FM bats (Myotis ~15–40 kHz).
    cf_frac (fraction of contour frames within 2 kHz of median) distinguishes
    mostly-CF bats (TABR 0.45–1.0) from steep-FM bats (Myotis 0.02–0.25).
    Both features are computed in trim_call_contour() from the cleaned contour.
    """
    bw = call.get("bw", call["Fmax"] - call["Fmin"])
    cf = call.get("cf_frac", 0.5)   # default neutral if missing
    return sum([
        p["Fchar"][0] <= call["Fpeak"] <= p["Fchar"][1],
        p["Fmin"][0]  <= call["Fmin"]  <= p["Fmin"][1],
        p["dur"][0]   <= call["dur"]   <= p["dur"][1],
        p["sweep"][0] <= call["sweep"] <= p["sweep"][1],
        p["bw"][0]    <= bw           <= p["bw"][1],
        p["cf_frac"][0] <= cf         <= p["cf_frac"][1],
    ]) / 6

def _best_species(raw_scores):
    """Given {name: raw_score} return (best_name, raw_score) or Unclassified."""
    weighted = {n: s * next(p.get("prior", 1.0) for p in PROFILES if p["name"] == n)
                for n, s in raw_scores.items()}
    best = max(weighted, key=weighted.get)
    return (best, round(raw_scores[best], 2)) if raw_scores[best] >= 0.5 else ("Unclassified", 0.0)

def classify_v1(call):
    raw = {p["name"]: score_v1(call, p) for p in PROFILES}
    return _best_species(raw)

def classify_v2(call):
    raw = {p["name"]: score_v2(call, p) for p in PROFILES}
    return _best_species(raw)

# Keep classify() as an alias for v2 (used during fresh detection)
def classify(call):
    return classify_v2(call)

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
            # Contour search windows extend ±1ms past t0/t1, so the seam of
            # two merged calls can have time-reversed points.  Sorting by time
            # interleaves the two frequency tracks, creating zig-zag loops.
            # Instead keep only time-forward points (strict monotone filter):
            # the first contour's points take priority at the overlap seam.
            mono = [p["contour"][0]]
            for pt in p["contour"][1:]:
                if pt[0] > mono[-1][0]:
                    mono.append(pt)
            p["contour"] = mono
        else:
            out.append(dict(c))
    return out
