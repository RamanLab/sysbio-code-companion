"""Small helpers shared by the companion's Python examples.

They only deal with bookkeeping (where outputs go, a common figure style),
so that each example script can concentrate on the science.
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def out_dir(script_file):
    """Return (and create) the folder where an example writes its outputs."""
    d = Path(script_file).resolve().parent / "out" / "python"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(x):
    """Convert NumPy types to plain Python, and NaN/inf to None (JSON null)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def save_results(script_file, results):
    """Write the key numbers of an example to out/python/results.json."""
    path = out_dir(script_file) / "results.json"
    path.write_text(json.dumps(_clean(results), indent=2, sort_keys=True) + "\n")


def style():
    """A common, plain figure style for the whole companion."""
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, script_file, name):
    """Save a figure as out/python/<name>.png."""
    fig.savefig(out_dir(script_file) / f"{name}.png", dpi=150, bbox_inches="tight")


def show():
    """Show figures when run interactively; do nothing in CI."""
    if not os.environ.get("CI"):
        plt.show()

def uniform(seed, n):
    """Random numbers that are identical in MATLAB and Python.

    Returns n numbers in (0, 1), from the "minimal standard" generator of
    Park and Miller (x <- 48271 x mod 2^31 - 1), started at the integer
    seed (1 to 2^31 - 2). The same numbers come from sbp.uniform in MATLAB.
    Use it where the two languages must make the same random choices;
    otherwise, use numpy.random.
    """
    m = 2147483647
    x = seed % m
    u = np.empty(n)
    for k in range(n):
        x = (48271 * x) % m
        u[k] = x / m
    return u
