"""Simulated Michaelis-Menten measurements for the linear regression
example (§6.2.1).

At the nine substrate concentrations of Exercise 6.5, rates follow
v = vmax S / (KM + S), with vmax = 2.5 mM/s and KM = 0.8 mM, plus
Gaussian noise of standard deviation 0.1 mM/s. We draw 200 replicate
datasets once, and store them, so that the MATLAB and Python examples
analyse exactly the same numbers.
"""
from pathlib import Path
import numpy as np

S = np.array([0.15, 0.35, 0.65, 1.2, 2.0, 3.0, 4.0, 5.0, 6.0])
vmax, KM, sigma, replicates = 2.5, 0.8, 0.1, 200
rng = np.random.default_rng(2021)
v = vmax * S / (KM + S) + sigma * rng.standard_normal((replicates, len(S)))
out = Path(__file__).resolve().parent.parent / "data" / "mm_replicates.csv"
np.savetxt(out, v, delimiter=",", fmt="%.6f",
           header=",".join(f"S={s:g}" for s in S), comments="")
print(f"wrote {out} ({replicates} x {len(S)})")
