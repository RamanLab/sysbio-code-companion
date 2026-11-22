"""Flux variability analysis (FVA) of the toy metabolic network.

FBA gives one optimal flux distribution, but often many exist. FVA finds,
for every reaction, the smallest and largest flux compatible with
(a chosen fraction of) the optimal growth rate.
"""
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import sbp

# --- The network (same as toy_fba.py; book Fig. 8.1)
rxns = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "b1", "b2", "vbio"]
#             v1  v2  v3  v4  v5  v6  v7  b1  b2 vbio
S = np.array([[-1, -1,  0,  0,  0,  0,  0,  1,  0,   0],   # A
              [ 1,  0,  1, -1, -1,  0,  0,  0,  0,   0],   # B
              [ 0,  1, -1,  0,  0,  0,  0,  0,  0,   0],   # C
              [ 0,  0,  0,  2,  0, -1, -1,  0,  0,   0],   # D
              [ 0,  0,  0,  0,  0,  1,  0,  0, -1,   0],   # E
              [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  -1],   # F
              [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  -2],   # G
              [ 1,  0,  1,  0,  0,  0, -1,  0,  0, -10]])  # ATP
lb = np.array([0, 0, 0, -1000, 0, -1000, 0, 0, -1000, 0])
ub = np.full(len(rxns), 1000)
bounds = list(zip(lb, ub))
n = len(rxns)
c = np.zeros(n)
c[rxns.index("vbio")] = 1
zeros = np.zeros(S.shape[0])

# --- Step 1: FBA, to find the optimal growth rate f*
fba = linprog(-c, A_eq=S, b_eq=zeros, bounds=bounds, method="highs")
f_star = -fba.fun


def fva(fraction):
    """Min and max of every flux, subject to growth >= fraction * f*."""
    # The extra constraint c'v >= fraction*f*, written as -c'v <= -fraction*f*
    A_ub, b_ub = -c[np.newaxis, :], [-fraction * f_star]
    vmin, vmax = np.zeros(n), np.zeros(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1
        vmin[j] = linprog(e, A_ub=A_ub, b_ub=b_ub, A_eq=S, b_eq=zeros,
                          bounds=bounds, method="highs").fun
        vmax[j] = -linprog(-e, A_ub=A_ub, b_ub=b_ub, A_eq=S, b_eq=zeros,
                           bounds=bounds, method="highs").fun
    return vmin, vmax


# --- Step 2: FVA at 100% and at 90% of the optimum
vmin100, vmax100 = fva(1.0)
vmin90, vmax90 = fva(0.9)

print(f"Optimal growth f* = {f_star:g}")
print(f"{'rxn':>5s} {'min':>8s} {'max':>8s}   (at 100% of f*)")
for r, lo, hi in zip(rxns, vmin100, vmax100):
    print(f"{r:>5s} {lo:8.1f} {hi:8.1f}")

# --- Plot: flux ranges at 100% and 90% of optimal growth
sbp.style()
fig, ax = plt.subplots()
y = np.arange(n)
ax.hlines(y + 0.15, vmin90, vmax90, color="#E69F00", lw=6,
          label="90% of optimum")
ax.hlines(y - 0.15, vmin100, vmax100, color="#0072B2", lw=6,
          label="100% of optimum")
# Fixed fluxes have zero width, so mark both ends
ax.plot(vmin100, y - 0.15, "|", color="#0072B2", ms=10)
ax.plot(vmin90, y + 0.15, "|", color="#E69F00", ms=10)
ax.set_yticks(y, rxns)
ax.invert_yaxis()
ax.set_xlabel("Flux (mmol/gDW/h)")
ax.set_title("Flux variability analysis")
ax.legend(loc="lower center", frameon=False)
sbp.save_figure(fig, __file__, "fig-toy-fva")

sbp.save_results(__file__, {
    "growth": f_star,
    "fva100": {"min": dict(zip(rxns, vmin100)),
               "max": dict(zip(rxns, vmax100))},
    "fva90": {"min": dict(zip(rxns, vmin90)),
              "max": dict(zip(rxns, vmax90))},
})
sbp.show()
