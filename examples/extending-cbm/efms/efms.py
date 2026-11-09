"""Elementary flux modes (EFMs) and extreme pathways of a small network.

The network is that of Figure 9.2 of the book. A and C can be exchanged
with the surroundings (v1, v4r), as can F and G (v10, v11); v3r and v4r
are reversible, and every other reaction is irreversible.

An EFM is a steady-state flux distribution that uses a minimal set of
reactions (book §9.8): no smaller set of reactions can carry a steady
flux. For a network this small, we can apply the definition directly.
For every set of reactions, we ask whether the stoichiometric matrix,
restricted to those reactions, has exactly one steady state (up to
scaling) that uses all of them, with every irreversible reaction running
forwards. Real networks need cleverer algorithms, such as the double
description method of efmtool, because the number of sets doubles with
every reaction.
"""
from itertools import product
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import sbp

reactions = ["v1", "v2", "v3r", "v4r", "v5", "v6", "v7", "v8", "v9", "v10", "v11"]
metabolites = ["A", "B", "C", "D", "E", "F", "G"]
reversible = np.array([r.endswith("r") for r in reactions])
S = np.array([
    # v1 v2 v3r v4r v5 v6 v7 v8 v9 v10 v11
    [1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # A
    [0,  1, -1,  0, -1,  0,  0,  0,  0,  0,  0],   # B
    [0,  0,  1, -1,  0,  0,  0,  0, -1,  0,  0],   # C
    [0,  0,  0,  0,  1, -1,  0, -1,  0,  0,  0],   # D
    [0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0],   # E
    [0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0],   # F
    [0,  0,  0,  0,  0,  0,  1,  2,  1,  0, -1],   # G
], dtype=float)
n = len(reactions)

efms = []
for mask in product([False, True], repeat=n):
    support = np.array(mask)
    if not support.any():
        continue
    K = null_space(S[:, support])
    if K.shape[1] != 1:              # no steady state, or not a minimal one
        continue
    v = np.zeros(n)
    v[support] = K[:, 0]
    if np.any(np.abs(v[support]) < 1e-9):
        continue                     # some reaction in the set is unused
    for sign in (1, -1):             # a mode and its reverse
        w = sign * v
        if np.all(w[~reversible] >= -1e-9):
            efms.append(w / np.min(np.abs(w[support])))
efms = np.round(np.array(efms), 9) + 0.0          # + 0.0 turns -0 into 0

# Systemic independence: is a mode a non-negative combination of the others?
extreme = []
for i, e in enumerate(efms):
    others = np.delete(efms, i, axis=0).T
    res = linprog(np.zeros(others.shape[1]), A_eq=others, b_eq=e,
                  bounds=(0, None), method="highs")
    extreme.append(res.status != 0)  # infeasible: not a combination
extreme = np.array(extreme)

# Number the modes as the book does: extreme pathways first, the mode
# that only converts A to C first in each group, then by the last
# reaction used before v11 (v10, then v8, then v7)
keys = [-np.abs(efms[:, j]) for j in range(4, 10)] + [efms[:, 10] != 0, ~extreme]
order = np.lexsort(keys)          # the last key sorts first
efms, extreme = efms[order], extreme[order]

# Yield of G on C, for the modes that take up C (v4r < 0)
print(f"{len(efms)} elementary flux modes, {extreme.sum()} of them extreme "
      "pathways (marked *):")
print("      " + " ".join(f"{r:>4s}" for r in reactions) + "   G/C")
yields = []
for k, (e, x) in enumerate(zip(efms, extreme), start=1):
    y = e[10] / -e[3] if e[3] < 0 else None
    yields.append(y)
    print(f"{k:2d}{'*' if x else ' '}   " + " ".join(f"{z:4.0f}" for z in e)
          + (f"  {y:4.1f}" if y is not None else ""))

# --- Plot: the modes as a matrix
sbp.style()
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.imshow(efms, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
for (i, j), z in np.ndenumerate(efms):
    if z != 0:
        ax.text(j, i, f"{z:.0f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(n), reactions)
ax.set_yticks(range(len(efms)),
              [f"EFM {k + 1}{'*' if x else ''}" for k, x in enumerate(extreme)])
ax.set_title("Elementary flux modes (* extreme pathways)")
sbp.save_figure(fig, __file__, "fig-efms")

sbp.save_results(__file__, {
    "reactions": reactions, "efms": efms, "extreme": extreme,
    "yield_G_on_C": yields,
})
sbp.show()
