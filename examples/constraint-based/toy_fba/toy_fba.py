"""Flux balance analysis of a toy metabolic network.

The network is the one in Fig. 8.1 / Table 8.2 of the book: 8 metabolites,
7 internal reactions, 2 exchange reactions and a 'biomass' reaction.
FBA is solved directly as a linear program with SciPy.
"""
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import sbp

# --- The network
rxns = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "b1", "b2", "vbio"]
mets = ["A", "B", "C", "D", "E", "F", "G", "ATP"]

# Stoichiometric matrix: one row per metabolite, one column per reaction
#             v1  v2  v3  v4  v5  v6  v7  b1  b2 vbio
S = np.array([[-1, -1,  0,  0,  0,  0,  0,  1,  0,   0],   # A
              [ 1,  0,  1, -1, -1,  0,  0,  0,  0,   0],   # B
              [ 0,  1, -1,  0,  0,  0,  0,  0,  0,   0],   # C
              [ 0,  0,  0,  2,  0, -1, -1,  0,  0,   0],   # D
              [ 0,  0,  0,  0,  0,  1,  0,  0, -1,   0],   # E
              [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  -1],   # F
              [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  -2],   # G
              [ 1,  0,  1,  0,  0,  0, -1,  0,  0, -10]])  # ATP

# Bounds (mmol/gDW/h): reversible reactions v4, v6 and b2 may run backwards
lb = np.array([0, 0, 0, -1000, 0, -1000, 0, 0, -1000, 0])
ub = np.full(len(rxns), 1000)

# Objective: maximise the biomass flux vbio
c = np.zeros(len(rxns))
c[rxns.index("vbio")] = 1

# --- Solve: max c'v  s.t.  Sv = 0,  lb <= v <= ub
# linprog minimises, so we minimise -c'v
res = linprog(-c, A_eq=S, b_eq=np.zeros(len(mets)),
              bounds=list(zip(lb, ub)), method="highs")
v = res.x + 0.0  # (+ 0.0 turns any -0.0 into 0.0 for printing)
growth = -res.fun

print(f"Status: {res.message}")
print(f"Maximum biomass flux vbio = {growth:g}")
for r, flux in zip(rxns, v):
    print(f"  {r:>4s} = {flux:8.1f}")
print(f"Largest mass-balance violation |Sv| = {np.abs(S @ v).max():.1e}")

# --- Plot the flux distribution
sbp.style()
fig, ax = plt.subplots()
ax.bar(rxns, v, color="#0072B2")
ax.set_xlabel("Reaction")
ax.set_ylabel("Flux (mmol/gDW/h)")
ax.set_title(f"An optimal flux distribution (vbio = {growth:g})")
sbp.save_figure(fig, __file__, "fig-toy-fba")

sbp.save_results(__file__, {
    "growth": growth,
    "fluxes": dict(zip(rxns, v)),
    "max_mass_balance_error": np.abs(S @ v).max(),
})
sbp.show()
