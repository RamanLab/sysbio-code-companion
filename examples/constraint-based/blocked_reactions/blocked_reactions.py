"""Blocked reactions and dead-end metabolites.

1. The small network of book Figure 8.3 (Eq. 8.13): find its blocked
   reactions from the reduced row echelon form of S (Eq. 8.14), and
   confirm them by FVA.
2. The E. coli core model: find dead-end metabolites from the
   stoichiometry alone, and the reactions that cannot carry flux, first
   with every exchange reaction open, then on glucose.
"""
import numpy as np
import sympy
from scipy.optimize import linprog
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp


def fva(S, lb, ub):
    """Smallest and largest value of every flux, with S v = 0 and the
    bounds (no objective constraint), or None if infeasible."""
    n = S.shape[1]
    vmin, vmax = np.zeros(n), np.zeros(n)
    for j in range(n):
        c = np.zeros(n)
        c[j] = 1
        lo = linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]),
                     bounds=np.c_[lb, ub], method="highs")
        hi = linprog(-c, A_eq=S, b_eq=np.zeros(S.shape[0]),
                     bounds=np.c_[lb, ub], method="highs")
        vmin[j], vmax[j] = lo.fun, -hi.fun
    return vmin, vmax


# --- 1. The network of book Figure 8.3
names = ["v1", "v2", "v3", "b1", "b2", "b3"]
#              v1  v2  v3  b1  b2  b3
S = np.array([[-1,  0, -1,  1,  0,  0],      # A
              [-1,  1, -1,  0,  0,  0],      # B
              [ 1, -1,  0,  0,  0,  0],      # C
              [ 0,  1,  0,  0, -1,  0],      # D
              [ 0,  0,  1,  0,  0, -1]])     # E
R, _ = sympy.Matrix(S).rref()
R = np.array(R.tolist(), dtype=float)
print("Reduced row echelon form of S (book Eq. 8.14):")
print(R.astype(int))
# A row with a single non-zero entry says that flux must be zero
by_rref = [names[np.flatnonzero(r)[0]] for r in R
           if np.count_nonzero(r) == 1]
vmin, vmax = fva(S, np.zeros(6), np.full(6, 1000.0))
by_fva = [n for n, lo, hi in zip(names, vmin, vmax)
          if abs(lo) < 1e-9 and abs(hi) < 1e-9]
print(f"Blocked, from the RREF: {by_rref}; from FVA: {by_fva}")

# --- 2. The E. coli core model
model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
S = create_stoichiometric_matrix(model)
lb = np.array([r.lower_bound for r in model.reactions])
ub = np.array([r.upper_bound for r in model.reactions])
rxns = [r.id for r in model.reactions]
mets = [m.id for m in model.metabolites]
exchange = np.array([r.startswith("EX_") for r in rxns])



def dead_end_metabolites(lb, ub):
    """Metabolites that, given the directions the reactions can run, can
    only ever be produced or only ever be consumed."""
    can_make = ((S > 0) & (ub > 0)) | ((S < 0) & (lb < 0))
    can_use = ((S < 0) & (ub > 0)) | ((S > 0) & (lb < 0))
    return [m for m, mk, us in zip(mets, can_make.any(1), can_use.any(1))
            if not (mk and us)]


# Every exchange reaction open. The ATP maintenance demand (a lower bound
# > 0) is relaxed to 0 here, since it asks for flux rather than allowing it.
lb_open = np.where(exchange, -1000.0, np.minimum(lb, 0.0))
dead_ends = dead_end_metabolites(lb_open, ub)
dead_ends_glucose = dead_end_metabolites(lb, ub)
print(f"Dead-end metabolites with every exchange open: {len(dead_ends)}; "
      f"on glucose: {dead_ends_glucose}")
vmin, vmax = fva(S, lb_open, ub)
blocked = [r for r, lo, hi in zip(rxns, vmin, vmax)
           if abs(lo) < 1e-9 and abs(hi) < 1e-9]
print(f"Reactions blocked in every condition: {len(blocked)}")

# No flux on glucose (the model's default medium), at any growth rate
vmin, vmax = fva(S, lb, ub)
no_flux = [r for r, lo, hi in zip(rxns, vmin, vmax)
           if abs(lo) < 1e-9 and abs(hi) < 1e-9]
print(f"Reactions that cannot carry flux on glucose: {len(no_flux)}")
print("  " + ", ".join(no_flux))

sbp.save_results(__file__, {
    "rref": R, "blocked_rref": by_rref, "blocked_fva": by_fva,
    "dead_ends": dead_ends, "dead_ends_glucose": dead_ends_glucose,
    "blocked_core": blocked,
    "no_flux_glucose": no_flux,
})
