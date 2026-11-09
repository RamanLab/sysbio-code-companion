"""Loops, parsimonious FBA (pFBA), and the pFBA classes of reactions.

1. The network of book Figure 8.4, which contains a stoichiometrically
   balanced cycle: FVA shows the cycle can carry any flux, and minimising
   the total flux, sum_j |v_j|, removes it.
2. The E. coli core model on glucose: classify every internal reaction,
   as in book §8.8.5, as essential, pFBA optimal, enzymatically less
   efficient (ELE), metabolically less efficient (MLE), no flux in this
   medium, or blocked.
Total flux is minimised as a linear programme, with variables t_j >= |v_j|.
"""
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp


def lp(c, S, lb, ub, A_ub=None, b_ub=None):
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=S, b_eq=np.zeros(S.shape[0]),
                  bounds=np.c_[lb, ub], method="highs")
    return res.fun if res.status == 0 else None


def with_abs(S, lb, ub):
    """Add variables t_j >= |v_j|: returns S, bounds and the inequality
    rows (v - t <= 0, -v - t <= 0) for the variables [v, t]."""
    m, n = S.shape
    S2 = np.c_[S, np.zeros((m, n))]
    lb2, ub2 = np.r_[lb, np.zeros(n)], np.r_[ub, np.full(n, 1000.0)]
    eye = np.eye(n)
    A = np.r_[np.c_[eye, -eye], np.c_[-eye, -eye]]
    return S2, lb2, ub2, A, np.zeros(2 * n)


def fva(S, lb, ub, A_ub=None, b_ub=None, which=None):
    """Range of each of the first variables listed in which."""
    out = []
    for j in which:
        c = np.zeros(S.shape[1])
        c[j] = 1
        out.append((lp(c, S, lb, ub, A_ub, b_ub),
                    -lp(-c, S, lb, ub, A_ub, b_ub)))
    return np.array(out)


# --- 1. Book Figure 8.4: A -> B -> C, and C -> D -> A closes a cycle
names = ["b1", "v1", "v2", "b2", "v3", "v4"]
#              b1  v1  v2  b2  v3  v4
S = np.array([[ 1, -1,  0,  0,  0,  1],      # A
              [ 0,  1, -1,  0,  0,  0],      # B
              [ 0,  0,  1, -1, -1,  0],      # C
              [ 0,  0,  0,  0,  1, -1]])     # D
lb, ub = np.zeros(6), np.array([10, 1000, 1000, 1000, 1000, 1000.0])
c = np.zeros(6)
c[3] = -1                                    # maximise b2
best = -lp(c, S, lb, ub)
lb_opt = lb.copy()
lb_opt[3] = best                             # hold b2 at its maximum
toy_fva = fva(S, lb_opt, ub, which=range(6))
S2, lb2, ub2, A, b = with_abs(S, lb_opt, ub)
c2 = np.r_[np.zeros(6), np.ones(6)]
toy_min_total = lp(c2, S2, lb2, ub2, A, b)
# Every flux is fixed once the total is minimal: check with FVA
toy_pfba = fva(S2, lb2, ub2, np.r_[A, c2[None, :]],
               np.r_[b, toy_min_total * (1 + 1e-9)], which=range(6))
print(f"Figure 8.4: max b2 = {best:g}")
for nm, r, p in zip(names, toy_fva, toy_pfba):
    print(f"  {nm}: FVA [{r[0]:g}, {r[1]:g}]; with minimal total flux "
          f"[{p[0]:.4g}, {p[1]:.4g}]")

# --- 2. pFBA classes for the E. coli core model
model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
S = create_stoichiometric_matrix(model)
lb = np.array([r.lower_bound for r in model.reactions])
ub = np.array([r.upper_bound for r in model.reactions])
rxns = [r.id for r in model.reactions]
n = len(rxns)
bio = rxns.index("Biomass_Ecoli_core")
internal = [j for j, r in enumerate(rxns)
            if not r.startswith(("EX_", "Biomass"))]
c = np.zeros(n)
c[bio] = -1
mu = -lp(c, S, lb, ub)


def zero(r):
    return abs(r[0]) < 1e-6 and abs(r[1]) < 1e-6


# Ranges: on glucose at any growth; at optimal growth; at optimal growth
# with minimal total flux
any_growth = fva(S, lb, ub, which=internal)
lb_mu = lb.copy()
lb_mu[bio] = mu * (1 - 1e-9)
at_optimum = fva(S, lb_mu, ub, which=internal)
S2, lb2, ub2, A, b = with_abs(S, lb_mu, ub)
c2 = np.r_[np.zeros(n), np.ones(n)]
min_total = lp(c2, S2, lb2, ub2, A, b)
parsimonious = fva(S2, lb2, ub2, np.r_[A, c2[None, :]],
                   np.r_[b, min_total * (1 + 1e-9)], which=internal)
# Essential: growth below 1% of wild type without the reaction
essential = []
for j in internal:
    lb_d, ub_d = lb.copy(), ub.copy()
    lb_d[j] = ub_d[j] = 0
    g = lp(c, S, lb_d, ub_d)
    essential.append(g is None or -g < 0.01 * mu)

classes = {k: [] for k in ["essential", "pFBA optima", "ELE", "MLE",
                           "no flux", "blocked"]}
for k, j in enumerate(internal):
    if essential[k]:
        cls = "essential"
    elif not zero(parsimonious[k]):
        cls = "pFBA optima"
    elif not zero(at_optimum[k]):
        cls = "ELE"
    elif not zero(any_growth[k]):
        cls = "MLE"
    else:
        cls = "no flux"       # the core model has no blocked reactions
    classes[cls].append(rxns[j])
print(f"\nE. coli core on glucose: growth {mu:.4f} /h; minimal total flux "
      f"{min_total:.2f} mmol/gDW/h")
for k, v in classes.items():
    print(f"  {k:12s} {len(v):3d}  {', '.join(v)}")

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(7, 3.5))
labels = list(classes)
ax.barh(labels[::-1], [len(classes[k]) for k in labels[::-1]],
        color="#1f77b4")
for y, k in enumerate(labels[::-1]):
    ax.text(len(classes[k]) + 0.3, y, str(len(classes[k])), va="center")
ax.set_xlabel("Number of internal reactions")
ax.set_title("pFBA classes, E. coli core on glucose")
sbp.save_figure(fig, __file__, "fig-pfba")

sbp.save_results(__file__, {
    "toy_max_b2": best, "toy_fva": toy_fva, "toy_pfba": toy_pfba,
    "toy_min_total": toy_min_total, "growth": mu,
    "min_total_flux": min_total,
    "classes": {k.replace(" ", "_"): v for k, v in classes.items()},
})
sbp.show()
