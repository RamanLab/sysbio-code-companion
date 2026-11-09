"""Regulatory on-off minimisation (ROOM) for single deletions.

ROOM (book §9.2, Eq. 9.5) assumes that a mutant changes as few fluxes as
possible, compared with the wild type w. A change is significant if the
flux leaves the range [w_l, w_u], with w_l = w - delta|w| - epsilon and
w_u = w + delta|w| + epsilon. A binary variable y_i is 1 if flux i changes
significantly, and ROOM minimises sum(y): a mixed-integer linear
programme (MILP). Many flux distributions reach that minimum, so we then
report the fastest growth among them, which is unique. The wild type w is
the flux distribution of smallest norm at optimal growth, as for MoMA.
Binary variables make ROOM much slower than FBA or MoMA, so we look at
five deletions in glycolysis and the pentose phosphate pathway, including pyruvate kinase
(PYK), the example of the book. To keep the MILP well conditioned, the
bounds lb and ub in Eq. 9.5 are the tightest the network allows, found
by flux variability analysis (FVA), rather than the nominal +/-1000.
"""
import numpy as np
import clarabel
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy import sparse
import matplotlib.pyplot as plt
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
S = create_stoichiometric_matrix(model)
lb = np.array([r.lower_bound for r in model.reactions])
ub = np.array([r.upper_bound for r in model.reactions])
ids = [r.id for r in model.reactions]
bio = ids.index("Biomass_Ecoli_core")
m, n = S.shape
delta, epsilon = 0.03, 0.001          # as Shlomi et al. used for fluxes


def fba(lb, ub):
    """Maximum growth rate, or None if the constraints cannot be met."""
    c = np.zeros(n)
    c[bio] = -1
    res = linprog(c, A_eq=S, b_eq=np.zeros(m), bounds=np.c_[lb, ub],
                  method="highs")
    return max(0.0, -res.fun) if res.status == 0 else None


def min_norm(lb, ub):
    """The flux vector of smallest Euclidean norm (QP, as in MoMA)."""
    eye = sparse.identity(n, format="csc")
    A = sparse.vstack([sparse.csc_matrix(S), eye, -eye], format="csc")
    cones = [clarabel.ZeroConeT(m), clarabel.NonnegativeConeT(2 * n)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    sol = clarabel.DefaultSolver(eye, np.zeros(n), A, np.r_[np.zeros(m), ub, -lb],
                                 cones, settings).solve()
    return np.array(sol.x)


def room(w, lb, ub):
    """Fewest significant changes from w, and the fastest growth that
    keeps to that number. Variables are [v, y], y binary."""
    wl = w - delta * np.abs(w) - epsilon
    wu = w + delta * np.abs(w) + epsilon
    eye = sparse.identity(n, format="csr")
    # S v = 0;  v - y (lb - wl) >= wl;  v - y (ub - wu) <= wu
    cons = [LinearConstraint(sparse.hstack([S, sparse.csr_matrix((m, n))]), 0, 0),
            LinearConstraint(sparse.hstack([eye, -sparse.diags(lb - wl)]), wl, np.inf),
            LinearConstraint(sparse.hstack([eye, -sparse.diags(ub - wu)]), -np.inf, wu)]
    bounds = Bounds(np.r_[lb, np.zeros(n)], np.r_[ub, np.ones(n)])
    integrality = np.r_[np.zeros(n), np.ones(n)]
    opts = {"mip_rel_gap": 0}
    res = milp(np.r_[np.zeros(n), np.ones(n)], constraints=cons, bounds=bounds,
               integrality=integrality, options=opts)
    assert res.status == 0
    changes = round(res.fun)
    # Second step: fastest growth with at most that many changes
    c = np.zeros(2 * n)
    c[bio] = -1
    cons.append(LinearConstraint(np.r_[np.zeros(n), np.ones(n)][None, :],
                                 -np.inf, changes))
    res = milp(c, constraints=cons, bounds=bounds, integrality=integrality,
               options=opts)
    v = res.x[:n]
    tol = 1e-6                       # check: count the changes from v itself
    assert np.sum((v < wl - tol) | (v > wu + tol)) <= changes
    return changes, -res.fun


# Wild type: optimal growth, smallest norm
mu_wt = fba(lb, ub)
lb_wt, ub_wt = lb.copy(), ub.copy()
lb_wt[bio], ub_wt[bio] = mu_wt * (1 - 1e-9), mu_wt
w = min_norm(lb_wt, ub_wt)
w[np.abs(w) < 1e-9] = 0
print(f"Wild-type growth {mu_wt:.4f} /h; {np.sum(w != 0)} reactions carry flux")

# Tightest bounds the network allows (FVA, at any growth rate)
lo, hi = np.zeros(n), np.zeros(n)
for j in range(n):
    c = np.zeros(n)
    c[j] = 1
    lo[j] = linprog(c, A_eq=S, b_eq=np.zeros(m), bounds=np.c_[lb, ub],
                    method="highs").fun
    hi[j] = -linprog(-c, A_eq=S, b_eq=np.zeros(m), bounds=np.c_[lb, ub],
                     method="highs").fun

rxns = ["PGI", "PFK", "TPI", "G6PDH2r", "PYK"]
growth_fba, growth_room, changes = [], [], []
for r in rxns:
    j = ids.index(r)
    lb_d, ub_d = lo.copy(), hi.copy()
    lb_d[j] = ub_d[j] = 0
    k, mu_room = room(w, lb_d, ub_d)
    growth_fba.append(fba(lb_d, ub_d) / mu_wt)
    growth_room.append(mu_room / mu_wt)
    changes.append(k)
    print(f"  {r:8s} {k:2d} significant changes; growth FBA "
          f"{growth_fba[-1]:.3f}, ROOM {growth_room[-1]:.3f} of wild type")
growth_fba, growth_room = np.array(growth_fba), np.array(growth_room)
changes = np.array(changes)

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(len(rxns))
ax1.bar(x - 0.2, growth_fba, 0.4, color="#ff7f0e", label="FBA")
ax1.bar(x + 0.2, growth_room, 0.4, color="#1f77b4", label="ROOM")
ax1.set_xticks(x, rxns, rotation=45)
ax1.set_ylabel("Growth (fraction of wild type)")
ax1.set_title("Growth after each deletion")
ax1.legend(frameon=False, loc="lower left")
ax2.bar(x, changes, color="#1f77b4")
ax2.set_xticks(x, rxns, rotation=45)
ax2.set_ylabel("Significant flux changes")
ax2.set_title("Fewest flux changes (ROOM)")
sbp.save_figure(fig, __file__, "fig-room")

sbp.save_results(__file__, {
    "growth_wild_type": mu_wt, "reactions": rxns,
    "growth_fba": growth_fba, "growth_room": growth_room,
    "changes": changes,
})
sbp.show()
