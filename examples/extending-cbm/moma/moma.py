"""Predicting the fluxes of a mutant: FBA and MoMA.

FBA assumes that a mutant, like the wild type, grows as fast as its
network allows. MoMA (minimisation of metabolic adjustment) assumes that
a freshly made mutant stays as close as possible to the wild-type flux
distribution w: it minimises the distance ||v - w||^2, subject to the
same constraints S v = 0 and lb <= v <= ub, with the deleted reaction
fixed at zero. That is a quadratic programme (QP). We write both
problems out with the stoichiometric matrix, solve the linear programmes
with SciPy and the QPs with Clarabel, and compare FBA and MoMA for every
deletion of an internal reaction of the E. coli core model.
"""
import numpy as np
import clarabel
from scipy.optimize import linprog
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
n = len(ids)


def fba(lb, ub):
    """Maximum growth rate, or None if the constraints cannot be met."""
    c = np.zeros(n)
    c[bio] = -1                          # linprog minimises
    res = linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]),
                  bounds=np.c_[lb, ub], method="highs")
    return max(0.0, -res.fun) if res.status == 0 else None


def nearest(target, lb, ub):
    """The flux vector closest to target: minimise ||v - target||^2 / 2,
    that is v'v/2 - target'v, subject to S v = 0 and lb <= v <= ub.
    Clarabel takes constraints as A v + s = b, with s = 0 for equalities
    and s >= 0 for inequalities: here v <= ub and -v <= -lb."""
    m = S.shape[0]
    eye = sparse.identity(n, format="csc")
    A = sparse.vstack([sparse.csc_matrix(S), eye, -eye], format="csc")
    b = np.r_[np.zeros(m), ub, -lb]
    cones = [clarabel.ZeroConeT(m), clarabel.NonnegativeConeT(2 * n)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    solution = clarabel.DefaultSolver(eye, -np.asarray(target, dtype=float),
                                      A, b, cones, settings).solve()
    assert str(solution.status) == "Solved"
    return np.array(solution.x)


def at_growth(mu, lb, ub):
    """Bounds that hold growth at (just below) mu."""
    lb, ub = lb.copy(), ub.copy()
    lb[bio], ub[bio] = mu * (1 - 1e-9), mu
    return lb, ub


# Wild type. Many flux distributions achieve the maximum growth rate;
# as reference we take the one with the smallest Euclidean norm, which
# is unique.
mu_wt = fba(lb, ub)
w = nearest(np.zeros(n), *at_growth(mu_wt, lb, ub))
print(f"Wild-type growth rate {mu_wt:.4f} /h")

# Every deletion of an internal reaction
rxns = sorted(r for r in ids if not r.startswith(("EX_", "Biomass")))
growth_fba, growth_moma = [], []
for r in rxns:
    lb_d, ub_d = lb.copy(), ub.copy()
    lb_d[ids.index(r)] = ub_d[ids.index(r)] = 0
    mu = fba(lb_d, ub_d)
    if mu is None:            # no flux distribution meets the constraints,
        growth_fba.append(0.0)    # e.g. the ATP maintenance demand
        growth_moma.append(0.0)
        continue
    growth_fba.append(mu / mu_wt)
    growth_moma.append(nearest(w, lb_d, ub_d)[bio] / mu_wt)
growth_fba, growth_moma = np.array(growth_fba), np.array(growth_moma)
lower = growth_moma < growth_fba - 0.01
print(f"{len(rxns)} deletions; MoMA predicts slower growth than FBA for "
      f"{lower.sum()} of them")
for r in ["PGI", "PFK", "TPI", "G6PDH2r", "PYK", "CS"]:
    i = rxns.index(r)
    print(f"  {r:8s} FBA {growth_fba[i]:.3f}, MoMA {growth_moma[i]:.3f} "
          f"of wild-type growth")

# Fluxes after deleting phosphofructokinase (PFK)
show = ["PGI", "PFK", "FBA", "G6PDH2r", "GND", "TKT1", "TALA", "GAPD", "PYK"]
lb_d, ub_d = lb.copy(), ub.copy()
lb_d[ids.index("PFK")] = ub_d[ids.index("PFK")] = 0
v_fba = nearest(np.zeros(n), *at_growth(fba(lb_d, ub_d), lb_d, ub_d))
v_moma = nearest(w, lb_d, ub_d)
pfk = {name: [v[ids.index(r)] for r in show]
       for name, v in [("wild_type", w), ("fba", v_fba), ("moma", v_moma)]}

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2),
                               gridspec_kw={"width_ratios": [2, 3]})
ax1.plot([0, 1.05], [0, 1.05], ":", color="#999999")
ax1.plot(growth_fba, growth_moma, "o", color="#1f77b4", alpha=0.7)
ax1.set_xlabel("FBA (fraction of wild type)")
ax1.set_ylabel("MoMA (fraction of wild type)")
ax1.set_title("Growth after 74 deletions")
x = np.arange(len(show))
for k, (name, colour) in enumerate([("wild_type", "#999999"),
                                    ("fba", "#ff7f0e"),
                                    ("moma", "#1f77b4")]):
    ax2.bar(x + (k - 1) * 0.27, pfk[name], 0.27, color=colour,
            label={"wild_type": "wild type", "fba": "FBA",
                   "moma": "MoMA"}[name])
ax2.set_xticks(x, show, rotation=45)
ax2.axhline(0, color="black", lw=0.5)
ax2.set_ylabel("Flux (mmol/gDW/h)")
ax2.set_title("Fluxes after deleting PFK")
ax2.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-moma")

sbp.save_results(__file__, {
    "growth_wild_type": mu_wt, "reactions": rxns,
    "growth_fba": growth_fba, "growth_moma": growth_moma,
    "pfk_reactions": show, "pfk_fluxes": pfk,
})
sbp.show()
