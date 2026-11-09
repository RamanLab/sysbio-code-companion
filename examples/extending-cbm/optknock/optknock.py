"""OptKnock: knock-outs that make E. coli over-produce lactate.

This follows Exercise 9.6 of the book. The E. coli core model grows
anaerobically on glucose (uptake up to 18 mmol/gDW/h), and we look for up
to three reaction knock-outs that maximise lactate secretion, while
keeping growth at no less than 10% of the wild type.

OptKnock (book §9.3.1, Eq. 9.6) is a bi-level problem: the outer problem
maximises lactate, over the choice of knock-outs, and the inner problem is
the cell maximising its growth. Replacing the inner problem by its
optimality conditions (primal feasibility, dual feasibility and strong
duality) turns it into a single MILP. As the book notes, OptKnock is slow
for large search spaces, so we let it choose from 14 reactions of central
carbon metabolism and fermentation. The search space is small enough that
we can also try every combination of knock-outs directly, and check the
two answers against each other.
"""
import itertools
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy import sparse
import matplotlib.pyplot as plt
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
model.reactions.EX_glc__D_e.lower_bound = -18     # glucose uptake
model.reactions.EX_o2_e.lower_bound = 0           # anaerobic
S = create_stoichiometric_matrix(model)
lb = np.array([r.lower_bound for r in model.reactions])
ub = np.array([r.upper_bound for r in model.reactions])
ids = [r.id for r in model.reactions]
m, n = S.shape
bio, lac = ids.index("Biomass_Ecoli_core"), ids.index("EX_lac__D_e")

candidates = ["ACALD", "ACKr", "ALCD2x", "FRD7", "FUM", "ME1", "ME2", "PDH",
              "PFL", "PPC", "PTAr", "PYK", "SUCCt3", "THD2"]
cand = [ids.index(r) for r in candidates]
nc = len(cand)


def optimise(j, lb, ub, sense=-1):
    """Maximise (sense=-1) or minimise (sense=1) flux j; None if infeasible."""
    c = np.zeros(n)
    c[j] = sense
    res = linprog(c, A_eq=S, b_eq=np.zeros(m), bounds=np.c_[lb, ub],
                  method="highs")
    return sense * res.fun if res.status == 0 else None


def knock_out(reactions):
    lb_k, ub_k = lb.copy(), ub.copy()
    for r in reactions:
        lb_k[ids.index(r)] = ub_k[ids.index(r)] = 0
    return lb_k, ub_k


def evaluate(reactions):
    """Growth of a design, and the least and most lactate it can secrete
    while growing at that rate."""
    lb_k, ub_k = knock_out(reactions)
    mu = optimise(bio, lb_k, ub_k)
    if mu is None or mu < 0.1 * mu_wt:
        return None
    lb_k[bio] = mu * (1 - 1e-9)
    return mu, optimise(lac, lb_k, ub_k, 1), optimise(lac, lb_k, ub_k)


def envelope(reactions, points=21):
    """Production envelope: least and most lactate at each growth rate."""
    lb_k, ub_k = knock_out(reactions)
    mus = np.linspace(0, optimise(bio, lb_k, ub_k), points)
    low, high = [], []
    for mu in mus:
        lb_k[bio] = ub_k[bio] = mu
        low.append(optimise(lac, lb_k, ub_k, 1))
        high.append(optimise(lac, lb_k, ub_k))
    return mus, np.array(low), np.array(high)


def solve_optknock(K, M=1000.0):
    """OptKnock as one MILP. Variables, in order: fluxes v (n), dual
    variables of S v = 0 (m), of the upper and lower bounds (n each),
    slacks that free the dual constraint of a knocked-out reaction (nc),
    and y (nc), with y = 0 for a knock-out."""
    iv, ilam, iu, il, isl, iy = 0, n, n + m, 2 * n + m, 3 * n + m, 3 * n + m + nc
    N = iy + nc
    rows, lo, hi = [], [], []

    def add(block, low, high):
        block = sparse.csr_matrix(block)
        rows.append(block)
        lo.extend(np.broadcast_to(low, block.shape[0]))
        hi.extend(np.broadcast_to(high, block.shape[0]))

    # Primal feasibility: S v = 0
    add(sparse.hstack([S, sparse.csr_matrix((m, N - n))]), 0, 0)
    # Dual feasibility: S'lam + mu_u - mu_l (+ slack) = c, c picks biomass
    E = sparse.csr_matrix((np.ones(nc), (cand, np.arange(nc))), shape=(n, nc))
    c = np.zeros(n)
    c[bio] = 1
    add(sparse.hstack([sparse.csr_matrix((n, n)), S.T, sparse.identity(n),
                       -sparse.identity(n), E, sparse.csr_matrix((n, nc))]),
        c, c)
    # Strong duality: growth = ub'mu_u - lb'mu_l
    r = np.zeros(N)
    r[bio], r[iu:iu + n], r[il:il + n] = 1, -ub, lb
    add(r[None, :], 0, 0)
    for k, j in enumerate(cand):
        def link(entries, low, high):
            r = np.zeros(N)
            for col, val in entries:
                r[col] = val
            add(r[None, :], low, high)
        link([(iv + j, 1), (iy + k, -ub[j])], -np.inf, 0)   # v_j <= ub_j y_k
        link([(iv + j, 1), (iy + k, -lb[j])], 0, np.inf)    # v_j >= lb_j y_k
        link([(iu + j, 1), (iy + k, -M)], -np.inf, 0)       # bound duals vanish
        link([(il + j, 1), (iy + k, -M)], -np.inf, 0)       #   when knocked out
        link([(isl + k, 1), (iy + k, M)], -np.inf, M)       # |slack| is 0
        link([(isl + k, -1), (iy + k, M)], -np.inf, M)      #   unless knocked out
    count = np.zeros(N)
    count[iy:] = 1                                          # at most K knock-outs
    constraints = [LinearConstraint(sparse.vstack(rows), lo, hi),
                   LinearConstraint(count[None, :], nc - K, np.inf)]
    low = np.r_[lb, np.full(m, -np.inf), np.zeros(2 * n), np.full(nc, -np.inf),
                np.zeros(nc)]
    high = np.r_[ub, np.full(m + 2 * n + nc, np.inf), np.ones(nc)]
    low[bio] = 0.1 * mu_wt
    objective = np.zeros(N)
    objective[lac] = -1                                     # maximise lactate
    integrality = np.r_[np.zeros(iy), np.ones(nc)]
    res = milp(objective, constraints=constraints, bounds=Bounds(low, high),
               integrality=integrality, options={"mip_rel_gap": 0})
    assert res.status == 0
    design = [candidates[k] for k in range(nc) if round(res.x[iy + k]) == 0]
    return -res.fun, design


mu_wt = optimise(bio, lb, ub)
_, lac_lo, lac_hi = evaluate([])
print(f"Wild type: growth {mu_wt:.4f} /h, lactate {lac_lo:.2f} to {lac_hi:.2f}")

# Every design of up to three knock-outs from the search space
designs = []
for K in (1, 2, 3):
    for ko in itertools.combinations(sorted(candidates), K):
        e = evaluate(ko)
        if e is not None:
            designs.append((K, list(ko)) + e)


def best(K, column):
    """The design with the most lactate (column 4: at best, 3: guaranteed),
    with ties broken alphabetically."""
    pool = [d for d in designs if d[0] <= K]
    top = max(d[column] for d in pool)
    return [d for d in pool if d[column] > top - 1e-6][0]


results = {"growth_wild_type": mu_wt}
for K in (1, 2, 3):
    value, design = solve_optknock(K)
    _, _, mu, low, high = best(K, 4)
    assert abs(value - high) < 1e-5            # OptKnock finds the true optimum
    g = best(K, 3)
    print(f"K = {K}: OptKnock picks {', '.join(design)}: up to {value:.2f} lactate")
    print(f"       such a design grows at {mu:.3f} /h, and secretes "
          f"{low:.2f} to {high:.2f} lactate")
    print(f"       most lactate guaranteed: {', '.join(g[1])}: growth "
          f"{g[2]:.3f} /h, lactate {g[3]:.2f} to {g[4]:.2f}")
    results[f"K{K}"] = {"optknock_lactate": value, "growth": mu,
                        "lactate_min": low, "lactate_max": high,
                        "guaranteed_design": g[1], "guaranteed_growth": g[2],
                        "guaranteed_lactate": g[3]}

# --- Plot: production envelopes
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
panels = [("Three knock-outs",
           [(", ".join(best(3, 4)[1]) + " (OptKnock)", best(3, 4)[1], "#d62728"),
            (", ".join(best(3, 3)[1]), best(3, 3)[1], "#1f77b4")]),
          ("One knock-out",
           [(", ".join(best(1, 3)[1]), best(1, 3)[1], "#2ca02c")])]
for ax, (title, shown) in zip(axes, panels):
    for label, ko, colour in [("wild type", [], "#999999")] + shown:
        mus, low, high = envelope(ko)
        ax.fill_between(mus, low, high, color=colour, alpha=0.2, lw=0)
        ax.plot(np.r_[mus, mus[::-1], mus[:1]], np.r_[high, low[::-1], high[:1]],
                color=colour, label=label)
    ax.axvline(0.1 * mu_wt, ls=":", color="black", lw=1)
    ax.set_xlabel("Growth rate (1/h)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
axes[0].set_ylabel("Lactate secretion (mmol/gDW/h)")
sbp.save_figure(fig, __file__, "fig-optknock")

sbp.save_results(__file__, results)
sbp.show()
