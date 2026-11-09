"""Two E. coli mutants that can only grow together.

Strain A cannot make glutamate (GLUDy and GLUSy deleted); strain B
cannot replenish its TCA cycle (PPC and ICL deleted). Neither grows
alone on glucose. In a community model, each strain keeps its own
network, and both exchange metabolites through a shared medium. With
the fraction x of strain A fixed, both strains growing at the same rate
mu (as they must in a stable community), and fluxes per gram of the
whole community, the problem is a linear programme:
    maximise mu
    subject to  S u_A = 0,  S u_B = 0,
                u_A[biomass] = x mu,  u_B[biomass] = (1 - x) mu,
                x lb_A <= u_A <= x ub_A, (1 - x) lb_B <= u_B <= (1 - x) ub_B,
                what the community takes from the medium is available.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
S = create_stoichiometric_matrix(model)
lb0 = np.array([r.lower_bound for r in model.reactions])
ub0 = np.array([r.upper_bound for r in model.reactions])
ids = [r.id for r in model.reactions]
n, m = len(ids), S.shape[0]
bio = ids.index("Biomass_Ecoli_core")
ex = [i for i, r in enumerate(ids) if r.startswith("EX_")]

# The medium supplies glucose, oxygen, ammonium, phosphate, water and
# protons, and takes up CO2; other metabolites come only from the cells
medium = ["EX_glc__D_e", "EX_o2_e", "EX_nh4_e", "EX_pi_e", "EX_h2o_e",
          "EX_h_e", "EX_co2_e"]


def strain(deleted):
    """Bounds per gram of a strain. Each cell can take up or secrete at
    most 10 mmol/gDW/h of each metabolite other than the inorganic ones,
    and glucose uptake is limited to 10 as before."""
    lb, ub = lb0.copy(), ub0.copy()
    for i in ex:
        if ids[i] in medium[1:]:
            lb[i] = -1000.0
        else:
            lb[i], ub[i] = -10.0, 10.0
    for r in deleted:
        lb[ids.index(r)] = ub[ids.index(r)] = 0
    return lb, ub


A = strain(["GLUDy", "GLUSy"])
B = strain(["PPC", "ICL"])


def community(x, minimise_exchange=False, mu_fixed=None):
    """Variables: u_A (n), u_B (n), mu, and, to minimise cross-feeding,
    t_A and t_B >= |exchange fluxes| (one per exchange reaction)."""
    k = len(ex) if minimise_exchange else 0
    N = 2 * n + 1 + 2 * k
    Aeq = np.zeros((2 * m + 2, N))
    Aeq[:2 * m, :2 * n] = block_diag(S, S)
    Aeq[2 * m, bio], Aeq[2 * m, 2 * n] = 1, -x
    Aeq[2 * m + 1, n + bio], Aeq[2 * m + 1, 2 * n] = 1, -(1 - x)
    beq = np.zeros(2 * m + 2)
    # Shared medium: net uptake by the community of a metabolite the
    # medium does not supply must be zero or negative
    rows = []
    for i in ex:
        if ids[i] not in medium:
            r = np.zeros(N)
            r[i] = r[n + i] = -1
            rows.append(r)
    Aub, bub = np.array(rows), np.zeros(len(rows))
    bounds = np.r_[np.c_[x * A[0], x * A[1]],
                   np.c_[(1 - x) * B[0], (1 - x) * B[1]],
                   [[0, np.inf]], np.tile([0, np.inf], (2 * k, 1))]
    c = np.zeros(N)
    if not minimise_exchange:
        c[2 * n] = -1                                   # maximise mu
    else:
        bounds[2 * n] = mu_fixed
        c[2 * n + 1:] = 1                               # minimise sum t
        extra = []
        for s, offset in enumerate([0, n]):
            for j, i in enumerate(ex):
                if ids[i] in medium:
                    continue
                t = 2 * n + 1 + s * k + j
                for sign in (1, -1):        # sign * u <= t
                    r = np.zeros(N)
                    r[offset + i], r[t] = sign, -1
                    extra.append(r)
        Aub = np.vstack([Aub, extra])
        bub = np.zeros(len(Aub))
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds,
                  method="highs")
    return res


def alone(lb, ub):
    """Growth of one strain on its own: it can take up only what the
    medium supplies."""
    lb = lb.copy()
    for i in ex:
        if ids[i] not in medium:
            lb[i] = max(lb[i], 0.0)
    c = np.zeros(n)
    c[bio] = -1
    res = linprog(c, A_eq=S, b_eq=np.zeros(m), bounds=np.c_[lb, ub],
                  method="highs")
    return max(0.0, -res.fun) if res.status == 0 else 0.0


print(f"Alone: strain A grows at {alone(*A):.4f} /h, strain B at "
      f"{alone(*B):.4f} /h")

fractions = np.round(np.arange(0.01, 1.0, 0.01), 10)
growth = np.array([max(0.0, -community(x).fun) for x in fractions])
best = growth.max()
plateau = fractions[growth > best - 1e-6]
print(f"Community growth up to {best:.4f} /h, for fractions of A from "
      f"{plateau.min():.2f} to {plateau.max():.2f}")

# What must be exchanged? At x = 0.5, hold growth at its maximum and
# find the smallest total exchange of metabolites between the strains
x = 0.5
mu = -community(x).fun
res = community(x, minimise_exchange=True, mu_fixed=[mu * (1 - 1e-7), mu])
uA, uB = res.x[:n], res.x[n:2 * n]
traded = {ids[i]: [uA[i], uB[i]] for i in ex
          if ids[i] not in medium and max(abs(uA[i]), abs(uB[i])) > 1e-4}
print(f"At x = 0.5 (growth {mu:.4f} /h), the least exchange needed, in "
      f"mmol per gram of community per hour (+ secreted, - taken up):")
for r, (a, b) in traded.items():
    print(f"  {r:12s} A {a:+.3f}   B {b:+.3f}")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                               gridspec_kw={"width_ratios": [3, 2]})
ax1.plot(fractions, growth, color="#1f77b4")
ax1.axhline(alone(lb0, ub0), color="#999999", ls=":", lw=1)
ax1.text(0.5, 0.9, "wild type alone", ha="center", color="#666666",
         fontsize=9)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Fraction of strain A in the community")
ax1.set_ylabel("Community growth rate (1/h)")
ax1.set_title("Growth depends on composition")
label = {"EX_akg_e": "2-oxoglutarate", "EX_glu__L_e": "glutamate",
         "EX_lac__D_e": "D-lactate", "EX_succ_e": "succinate",
         "EX_mal__L_e": "malate", "EX_fum_e": "fumarate"}
names = [label.get(r, r) for r in traded]
y = np.arange(len(traded))
ax2.barh(y + 0.2, [v[0] for v in traded.values()], 0.4, color="#ff7f0e",
         label="strain A")
ax2.barh(y - 0.2, [v[1] for v in traded.values()], 0.4, color="#2ca02c",
         label="strain B")
ax2.set_yticks(y, names)
ax2.axvline(0, color="black", lw=0.5)
ax2.set_xlabel("Exchange flux (+ secreted, − taken up)")
ax2.set_title("Least cross-feeding, x = 0.5")
ax2.set_ylim(-0.6, len(traded) + 0.2)
ax2.legend(frameon=False, loc="upper center", ncol=2)
sbp.save_figure(fig, __file__, "fig-cross-feeding")

sbp.save_results(__file__, {
    "growth_alone": [alone(*A), alone(*B)],
    "fractions": fractions, "growth": growth, "best_growth": best,
    "plateau": [plateau.min(), plateau.max()],
    "least_exchange_total": res.fun,
})
sbp.show()
