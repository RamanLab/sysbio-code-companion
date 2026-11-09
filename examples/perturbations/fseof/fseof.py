"""Over-expression targets by FSEOF, and the production envelope.

Flux Scanning based on Enforced Objective Flux (FSEOF; book §10.3.1) for
succinate secretion by the E. coli core model on glucose:
  1. find the maximum growth rate, and the maximum succinate secretion;
  2. enforce succinate secretion at fractions f = 0, 0.1, ..., 0.9 of its
     maximum, and maximise growth each time;
  3. pick the reactions whose flux rises steadily with f: candidates for
     over-expression.
Each FBA problem has many optimal flux distributions, so at each step we
take the one with the smallest Euclidean norm, which is unique (as for
MoMA in §9.1). The production envelope (§10.6.1) shows the largest and
smallest succinate secretion possible at each growth rate.
"""
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import clarabel
import matplotlib.pyplot as plt
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
S = create_stoichiometric_matrix(model)
lb = np.array([r.lower_bound for r in model.reactions])
ub = np.array([r.upper_bound for r in model.reactions])
ids = [r.id for r in model.reactions]
n = len(ids)
bio, prod = ids.index("Biomass_Ecoli_core"), ids.index("EX_succ_e")


def optimum(j, lb, ub, sense=1):
    """Largest (sense = 1) or smallest (sense = -1) value of flux j."""
    c = np.zeros(n)
    c[j] = -sense
    res = linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]), bounds=np.c_[lb, ub],
                  method="highs")
    return -sense * res.fun


def nearest(target, lb, ub):
    """argmin ||v - target||^2 with S v = 0 and lb <= v <= ub (Clarabel)."""
    m = S.shape[0]
    eye = sparse.identity(n, format="csc")
    A = sparse.vstack([sparse.csc_matrix(S), eye, -eye], format="csc")
    b = np.r_[np.zeros(m), ub, -lb]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    sol = clarabel.DefaultSolver(eye, -np.asarray(target, dtype=float), A, b,
                                 [clarabel.ZeroConeT(m),
                                  clarabel.NonnegativeConeT(2 * n)],
                                 settings).solve()
    return np.array(sol.x)


mu_max = optimum(bio, lb, ub)
succ_max = optimum(prod, lb, ub)
print(f"Maximum growth {mu_max:.4f} /h; maximum succinate secretion "
      f"{succ_max:.4f} mmol/gDW/h")

# --- FSEOF
fractions = np.round(np.arange(0, 0.91, 0.1), 10)
growth, fluxes = [], []
for f in fractions:
    lb_f = lb.copy()
    lb_f[prod] = f * succ_max                  # enforce succinate secretion
    mu = optimum(bio, lb_f, ub)
    lb_f[bio] = mu * (1 - 1e-9)
    ub_f = ub.copy()
    ub_f[bio] = mu
    growth.append(mu)
    fluxes.append(nearest(np.zeros(n), lb_f, ub_f))
growth, F = np.array(growth), np.abs(np.array(fluxes))
internal = [j for j in range(n) if not ids[j].startswith(("EX_", "Biomass"))]
steps = np.diff(F, axis=0)
up = [ids[j] for j in internal if np.all(steps[:, j] > 1e-4)]
down = [ids[j] for j in internal if np.all(steps[:, j] < -1e-4)]
print(f"Growth falls from {growth[0]:.3f} to {growth[-1]:.3f} /h as "
      f"succinate secretion is enforced")
print(f"Fluxes that rise with succinate secretion ({len(up)}): "
      f"{', '.join(up)}")
print(f"Fluxes that fall ({len(down)}): {', '.join(down)}")

# --- Production envelope
mu_grid = np.linspace(0, mu_max, 21)
env_max, env_min = [], []
for mu in mu_grid:
    lb_m = lb.copy()
    lb_m[bio] = mu * (1 - 1e-9)
    env_max.append(optimum(prod, lb_m, ub, 1))
    env_min.append(optimum(prod, lb_m, ub, -1))
env_max, env_min = np.array(env_max), np.array(env_min)

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.fill_between(mu_grid, env_min, env_max, color="#c6dbef")
ax1.plot(mu_grid, env_max, color="#1f77b4")
ax1.plot(growth, fractions * succ_max, "o", color="#d62728",
         label="FSEOF steps")
ax1.set_xlabel("Growth rate (1/h)")
ax1.set_ylabel("Succinate secretion (mmol/gDW/h)")
ax1.set_title("Production envelope")
ax1.legend(frameon=False)
show = ["PPC", "PFK", "GAPD", "SUCCt3", "CS", "AKGDH"]
for r in show:
    ax2.plot(fractions * succ_max, F[:, ids.index(r)], "o-", ms=3, label=r)
ax2.set_xlabel("Enforced succinate secretion (mmol/gDW/h)")
ax2.set_ylabel("|flux| (mmol/gDW/h)")
ax2.set_title("Fluxes as succinate is enforced")
ax2.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
ax2.set_ylim(0, 26)
sbp.save_figure(fig, __file__, "fig-fseof")

sbp.save_results(__file__, {
    "growth_max": mu_max, "succinate_max": succ_max,
    "fseof_growth": growth, "up": up, "down": down,
    "envelope_growth": mu_grid, "envelope_max": env_max,
    "envelope_min": env_min,
})
sbp.show()
