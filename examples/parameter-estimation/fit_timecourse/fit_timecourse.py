"""Fitting an ODE model to a time course.

A gene is switched on at t = 0; its mRNA m and protein p follow
    dm/dt = k_m - g_m m,    dp/dt = k_p m - g_p p,    m(0) = p(0) = 0.
We estimate the four parameters from measurements of both mRNA and
protein, and then from the protein alone. Each evaluation of the
residuals solves the ODEs; the fits start from the same eight guesses.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import sbp

data = np.loadtxt(sbp.DATA / "gene_expression_timecourse.csv",
                  delimiter=",", skiprows=1)
t, m_obs, p_obs = data.T
delta = 0.1                       # relative measurement error (10%)


def simulate(p, t):
    k_m, g_m, k_p, g_p = p
    rhs = lambda _, y: [k_m - g_m * y[0], k_p * y[0] - g_p * y[1]]
    sol = solve_ivp(rhs, [0, t[-1]], [0, 0], t_eval=t, rtol=1e-10,
                    atol=1e-10)
    return sol.y                  # rows: mRNA, protein


def residuals(p, use_mrna):
    m, prot = simulate(p, t)
    r = (prot - p_obs) / (delta * p_obs)
    if use_mrna:
        r = np.r_[(m - m_obs) / (delta * m_obs), r]
    return r


starts = [(1, 1, 1, 1), (5, 0.5, 5, 0.5), (20, 2, 1, 0.1),
          (1, 0.1, 20, 2), (10, 0.3, 3, 1), (3, 3, 3, 0.3),
          (50, 5, 0.5, 0.05), (0.5, 0.05, 50, 5)]
fits = {}
for use_mrna, name in [(True, "mRNA and protein"), (False, "protein only")]:
    rows = []
    for p0 in starts:
        sol = least_squares(residuals, p0, args=(use_mrna,),
                            bounds=(1e-6, 100), xtol=1e-12, ftol=1e-12,
                            gtol=1e-12)
        rows.append([*sol.x, np.sum(sol.fun ** 2)])
    fits[name] = np.array(rows)
    print(f"Fits to {name} (k_m, g_m, k_p, g_p, SSE):")
    for r in fits[name]:
        print("  " + "  ".join(f"{x:8.4f}" for x in r))

# Best fits, and what the protein data alone can determine
both = fits["mRNA and protein"]
best_both = both[np.argmin(both[:, 4])]
prot = fits["protein only"]
good = prot[prot[:, 4] < prot[:, 4].min() + 1e-6]       # equally good fits
product = good[:, 0] * good[:, 2]                        # k_m * k_p
rates = np.sort(good[:, [1, 3]], axis=1)                 # {g_m, g_p}
print(f"Protein only: {len(good)} of {len(prot)} fits reach the best SSE, "
      f"{prot[:, 4].min():.4f}; across them k_m*k_p = "
      f"{product.min():.3f} to {product.max():.3f}, and the two "
      f"degradation rates are {rates[:, 0].mean():.4f} and "
      f"{rates[:, 1].mean():.4f}")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
tt = np.linspace(0, 12, 241)
mm, pp = simulate(best_both[:4], tt)
ax1.plot(t, p_obs, "o", color="#1f77b4", label="protein (data)")
ax1.plot(tt, pp, color="#1f77b4", label="protein (fit)")
ax1.plot(t, m_obs * 10, "s", color="#ff7f0e", ms=4,
         label="mRNA × 10 (data)")
ax1.plot(tt, mm * 10, color="#ff7f0e", label="mRNA × 10 (fit)")
ax1.set_xlabel("Time (h)")
ax1.set_ylabel("Molecules per cell")
ax1.set_title("Fit to mRNA and protein")
ax1.legend(frameon=False, fontsize=9)
k = np.linspace(0.5, 100, 400)
ax2.loglog(k, product.mean() / k, ":", color="#999999",
           label=r"$k_m k_p$ = constant")
for name, marker, colour in [("mRNA and protein", "s", "#ff7f0e"),
                             ("protein only", "o", "#1f77b4")]:
    f = fits[name]
    best = f[:, 4] < f[:, 4].min() + 1e-6
    ax2.loglog(f[best, 0], f[best, 2], marker, color=colour, ms=8,
               alpha=0.7, label=f"best fits, {name}")
ax2.set_xlim(0.5, 100)
ax2.set_ylim(0.5, 100)
ax2.set_xlabel("$k_m$ (molecules/h)")
ax2.set_ylabel("$k_p$ (1/h)")
ax2.set_title("Estimates from eight starting points")
ax2.legend(frameon=False, fontsize=9)
sbp.save_figure(fig, __file__, "fig-fit-timecourse")

sbp.save_results(__file__, {
    "best_both": best_both[:4], "sse_both": best_both[4],
    "sse_protein": prot[:, 4].min(),
    "product_protein": product.mean(),
    "rates_protein": rates.mean(axis=0),
})
sbp.show()
