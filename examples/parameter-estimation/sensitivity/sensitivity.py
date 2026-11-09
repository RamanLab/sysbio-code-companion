"""Local sensitivity analysis of a gene expression model.

The model is that of fit_timecourse: a gene switched on at t = 0, with
    dm/dt = k_m - g_m m,    dp/dt = k_p m - g_p p,    m(0) = p(0) = 0,
at the parameters fitted there to mRNA and protein data. The local
sensitivity of state x_i to parameter theta_j (book Eq. 6.12) is
S_ij(t) = dx_i(t)/dtheta_j, which we compute by central differences:
solve the ODEs with theta_j raised and lowered by 0.01%. We report the
normalised sensitivity (theta_j / x_i) dx_i/dtheta_j, the percentage
change in x_i for a 1% change in theta_j, which can be compared across
parameters of different units.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

names = ["k_m", "g_m", "k_p", "g_p"]
theta = np.array([9.4984, 0.96342, 5.2262, 0.20479])   # fit_timecourse
t = np.linspace(0.5, 24, 48)


def simulate(p):
    k_m, g_m, k_p, g_p = p
    rhs = lambda _, y: [k_m - g_m * y[0], k_p * y[0] - g_p * y[1]]
    sol = solve_ivp(rhs, [0, t[-1]], [0, 0], t_eval=t, method="LSODA",
                    rtol=1e-11, atol=1e-11)
    return sol.y                                  # rows: mRNA, protein


x = simulate(theta)
normalised = np.zeros((2, len(t), len(theta)))   # state x time x parameter
for j in range(len(theta)):
    h = 1e-4 * theta[j]
    up, down = theta.copy(), theta.copy()
    up[j] += h
    down[j] -= h
    dx = (simulate(up) - simulate(down)) / (2 * h)
    normalised[:, :, j] = theta[j] * dx / x

print("Normalised sensitivity of protein to each parameter:")
print("   t (h)  " + "".join(f"{n:>8s}" for n in names))
for k in (0, 3, 11, 23, 47):
    print(f"  {t[k]:5.1f}   " + "".join(f"{s:8.3f}" for s in normalised[1, k]))
print("At steady state, p* = k_m k_p / (g_m g_p): the sensitivities "
      "approach +1, -1, +1, -1.")

# --- Plot
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for ax, i, title in zip(axes, (0, 1), ("mRNA", "Protein")):
    for j, (n, c) in enumerate(zip(names, colours)):
        ax.plot(t, normalised[i, :, j], color=c, label=n)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Time (h)")
    ax.set_title(title)
axes[0].set_ylabel("Normalised sensitivity")
axes[1].legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-sensitivity")

sbp.save_results(__file__, {"t": t, "mrna": normalised[0],
                            "protein": normalised[1]})
sbp.show()
