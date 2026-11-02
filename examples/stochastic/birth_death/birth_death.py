"""Stochastic production and degradation of an mRNA.

    0 -> M     at rate k          (transcription)
    M -> 0     at rate gamma * M  (degradation)

The deterministic (ODE) model, dM/dt = k - gamma*M, predicts a smooth
rise to M = k/gamma. With few molecules, each reaction is a random
event, simulated here with Gillespie's algorithm. At steady state the
number of molecules follows a Poisson distribution with mean k/gamma.
"""
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import sbp
from sbp.ssa import gillespie

k, gamma = 10.0, 1.0              # per minute
t_grid = np.linspace(0, 10, 201)  # minutes
n_runs = 2000
rng = np.random.default_rng(1)

changes = [[+1], [-1]]


def propensities(x):
    return np.array([k, gamma * x[0]])


runs = np.array([gillespie([0], changes, propensities, t_grid, rng)[:, 0]
                 for _ in range(n_runs)])
final = runs[:, -1]
ode = k / gamma * (1 - np.exp(-gamma * t_grid))

print(f"At t = {t_grid[-1]:g} min, over {n_runs} runs: mean {final.mean():.3f}"
      f", variance {final.var():.3f} (Poisson: both {k / gamma:g})")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
colors = ["#1f77b4", "#2ca02c", "#9467bd"]
for r, col in zip(range(3), colors):
    ax1.step(t_grid, runs[r], where="post", color=col, lw=1,
             label="one simulation" if r == 0 else None)
ax1.plot(t_grid, runs.mean(axis=0), "-", color="#ff7f0e", lw=2,
         label=f"mean of {n_runs} simulations")
ax1.plot(t_grid, ode, "k--", lw=1.5, label="ODE")
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("Number of mRNA molecules")
ax1.set_ylim(0, 25)
ax1.legend(frameon=False, loc="upper left")
ax1.set_title("Trajectories")

m = np.arange(0, 26)
counts = np.bincount(final.astype(int), minlength=26)[:26] / n_runs
ax2.bar(m, counts, color="#1f77b4", alpha=0.7,
        label=f"simulated, t = {t_grid[-1]:g} min")
ax2.plot(m, poisson.pmf(m, k / gamma), "o-", color="#ff7f0e", ms=4,
         label=f"Poisson, mean {k / gamma:g}")
ax2.set_xlabel("Number of mRNA molecules")
ax2.set_ylabel("Probability")
ax2.set_ylim(0, 0.17)
ax2.legend(frameon=False, loc="upper right")
ax2.set_title("Distribution at steady state")
sbp.save_figure(fig, __file__, "fig-birth-death")

sbp.save_results(__file__, {
    "steady_state": {"mean": final.mean(), "variance": final.var()},
    "mean_trajectory": runs.mean(axis=0)[::20],
})
sbp.show()
