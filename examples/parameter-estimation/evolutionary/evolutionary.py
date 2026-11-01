"""A simple evolutionary algorithm for parameter estimation.

The same model and data as in fit_inhibition (D1 and D2 together), but
now estimated by a (10 + 40) evolution strategy, which needs no starting
guess and no gradients:
  1. start from 50 random parameter sets and keep the best 10 (parents);
  2. each generation, make 40 children, each a copy of a random parent
     with small random changes to its parameters ('mutation');
  3. keep the best 10 of parents and children; shrink the mutations a
     little; repeat.
Parameters are searched on a log scale, from 0.1 to about 316.
"""
import numpy as np
import matplotlib.pyplot as plt
import sbp

data = np.loadtxt(sbp.DATA / "competitive_inhibition.csv", delimiter=",",
                  skiprows=1)
S = np.r_[data[:, 0], data[:, 0]]
I = np.r_[np.full(20, 10.0), np.full(20, 20.0)]
v = np.r_[data[:, 1], data[:, 2]]
delta = 0.1


def sse(log_p):
    vmax, KM, KI = 10.0 ** log_p
    v_pred = vmax * S / (KM * (1 + I / KI) + S)
    return np.sum(((v_pred - v) / (delta * v)) ** 2)


rng = np.random.default_rng(1)     # fixed seed: the same run every time
lo, hi = -1.0, 2.5                 # bounds on log10 of each parameter
n_parents, n_children, n_generations = 10, 40, 60
sigma = 0.5                        # initial mutation size (log10 units)

# 1. Random start; keep the best 10
pop = rng.uniform(lo, hi, size=(50, 3))
cost = np.array([sse(p) for p in pop])
keep = np.argsort(cost)[:n_parents]
parents, parent_cost = pop[keep], cost[keep]
snapshots = {0: parents.copy()}
best_per_generation = [parent_cost[0]]
n_evaluations = 50

for g in range(1, n_generations + 1):
    # 2. Children: mutated copies of randomly chosen parents
    pick = rng.integers(n_parents, size=n_children)
    children = parents[pick] + sigma * rng.standard_normal((n_children, 3))
    children = np.clip(children, lo, hi)
    child_cost = np.array([sse(p) for p in children])
    n_evaluations += n_children
    # 3. Survival of the fittest, among parents and children together
    everyone = np.vstack([parents, children])
    everyone_cost = np.r_[parent_cost, child_cost]
    keep = np.argsort(everyone_cost)[:n_parents]
    parents, parent_cost = everyone[keep], everyone_cost[keep]
    sigma *= 0.93
    best_per_generation.append(parent_cost[0])
    if g in (5, 20):
        snapshots[g] = parents.copy()
snapshots[n_generations] = parents.copy()

vmax, KM, KI = 10.0 ** parents[0]
print(f"Best after {n_generations} generations ({n_evaluations} SSE "
      f"evaluations): vmax = {vmax:.3f}, KM = {KM:.3f}, KI = {KI:.3f}, "
      f"SSE = {parent_cost[0]:.4f}")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogy(best_per_generation, "o-", color="#1f77b4", ms=3)
ax1.axhline(63.7765, color="#999999", ls="--",
            label="least-squares optimum (fit_inhibition)")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Best SSE so far")
ax1.set_title("Convergence")
ax1.legend(frameon=False)

colors = {0: "#bbbbbb", 5: "#ff7f0e", 20: "#2ca02c", n_generations: "#1f77b4"}
for g, P in snapshots.items():
    ax2.plot(10.0 ** P[:, 1], 10.0 ** P[:, 2], "o", color=colors[g], ms=5,
             label=f"generation {g}")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(0.1, 316)
ax2.set_ylim(0.1, 316)
ax2.set_xlabel("$K_M$ (mM)")
ax2.set_ylabel("$K_I$ (mM)")
ax2.set_title("The 10 parents, as the search proceeds")
ax2.legend(frameon=False, loc="lower right")
sbp.save_figure(fig, __file__, "fig-evolutionary")

sbp.save_results(__file__, {
    "best": {"SSE": parent_cost[0], "vmax": vmax, "KM": KM, "KI": KI},
    "n_evaluations": n_evaluations,
    "best_per_generation": best_per_generation,
})
sbp.show()
