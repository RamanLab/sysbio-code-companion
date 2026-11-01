"""Grid search, and the shape of the error landscape.

The same competitive-inhibition model and data as in fit_inhibition.
We evaluate the weighted SSE on a grid of whole-number values of KM and
KI (1 to 60 mM). The rate is proportional to vmax, so for each (KM, KI)
the best vmax can be computed exactly, as in linear regression, and the
grid only needs two dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt
import sbp

data = np.loadtxt(sbp.DATA / "competitive_inhibition.csv", delimiter=",",
                  skiprows=1)
S, v10, v20 = data[:, 0], data[:, 1], data[:, 2]
delta = 0.1
datasets = {
    "D1": (S, np.full_like(S, 10.0), v10),
    "D1_D2": (np.r_[S, S], np.r_[np.full_like(S, 10.0),
                                  np.full_like(S, 20.0)], np.r_[v10, v20]),
}
KM_grid = np.arange(1, 61)
KI_grid = np.arange(1, 61)


def best_vmax_and_sse(KM, KI, S, I, v):
    """Weighted residuals are vmax*a - b, so the best vmax is sum(a*b)/sum(a^2)."""
    a = S / (KM * (1 + I / KI) + S) / (delta * v)
    b = 1 / delta
    vmax = np.sum(a * b) / np.sum(a * a)
    return vmax, np.sum((vmax * a - b) ** 2)


results = {}
SSE = {}
for name, (s, i, v) in datasets.items():
    grid = np.zeros((len(KI_grid), len(KM_grid)))   # rows: KI, columns: KM
    vmax = np.zeros_like(grid)
    for r, KI in enumerate(KI_grid):
        for c, KM in enumerate(KM_grid):
            vmax[r, c], grid[r, c] = best_vmax_and_sse(KM, KI, s, i, v)
    r, c = np.unravel_index(np.argmin(grid), grid.shape)
    # How many grid points fit almost as well as the best (within 1 unit)?
    n_close = int(np.sum(grid < grid[r, c] + 1))
    print(f"{name}: best on grid KM = {KM_grid[c]}, KI = {KI_grid[r]}, "
          f"vmax = {vmax[r, c]:.4f}, SSE = {grid[r, c]:.4f}; "
          f"{n_close} grid points within 1 of the best SSE")
    SSE[name] = grid
    results[name] = {"KM": KM_grid[c], "KI": KI_grid[r],
                     "vmax": vmax[r, c], "SSE": grid[r, c],
                     "n_within_1": n_close,
                     "SSE_at_KI_20": grid[KI_grid == 20, :].ravel()}

# --- Plot: log10(SSE - best SSE + 1), so that the best point is at 0
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
levels = np.linspace(0, 3, 13)
titles = {"D1": "D1 only (I = 10 mM)", "D1_D2": "D1 and D2 together"}
for ax, name in zip(axes, SSE):
    Z = np.log10(SSE[name] - SSE[name].min() + 1)
    cs = ax.contourf(KM_grid, KI_grid, Z, levels=levels, cmap="viridis_r",
                     extend="max")
    ax.plot(results[name]["KM"], results[name]["KI"], "*", color="#d62728",
            ms=14)
    ax.set_xlabel("$K_M$ (mM)")
    ax.set_title(titles[name])
axes[0].set_ylabel("$K_I$ (mM)")
fig.colorbar(cs, ax=axes, label="log$_{10}$(SSE − best SSE + 1)")
sbp.save_figure(fig, __file__, "fig-sse-landscape")

sbp.save_results(__file__, results)
sbp.show()
