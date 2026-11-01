"""Fitting a rate law to data by weighted least squares.

Competitive inhibition:  v = vmax*S / (KM*(1 + I/KI) + S).
Rates were measured at 20 substrate concentrations, with the inhibitor
at I = 10 mM (dataset D1) and at I = 20 mM (dataset D2). We estimate
vmax, KM and KI from D1 alone, and from both datasets together.
"""
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import sbp

data = np.loadtxt(sbp.DATA / "competitive_inhibition.csv", delimiter=",",
                  skiprows=1)
S, v10, v20 = data[:, 0], data[:, 1], data[:, 2]
delta = 0.1                       # relative measurement error (10%)


def rate(p, S, I):
    vmax, KM, KI = p
    return vmax * S / (KM * (1 + I / KI) + S)


def residuals(p, S, I, v):
    """Weighted residuals; their sum of squares is the SSE (book Eq. 6.5)."""
    return (rate(p, S, I) - v) / (delta * v)


def fit(S, I, v, p0):
    sol = least_squares(residuals, p0, args=(S, I, v), bounds=(0, np.inf),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return sol.x, np.sum(sol.fun ** 2)


# The same eight starting guesses for every fit: vmax = 20 and a spread
# of values for KM and KI
starts = [(20, km, ki) for km, ki in
          [(1, 1), (2, 8), (5, 2), (5, 20), (10, 5), (20, 40), (40, 10),
           (40, 80)]]

datasets = {
    "D1": (S, np.full_like(S, 10.0), v10),
    "D2": (S, np.full_like(S, 20.0), v20),
    "D1_D2": (np.r_[S, S], np.r_[np.full_like(S, 10.0),
                                  np.full_like(S, 20.0)], np.r_[v10, v20]),
}

results = {}
for name, (s, i, v) in datasets.items():
    fits = [fit(s, i, v, p0) for p0 in starts]
    P = np.array([p for p, _ in fits])
    SSE = np.array([sse for _, sse in fits])
    best = np.argmin(SSE)
    # KM*(1 + I/KI) at the first inhibitor level of the dataset
    KM_app = P[:, 1] * (1 + i[0] / P[:, 2])
    print(f"{name}: SSE = {SSE[best]:.4f} (spread over starts "
          f"{np.ptp(SSE):.1e})")
    print(f"   vmax = {P[best, 0]:.4f};  KM ranges {P[:, 1].min():.3g} to "
          f"{P[:, 1].max():.3g};  KI ranges {P[:, 2].min():.3g} to "
          f"{P[:, 2].max():.3g}")
    # What every start agrees on; KM and KI only if they are pinned down
    results[name] = {"SSE": SSE[best], "vmax": P[best, 0],
                     "KM_app": KM_app[best]}
    if name == "D1_D2":
        results[name].update(KM=P[best, 1], KI=P[best, 2])
    else:
        results[name + "_estimates"] = {"KM": P[:, 1], "KI": P[:, 2]}

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
p = [results["D1_D2"][k] for k in ("vmax", "KM", "KI")]
S_fine = np.linspace(0, 21, 200)
for I, v, col in [(10, v10, "#1f77b4"), (20, v20, "#ff7f0e")]:
    ax1.errorbar(S, v, yerr=delta * v, fmt="o", color=col, ms=4,
                 capsize=2, label=f"data, I = {I} mM")
    ax1.plot(S_fine, rate(p, S_fine, I), "-", color=col)
ax1.set_xlabel("S (mM)")
ax1.set_ylabel("v (mmol L$^{-1}$ s$^{-1}$)")
ax1.set_title("Fit to D1 and D2 together")
ax1.legend(frameon=False, loc="lower right")

KI_line = np.logspace(-1, 2.3, 200)
ax2.plot(results["D1"]["KM_app"] / (1 + 10 / KI_line), KI_line, "-",
         color="#999999", label="$K_M(1 + 10/K_I)$ = best $K_M^{app}$ for D1")
est = results["D1_estimates"]
ax2.plot(est["KM"], est["KI"], "o", color="#1f77b4", ms=6,
         label="estimates from D1, 8 starts")
ax2.plot(p[1], p[2], "*", color="#d62728", ms=14,
         label="estimate from D1 and D2")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("$K_M$ (mM)")
ax2.set_ylabel("$K_I$ (mM)")
ax2.set_title("Where the estimates land")
ax2.legend(frameon=False, loc="upper left", fontsize=9)
sbp.save_figure(fig, __file__, "fig-fit-inhibition")

sbp.save_results(__file__, results)
sbp.show()
