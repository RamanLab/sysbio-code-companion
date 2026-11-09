"""Model selection: which kind of inhibition do the data support?

The data are those of fit_inhibition: initial rates at 20 substrate
concentrations, with an inhibitor at 10 mM and at 20 mM. We fit four
rate laws, with three or four parameters, by weighted least squares
(10% relative error), and compare them by
  * the weighted SSE, chi^2 (book Eq. 6.9, with sigma = 0.1 v);
  * the Akaike Information Criterion, AIC = chi^2 + 2k, for k parameters,
    with the correction for small samples, AICc = AIC + 2k(k+1)/(n-k-1);
  * Akaike weights, the relative support for each model;
  * the cross-correlation of the parameter estimates (book Eq. 6.11),
    from the covariance matrix V = (J'J)^-1, where J is the Jacobian of
    the weighted residuals at the optimum.
"""
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import sbp

data = np.loadtxt(sbp.DATA / "competitive_inhibition.csv", delimiter=",",
                  skiprows=1)
S = np.r_[data[:, 0], data[:, 0]]
I = np.r_[np.full(20, 10.0), np.full(20, 20.0)]
v = np.r_[data[:, 1], data[:, 2]]
n = len(v)
delta = 0.1

models = {
    "competitive":    (["vmax", "KM", "KI"],
                       lambda p, S, I: p[0] * S / (p[1] * (1 + I / p[2]) + S)),
    "uncompetitive":  (["vmax", "KM", "KI"],
                       lambda p, S, I: p[0] * S / (p[1] + S * (1 + I / p[2]))),
    "noncompetitive": (["vmax", "KM", "KI"],
                       lambda p, S, I: p[0] * S / ((p[1] + S) * (1 + I / p[2]))),
    "mixed":          (["vmax", "KM", "KI", "KI2"],
                       lambda p, S, I: p[0] * S / (p[1] * (1 + I / p[2])
                                             + S * (1 + I / p[3]))),
}


def residuals(rate, p):
    return (rate(p, S, I) - v) / (delta * v)


def jacobian(rate, p):
    """Central differences of the weighted residuals."""
    J = np.zeros((n, len(p)))
    for j in range(len(p)):
        h = 1e-6 * p[j]
        up, down = p.copy(), p.copy()
        up[j] += h
        down[j] -= h
        J[:, j] = (residuals(rate, up) - residuals(rate, down)) / (2 * h)
    return J


results = {}
for name, (names, rate) in models.items():
    k = len(names)
    best = None
    for scale_km in (0.2, 1, 5):           # nine starting guesses
        for scale_ki in (0.2, 1, 5):
            p0 = np.array([20, 5 * scale_km] + [5 * scale_ki] * (k - 2))
            if k == 4:
                p0[3] *= 10
            sol = least_squares(lambda p: residuals(rate, p), p0,
                                bounds=(0, np.inf), xtol=1e-14, ftol=1e-14,
                                gtol=1e-14)
            if best is None or sol.cost < best.cost:
                best = sol
    p = best.x
    chi2 = np.sum(residuals(rate, p) ** 2)
    V = np.linalg.inv(jacobian(rate, p).T @ jacobian(rate, p))
    sd = np.sqrt(np.diag(V))
    kappa = V / np.outer(sd, sd)
    aic = chi2 + 2 * k
    results[name] = {"parameters": p, "sd": sd, "chi2": chi2, "aic": aic,
                     "aicc": aic + 2 * k * (k + 1) / (n - k - 1),
                     "kappa": kappa}

aicc = np.array([r["aicc"] for r in results.values()])
weights = np.exp(-(aicc - aicc.min()) / 2)
weights /= weights.sum()
for (name, r), w in zip(results.items(), weights):
    r["weight"] = w
    names = models[name][0]
    print(f"{name:15s} chi2 = {r['chi2']:6.2f}  AIC = {r['aic']:6.2f}  "
          f"AICc = {r['aicc']:6.2f}  weight = {w:.2f}")
    print("   " + ", ".join(f"{a} = {b:.3g} ± {c:.2g}"
                            for a, b, c in zip(names, r["parameters"], r["sd"])))
    worst = np.max(np.abs(r["kappa"] - np.eye(len(names))))
    print(f"   largest |cross-correlation| = {worst:.3f}")

# --- Plot: data and the two best models, and the Akaike weights
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                               gridspec_kw={"width_ratios": [1.4, 1]})
s_fine = np.linspace(0.5, 20, 200)
for level, marker in ((10, "o"), (20, "s")):
    keep = I == level
    ax1.errorbar(S[keep], v[keep], yerr=delta * v[keep], fmt=marker,
                 color="black", ms=4, capsize=2, label=f"data, I = {level} mM")
    for name, colour in (("competitive", "#1f77b4"),
                         ("noncompetitive", "#d62728")):
        rate = models[name][1]
        ax1.plot(s_fine, rate(results[name]["parameters"], s_fine, level),
                 color=colour, label=name if level == 10 else None)
ax1.set_xlabel("S (mM)")
ax1.set_ylabel("v (mmol/L/s)")
ax1.legend(frameon=False, fontsize=9)
ax2.bar(list(results), weights, color="#1f77b4")
ax2.set_ylabel("Akaike weight")
ax2.tick_params(axis="x", rotation=30)
ax2.set_title("Support for each model")
sbp.save_figure(fig, __file__, "fig-model-selection")

sbp.save_results(__file__, results)
sbp.show()
