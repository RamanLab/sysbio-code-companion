"""Sloppiness: the eigenvalues of the Hessian of the cost function.

Near its minimum, the weighted SSE behaves like a quadratic form whose
Hessian is approximated by J'J, where J is the Jacobian of the weighted
residuals (the Fisher Information Matrix, FIM). Following Gutenkunst et
al., we differentiate with respect to log(theta), so that the
eigenvalues measure relative changes in the parameters. A model is
sloppy if lambda_min / lambda_max < 1e-3 (book §6.5.2).

The model and data are those of fit_timecourse, at the parameters fitted
there, and we compare two experiments: measuring both mRNA and protein,
or the protein alone.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

names = ["k_m", "g_m", "k_p", "g_p"]
theta = np.array([9.4984, 0.96342, 5.2262, 0.20479])   # fit_timecourse
data = np.loadtxt(sbp.DATA / "gene_expression_timecourse.csv",
                  delimiter=",", skiprows=1)
t, m_obs, p_obs = data.T
delta = 0.1


def simulate(p):
    k_m, g_m, k_p, g_p = p
    rhs = lambda _, y: [k_m - g_m * y[0], k_p * y[0] - g_p * y[1]]
    sol = solve_ivp(rhs, [0, t[-1]], [0, 0], t_eval=t, method="LSODA",
                    rtol=1e-11, atol=1e-11)
    return sol.y


def residuals(log_p, use_mrna):
    m, prot = simulate(np.exp(log_p))
    r = (prot - p_obs) / (delta * p_obs)
    if use_mrna:
        r = np.r_[(m - m_obs) / (delta * m_obs), r]
    return r


def fim(use_mrna):
    """J'J, by central differences in log(theta)."""
    log_p = np.log(theta)
    columns = []
    for j in range(len(theta)):
        up, down = log_p.copy(), log_p.copy()
        up[j] += 1e-5
        down[j] -= 1e-5
        columns.append((residuals(up, use_mrna) - residuals(down, use_mrna))
                       / 2e-5)
    J = np.array(columns).T
    return J.T @ J


results = {}
for key, use_mrna in (("mrna_and_protein", True), ("protein_only", False)):
    eigenvalues, eigenvectors = np.linalg.eigh(fim(use_mrna))
    order = np.argsort(eigenvalues)[::-1]          # largest first
    eigenvalues = eigenvalues[order]
    stiffest = eigenvectors[:, order[0]]
    sloppiest = eigenvectors[:, order[-1]]
    first = sloppiest[np.abs(sloppiest) > 1e-3][0]  # fix the sign: first
    sign = 1 if first > 0 else -1                      #   clear entry positive
    results[key] = {"eigenvalues": eigenvalues,
                    "ratio": eigenvalues[-1] / eigenvalues[0],
                    "sloppiest": sign * sloppiest}
    print(f"{key}: eigenvalues " + ", ".join(f"{e:.3g}" for e in eigenvalues)
          + f";  lambda_min/lambda_max = {results[key]['ratio']:.2g}")
    print("   sloppiest direction (in log parameters): "
          + ", ".join(f"{n} {c:+.2f}" for n, c in zip(names, sign * sloppiest)))

# --- Plot: the eigenvalue spectra
sbp.style()
fig, ax = plt.subplots(figsize=(5, 4.5))
for x, (key, label) in enumerate([("mrna_and_protein", "mRNA and protein"),
                                  ("protein_only", "protein only")]):
    ev = np.maximum(results[key]["eigenvalues"], 1e-12)  # for a log scale
    ax.hlines(ev / ev[0], x - 0.3, x + 0.3, color="#1f77b4")
ax.set_yscale("log")
ax.set_ylim(1e-12, 3)
ax.axhline(1e-3, ls=":", color="black", lw=1)
ax.set_xticks([0, 1], ["mRNA and protein", "protein only"])
ax.set_ylabel("Eigenvalue / largest eigenvalue")
ax.set_title("Eigenvalues of J'J")
sbp.save_figure(fig, __file__, "fig-sloppiness")

sbp.save_results(__file__, results)
sbp.show()
