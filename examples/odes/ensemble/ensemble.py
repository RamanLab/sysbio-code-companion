"""Handling uncertainty with an ensemble of models.

Parameters estimated from data are uncertain, and so are the predictions
made with them. One way to see how uncertain is to build an ensemble:
many parameter sets that are all consistent with the data, each making
its own prediction (book §5.6.2). Predictions on which the members agree
can be trusted; those on which they disagree cannot, however good the fit.

The model and data are those of fit_timecourse (gene expression, mRNA and
protein measured for 12 h). Near the best fit, the parameters (in logs)
are approximately Gaussian, with covariance (J'J)^-1, where J is the
Jacobian of the weighted residuals (see sloppiness, §6.5.2). We draw 200
parameter sets from this distribution, and compare their predictions:
  * the protein level at 12 h (within the data) and at 48 h (beyond it);
  * the protein level at 48 h if translation is halved;
  * the half-lives of the mRNA and of the protein.
The Gaussian numbers come from sbp.uniform (Box-Muller), so that the
ensemble is the same in MATLAB and Python.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

theta = np.array([9.4984, 0.96342, 5.2262, 0.20479])   # fit_timecourse
data = np.loadtxt(sbp.DATA / "gene_expression_timecourse.csv",
                  delimiter=",", skiprows=1)
t_obs, m_obs, p_obs = data.T
delta = 0.1


def simulate(p, t):
    k_m, g_m, k_p, g_p = p
    rhs = lambda _, y: [k_m - g_m * y[0], k_p * y[0] - g_p * y[1]]
    return solve_ivp(rhs, [0, t[-1]], [0, 0], t_eval=t, method="LSODA",
                     rtol=1e-10, atol=1e-10).y


def residuals(log_p):
    m, prot = simulate(np.exp(log_p), t_obs)
    return np.r_[(m - m_obs) / (delta * m_obs), (prot - p_obs) / (delta * p_obs)]


# Covariance of log(theta): (J'J)^-1, J by central differences
log_theta = np.log(theta)
J = np.zeros((2 * len(t_obs), 4))
for j in range(4):
    up, down = log_theta.copy(), log_theta.copy()
    up[j] += 1e-5
    down[j] -= 1e-5
    J[:, j] = (residuals(up) - residuals(down)) / 2e-5
L = np.linalg.cholesky(np.linalg.inv(J.T @ J))

# 200 parameter sets: log(theta) + L z, z standard Gaussian
members = 200
u = sbp.uniform(31, 2 * 4 * members)
z = (np.sqrt(-2 * np.log(u[0::2])) * np.cos(2 * np.pi * u[1::2]))
z = z.reshape(members, 4)
members_theta = np.exp(log_theta + z @ L.T)

t = np.array([12.0, 48.0])
predictions = {"protein_12h": [], "protein_48h": [],
               "protein_48h_half_translation": [],
               "mrna_half_life": [], "protein_half_life": []}
for p in members_theta:
    prot = simulate(p, t)[1]
    halved = simulate(p * [1, 1, 0.5, 1], t)[1]
    predictions["protein_12h"].append(prot[0])
    predictions["protein_48h"].append(prot[1])
    predictions["protein_48h_half_translation"].append(halved[1])
    predictions["mrna_half_life"].append(np.log(2) / p[1])
    predictions["protein_half_life"].append(np.log(2) / p[3])


def summary(x):
    """Median, and the 10th and 190th of the 200 sorted values (the 5th
    and 95th percentiles)."""
    s = np.sort(x)
    return {"median": np.median(s), "low": s[9], "high": s[189],
            "spread_percent": 100 * (s[189] - s[9]) / np.median(s)}


results = {k: summary(v) for k, v in predictions.items()}
print("Prediction (200 members)          median   5% to 95%       spread")
for k, r in results.items():
    print(f"  {k:30s} {r['median']:7.2f}   {r['low']:6.2f} to {r['high']:6.2f}"
          f"   {r['spread_percent']:5.1f}%")

# --- Plot: the ensemble of protein time courses
sbp.style()
fig, ax = plt.subplots(figsize=(6.5, 4.5))
tt = np.linspace(0, 48, 97)
for p in members_theta[:50]:
    ax.plot(tt, simulate(p, tt)[1], color="#1f77b4", alpha=0.15)
ax.plot(t_obs, p_obs, "o", color="black", ms=4, label="data (12 h)")
ax.axvline(12, ls=":", color="black")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Protein (molecules per cell)")
ax.set_title("50 members of the ensemble")
ax.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-ensemble")

sbp.save_results(__file__, results)
sbp.show()
