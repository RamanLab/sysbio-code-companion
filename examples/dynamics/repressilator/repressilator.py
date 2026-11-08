"""Oscillations from a ring of three repressors: the repressilator.

Elowitz and Leibler (2000), in dimensionless form. Gene i makes mRNA m_i
and protein p_i, and each protein represses the next gene round the ring
(p3 represses gene 1, p1 gene 2, p2 gene 3):
    dm_i/dt = -m_i + alpha/(1 + p_j^n) + alpha0
    dp_i/dt = -beta (p_i - m_i)
The single steady state is unstable for these parameters, and the system
settles onto a limit cycle. We find the steady state and its eigenvalues,
simulate, measure the period, and map where in the (alpha, beta) plane
the steady state is unstable.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import sbp

alpha, alpha0, beta, n = 216.0, 0.216, 5.0, 2
repressor = [2, 0, 1]   # gene i is repressed by protein repressor[i]


def rhs(t, y, alpha=alpha, beta=beta):
    m, p = y[:3], y[3:]
    dm = -m + alpha / (1 + p[repressor] ** n) + alpha0
    dp = -beta * (p - m)
    return np.r_[dm, dp]


def steady_state(alpha, beta):
    """By symmetry m_i = p_i = s, with s = alpha/(1 + s^n) + alpha0."""
    return brentq(lambda s: alpha / (1 + s ** n) + alpha0 - s, 0,
                  alpha + alpha0)


def jacobian(s, alpha, beta):
    X = alpha * n * s ** (n - 1) / (1 + s ** n) ** 2   # -d(repression)/dp
    J = np.zeros((6, 6))
    for i in range(3):
        J[i, i] = -1
        J[i, 3 + repressor[i]] = -X
        J[3 + i, i] = beta
        J[3 + i, 3 + i] = -beta
    return J


s = steady_state(alpha, beta)
eig = np.linalg.eigvals(jacobian(s, alpha, beta))
max_real = eig.real.max()
state = "unstable" if max_real > 0 else "stable"
print(f"Steady state m = p = {s:.4f}; largest real part of an "
      f"eigenvalue = {max_real:.4f} ({state})")

# --- Simulate from a slightly asymmetric start, and find the period from
# successive upward crossings of p1 through its steady value
def crossing(t, y):
    return y[3] - s
crossing.direction = 1

y0 = [1.0, 0.0, 0.0, 2.0, 1.0, 3.0]
sol = solve_ivp(rhs, [0, 300], y0, rtol=1e-8, atol=1e-10,
                events=crossing, dense_output=True)
t_up = sol.t_events[0]
t_up = t_up[t_up > 150]                  # after the transient has died away
period = np.mean(np.diff(t_up))
t = np.linspace(0, 50, 1001)
Y = sol.sol(t)
late = sol.sol(np.linspace(150, 300, 3001))[3]
amplitude = [late.min(), late.max()]
print(f"Period = {period:.4f}; p1 oscillates between {amplitude[0]:.4f} "
      f"and {amplitude[1]:.4f}")

# --- Where is the steady state unstable? A grid in the (alpha, beta) plane
alphas = np.logspace(0, 4, 41)
betas = np.logspace(-1, 3, 41)
unstable = np.zeros((len(betas), len(alphas)), dtype=bool)  # rows: beta
for r, b in enumerate(betas):
    for c, a in enumerate(alphas):
        ss = steady_state(a, b)
        unstable[r, c] = np.linalg.eigvals(jacobian(ss, a, b)).real.max() > 0
print(f"Steady state unstable at {unstable.sum()} of {unstable.size} "
      f"grid points")

# --- Plot
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, name in enumerate(["p1", "p2", "p3"]):
    axes[0].plot(t, Y[3 + i], label=name)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Protein level")
axes[0].set_title("Oscillations")
axes[0].set_ylim(0, 75)
axes[0].legend(frameon=False, loc="upper right", ncol=3)
axes[1].contourf(alphas, betas, unstable.astype(float), levels=[0.5, 1.5],
                 colors=["#c6dbef"])
axes[1].contour(alphas, betas, unstable.astype(float), levels=[0.5],
                colors="k", linewidths=1)
axes[1].plot(alpha, beta, "*", color="#d62728", ms=12)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel(r"$\alpha$ (maximum transcription)")
axes[1].set_ylabel(r"$\beta$ (mRNA / protein lifetime)")
axes[1].set_title("Steady state unstable (shaded)")
sbp.save_figure(fig, __file__, "fig-repressilator")

sbp.save_results(__file__, {
    "steady_state": s, "max_real_eigenvalue": max_real,
    "period": period, "amplitude": amplitude,
    "n_unstable": int(unstable.sum()), "unstable": unstable.astype(int),
})
sbp.show()
