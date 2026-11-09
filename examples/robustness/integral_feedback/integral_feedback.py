"""Robust adaptation by integral feedback.

A signal u drives an output y, which is removed by a controller x; y in
turn makes x, and x is degraded by an enzyme with Michaelis constant K:
    dy/dt = c u - e x y
    dx/dt = a y - b x / (K + x)
When the enzyme is saturated (K small), x integrates the difference
between a y and b, and y returns to b/a after any change in u, whatever
the other parameters: perfect adaptation. We compare K = 0.01 with
K = 1, first in time, then for 81 combinations of a, b, c and e, each
at half, once and twice its nominal value of 1.
"""
import itertools
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

K_values = [0.01, 1.0]


def steady_y(u, a, b, c, e, K):
    """At steady state, r = x / (K + x) = a y / b solves
    e K b r^2 + a c u r - a c u = 0."""
    p = a * c * u
    r = (-p + np.sqrt(p ** 2 + 4 * e * K * b * p)) / (2 * e * K * b)
    return r * b / a


def u_of_t(t):
    return 2.0 if 20 <= t < 60 else 1.0


def rhs(t, z, K, a=1, b=1, c=1, e=1):
    y, x = z
    return [c * u_of_t(t) - e * x * y, a * y - b * x / (K + x)]


# --- Time courses with the nominal parameters: u doubles at t = 20 and
# returns to 1 at t = 60
t = np.linspace(0, 100, 2001)
courses, peaks = {}, []
for K in K_values:
    y0 = steady_y(1, 1, 1, 1, 1, K)
    x0 = 1 / y0                     # from c u = e x y
    sol = solve_ivp(rhs, [0, 100], [y0, x0], args=(K,), method="Radau",
                    t_eval=t, rtol=1e-8, atol=1e-10, max_step=0.1)
    courses[K] = sol.y[0]
    peaks.append(sol.y[0][(t >= 20) & (t < 60)].max())
    print(f"K = {K}: y = {y0:.4f} before the step, peak {peaks[-1]:.4f}, "
          f"{courses[K][t == 59.95][0]:.4f} just before u falls again")

# --- 81 parameter sets: how much does the steady-state y change when u
# doubles?
factors = [0.5, 1.0, 2.0]
params = np.array(list(itertools.product(factors, repeat=4)))   # a b c e
ratio = {K: np.array([steady_y(2, *p, K) / steady_y(1, *p, K)
                      for p in params]) for K in K_values}
level = {K: np.array([steady_y(1, *p, K) for p in params])
         for K in K_values}
for K in K_values:
    print(f"K = {K}: y(u = 2) / y(u = 1) ranges from {ratio[K].min():.4f} "
          f"to {ratio[K].max():.4f}; the level y(u = 1) from "
          f"{level[K].min():.3f} to {level[K].max():.3f}")

# --- Plot
sbp.style()
colours = {0.01: "#1f77b4", 1.0: "#ff7f0e"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for K in K_values:
    ax1.plot(t, courses[K], color=colours[K], label=f"K = {K:g}")
ax1.axvspan(20, 60, color="#eeeeee", zorder=0)
ax1.text(40, 2.05, "u = 2", ha="center", color="#666666")
ax1.set_ylim(0.3, 2.2)
ax1.set_xlabel("Time")
ax1.set_ylabel("Output y")
ax1.set_title("Response to a step in the signal")
ax1.legend(frameon=False)
for K in K_values:
    ax2.semilogx(level[K], ratio[K], "o", color=colours[K], alpha=0.7,
                 label=f"K = {K:g}")
ax2.axhline(1, color="#999999", ls=":", lw=1)
ax2.set_xticks([0.2, 0.5, 1, 2, 4], labels=["0.2", "0.5", "1", "2", "4"])
ax2.minorticks_off()
ax2.set_xlabel("Steady-state y before the step")
ax2.set_ylabel("y after / y before")
ax2.set_title("81 parameter sets")
ax2.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-integral-feedback")

sbp.save_results(__file__, {
    "peak": peaks,
    "ratio_K001": ratio[0.01], "ratio_K1": ratio[1.0],
    "level_K001": level[0.01], "level_K1": level[1.0],
})
sbp.show()
