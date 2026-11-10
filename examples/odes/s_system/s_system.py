"""Biochemical systems theory: an S-system approximation of a pathway.

A two-step pathway with feedback inhibition, written with familiar rate
laws (book §5.3):
    v1 = V1 / (1 + (X2/Ki)^h)       (supply, inhibited by the end product)
    v2 = V2 X1 / (K2 + X1)          (conversion of X1 to X2)
    v3 = V3 X2 / (K3 + X2)          (demand for X2)
    dX1/dt = v1 - v2,   dX2/dt = v2 - v3.
In the S-system form of BST (book Eq. 5.21), each rate is replaced by a
power law, v ~ gamma X^f, whose kinetic order f is the elasticity
d ln v / d ln X at the steady state, and gamma matches the rate there:
    dX1/dt = a1 X2^g12 - b1 X1^h11,   dX2/dt = a2 X1^g21 - b2 X2^h22.
We compare the two models when the supply V1 is raised by 10%, 50% and
200%. Note that the steady state of an S-system can be found with linear
algebra, since in logarithms its steady-state equations are linear.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import sbp

V1, Ki, h = 2.0, 1.0, 2.0
V2, K2 = 3.0, 1.0
V3, K3 = 2.5, 1.0


def rates(x1, x2, v1max=V1):
    return (v1max / (1 + (x2 / Ki) ** h), V2 * x1 / (K2 + x1),
            V3 * x2 / (K3 + x2))


def steady_state(v1max=V1):
    """v1 = v3 fixes X2; then v2 = v3 fixes X1."""
    x2 = brentq(lambda x: rates(1, x, v1max)[0] - rates(1, x)[2], 1e-9, 100,
               xtol=1e-14)
    flux = rates(1, x2)[2]
    x1 = flux * K2 / (V2 - flux)                      # v2(x1) = flux
    return np.array([x1, x2])


x1s, x2s = steady_state()
flux = rates(x1s, x2s)[2]
# Kinetic orders: elasticities at the steady state
g12 = -h * (x2s / Ki) ** h / (1 + (x2s / Ki) ** h)
h11 = g21 = K2 / (K2 + x1s)
h22 = K3 / (K3 + x2s)
# Rate constants: each power law equals the flux at the steady state
a1 = flux / x2s ** g12
b1 = a2 = flux / x1s ** h11
b2 = flux / x2s ** h22
print(f"Steady state: X1 = {x1s:.4f}, X2 = {x2s:.4f}, flux = {flux:.4f}")
print(f"Kinetic orders: g12 = {g12:.3f}, h11 = g21 = {h11:.3f}, "
      f"h22 = {h22:.3f}")


def full_model(t, x, v1max):
    v1, v2, v3 = rates(x[0], x[1], v1max)
    return [v1 - v2, v2 - v3]


def bst_model(t, x, factor):
    return [factor * a1 * x[1] ** g12 - b1 * x[0] ** h11,
            a2 * x[0] ** g21 - b2 * x[1] ** h22]


def s_steady_state(factor):
    """In logs: g12 y2 - h11 y1 = ln(b1 / (factor a1)), and
    g21 y1 - h22 y2 = ln(b2 / a2), with y = ln X."""
    M = np.array([[-h11, g12], [g21, -h22]])
    rhs = np.array([np.log(b1 / (factor * a1)), np.log(b2 / a2)])
    return np.exp(np.linalg.solve(M, rhs))


factors = [1.1, 1.5, 3.0]
t = np.linspace(0, 10, 201)
results = {"steady_state": [x1s, x2s], "kinetic_orders": [g12, h11, h22]}
sbp.style()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, f in zip(axes, factors):
    full = solve_ivp(full_model, [0, 10], [x1s, x2s], t_eval=t,
                     args=(f * V1,), rtol=1e-10, atol=1e-12).y
    approx = solve_ivp(bst_model, [0, 10], [x1s, x2s], t_eval=t,
                       args=(f,), rtol=1e-10, atol=1e-12).y
    exact_ss, bst_ss = steady_state(f * V1), s_steady_state(f)
    results[f"supply_up_{round(100 * (f - 1))}"] = {"full": exact_ss, "s_system": bst_ss,
                               "error_percent": 100 * (bst_ss / exact_ss - 1)}
    print(f"Supply x{f}: new steady state, full model X1 = {exact_ss[0]:.4f},"
          f" X2 = {exact_ss[1]:.4f}; S-system X1 = {bst_ss[0]:.4f}, "
          f"X2 = {bst_ss[1]:.4f}")
    for i, (name, colour) in enumerate((("X1", "#1f77b4"), ("X2", "#d62728"))):
        ax.plot(t, full[i], color=colour, label=f"{name}, full model")
        ax.plot(t, approx[i], "--", color=colour, label=f"{name}, S-system")
    ax.set_title(f"Supply raised {round(100 * (f - 1))}%")
    ax.set_xlabel("Time")
axes[0].set_ylabel("Concentration")
axes[0].legend(frameon=False, fontsize=8)
sbp.save_figure(fig, __file__, "fig-s-system")

sbp.save_results(__file__, results)
sbp.show()
