"""Within-host dynamics of an infection: the target-cell-limited model.

The simplest model of viral dynamics in a host (book §11.2.2.2, after
Perelson): target cells T are infected by virus V, at rate beta T V;
infected cells I produce virus at rate p and die at rate delta; free
virus is cleared at rate c.
    dT/dt = -beta T V
    dI/dt =  beta T V - delta I
    dV/dt =  p I - c V
The parameters are those estimated by Baccam et al. (2006) for influenza
A in people. The basic reproductive number, R0 = beta p T0 / (c delta),
is the number of cells infected by the virus from one infected cell, at
the start. An antiviral (a neuraminidase inhibitor) that blocks virus
release with efficacy eps multiplies p by (1 - eps); we compare
treatment started on day 1, 2 and 3 of the infection.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

beta, p, c, delta = 2.7e-5, 1.2e-2, 3.0, 4.0     # per day; V in TCID50/mL
T0, V0 = 4e8, 9.3e-2
R0 = beta * p * T0 / (c * delta)


def rhs(t, y, production):
    T, I, V = y
    return [-beta * T * V, beta * T * V - delta * I, production * I - c * V]


def simulate(start=np.inf, eps=0.0):
    """Up to day `start`, untreated; from then on, virus production is
    reduced by the antiviral. Returns T, I and V every 0.01 days."""
    t = np.linspace(0, 10, 1001)
    before = t <= start
    y = np.zeros((3, len(t)))
    sol = solve_ivp(rhs, [0, t[before][-1]], [T0, 0, V0], t_eval=t[before],
                    args=(p,), method="LSODA", rtol=1e-10, atol=1e-8)
    y[:, before] = sol.y
    if start < t[-1]:
        sol = solve_ivp(rhs, [start, t[-1]], sol.y[:, -1], t_eval=t[~before],
                        args=(p * (1 - eps),), method="LSODA", rtol=1e-10,
                        atol=1e-8)
        y[:, ~before] = sol.y
    return t, y


results = {"R0": R0}
print(f"R0 = {R0:.1f}")
scenarios = {"untreated": (np.inf, 0.0), "day_1": (1.0, 0.98),
             "day_2": (2.0, 0.98), "day_3": (3.0, 0.98)}
curves = {}
for name, (start, eps) in scenarios.items():
    t, (T, I, V) = simulate(start, eps)
    k = np.argmax(V)
    dt = t[1] - t[0]
    # (the day of the peak is printed, but not stored: on a grid of
    # 0.01 days, two solvers could disagree about which point is highest)
    results[name] = {"peak_virus": V[k], "target_cells_left": T[-1] / T0,
                     "virus_area": np.sum((V[1:] + V[:-1]) / 2) * dt}
    curves[name] = V
    print(f"{name:10s} peak {V[k]:.3g} TCID50/mL on day {t[k]:.2f}; "
          f"{100 * T[-1] / T0:.1f}% of target cells left on day 10")

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(6.5, 4.5))
colours = {"untreated": "black", "day_1": "#2ca02c", "day_2": "#1f77b4",
           "day_3": "#d62728"}
for name, V in curves.items():
    ax.semilogy(t, np.maximum(V, 1e-3), color=colours[name],
                label=name.replace("_", " ").replace("day", "treated, day"))
ax.set_ylim(1e-2, 1e7)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Virus (TCID50/mL)")
ax.legend(frameon=False)
ax.set_title(f"Influenza A, target-cell-limited model (R0 = {R0:.1f})")
sbp.save_figure(fig, __file__, "fig-viral-dynamics")

sbp.save_results(__file__, results)
sbp.show()
