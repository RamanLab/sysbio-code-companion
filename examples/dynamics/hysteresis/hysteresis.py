"""Bistability and hysteresis in the toggle switch.

The same toggle switch, du/dt = a1/(1 + v^n) - u, dv/dt = a2/(1 + u^n) - v,
now with a2 = 5 fixed and a1, the strength of the promoter making u,
varied from 1 to 12 (in an experiment, an inducer does this). For each a1
we find all steady states and their stability. Then we simulate the
switch while slowly raising a1 and lowering it again: the state the
switch is in depends on where it has been.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import sbp

a2, n = 5.0, 2
a1_values = np.round(np.arange(1, 12.0001, 0.05), 10)


def steady_states(a1):
    """All steady values of u, as roots of g(u) on [0, a1], with stability."""
    def g(u):
        return a1 / (1 + (a2 / (1 + u ** n)) ** n) - u
    grid = np.linspace(0, a1, 4001)
    vals = g(grid)
    roots = []
    for k in np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:])):
        u = brentq(g, grid[k], grid[k + 1])
        if not roots or u - roots[-1] > 1e-6:
            roots.append(u)
    out = []
    for u in roots:
        v = a2 / (1 + u ** n)
        J = np.array([[-1, -a1 * n * v ** (n - 1) / (1 + v ** n) ** 2],
                      [-a2 * n * u ** (n - 1) / (1 + u ** n) ** 2, -1]])
        out.append((u, bool(np.linalg.eigvals(J).real.max() < 0)))
    return out


branches = [steady_states(a1) for a1 in a1_values]
n_states = np.array([len(b) for b in branches])
bistable = a1_values[n_states == 3]
print(f"Bistable (three steady states) for a1 from {bistable.min():.2f} to "
      f"{bistable.max():.2f}")


# --- Hysteresis: raise a1 step by step, letting the switch settle each
# time from where it was, then lower it again
def toggle(t, y, a1):
    u, v = y
    return [a1 / (1 + v ** n) - u, a2 / (1 + u ** n) - v]


def sweep(a1_path, y0):
    """Settle at each a1 in turn, starting from the previous state."""
    y, out = np.array(y0, dtype=float), []
    for a1 in a1_path:
        y = solve_ivp(toggle, [0, 100], y, args=(a1,), rtol=1e-8,
                      atol=1e-10).y[:, -1]
        out.append(y)
    return np.array(out)


ramp = np.round(np.arange(1, 12.0001, 0.25), 10)
up = sweep(ramp, [0.0, a2])                        # start with u off
down = sweep(ramp[::-1], up[-1])[::-1]
# u is 'on' when it exceeds v
switch_up = ramp[np.argmax(up[:, 0] > up[:, 1])]   # first a1 with u on
switch_down = ramp[np.argmax(down[:, 0] > down[:, 1])]  # lowest a1 with u on
print(f"Raising a1, u switches on at a1 = {switch_up:g}; lowering a1, u "
      f"stays on down to a1 = {switch_down:g}")
up, down = up[:, 0], down[:, 0]

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(7, 4.5))
for a1, b in zip(a1_values, branches):
    for u, stable in b:
        ax.plot(a1, u, ".", ms=3, color="k" if stable else "#bbbbbb")
ax.plot(ramp, up, "^", color="#1f77b4", ms=6, mfc="none",
        label="simulated, a1 increasing")
ax.plot(ramp, down, "v", color="#ff7f0e", ms=6, mfc="none",
        label="simulated, a1 decreasing")
ax.plot([], [], ".", color="k", label="stable steady state")
ax.plot([], [], ".", color="#bbbbbb", label="unstable steady state")
ax.axvspan(bistable.min(), bistable.max(), color="#eeeeee", zorder=0)
ax.set_xlabel("Promoter strength a1")
ax.set_ylabel("Steady-state level of u")
ax.set_title("Bistability (shaded) and hysteresis")
ax.legend(frameon=False, loc="upper left", fontsize=9)
sbp.save_figure(fig, __file__, "fig-hysteresis")

sbp.save_results(__file__, {
    "bistable_range": [bistable.min(), bistable.max()],
    "n_states": n_states,
    "switch_up": switch_up, "switch_down": switch_down,
    "up": up, "down": down,
})
sbp.show()
