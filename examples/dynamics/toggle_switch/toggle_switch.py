"""Steady states, stability and the phase plane: a genetic toggle switch.

Two genes repress each other (Gardner, Cantor and Collins, 2000):
    du/dt = a1/(1 + v^n) - u
    dv/dt = a2/(1 + u^n) - v
We find every steady state, classify each by the eigenvalues of the
Jacobian matrix there, and draw the phase plane.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import sbp

a1, a2, n = 10.0, 10.0, 2


def toggle(t, y):
    u, v = y
    return [a1 / (1 + v ** n) - u, a2 / (1 + u ** n) - v]


def jacobian(u, v):
    return np.array([[-1, -a1 * n * v ** (n - 1) / (1 + v ** n) ** 2],
                     [-a2 * n * u ** (n - 1) / (1 + u ** n) ** 2, -1]])


# Steady states: on the v-nullcline v = a2/(1 + u^n), so the steady values
# of u are the roots of g(u) = a1/(1 + v(u)^n) - u, which lie in [0, a1].
def g(u):
    return a1 / (1 + (a2 / (1 + u ** n)) ** n) - u


grid = np.linspace(0, a1, 10001)
vals = g(grid)
roots = []
for k in np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:])):
    u = brentq(g, grid[k], grid[k + 1])
    if not roots or u - roots[-1] > 1e-6:   # a root on a grid point is
        roots.append(u)                     # found twice; keep it once
states = []
for u in roots:
    v = a2 / (1 + u ** n)
    eig = np.sort(np.linalg.eigvals(jacobian(u, v)).real)
    kind = "stable" if eig.max() < 0 else "unstable (saddle)"
    states.append({"u": u, "v": v, "eigenvalues": eig,
                   "stable": bool(eig.max() < 0)})
    print(f"Steady state u = {u:.4f}, v = {v:.4f}: eigenvalues "
          f"{eig[0]:.4f}, {eig[1]:.4f} -> {kind}")

# --- Trajectories from a grid of starting points (off the diagonal u = v,
# which by symmetry leads to the saddle)
starts = [(u0, v0) for u0 in np.linspace(0.25, 11.75, 8)
          for v0 in np.linspace(0.5, 12, 8)]
trajs = [solve_ivp(toggle, [0, 20], y0, rtol=1e-8, atol=1e-10,
                   t_eval=np.linspace(0, 20, 401)).y for y0 in starts]
ends_high_u = [bool(Y[0, -1] > Y[1, -1]) for Y in trajs]

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(6, 5.5))
U, V = np.meshgrid(np.linspace(0, 12, 25), np.linspace(0, 12, 25))
dU, dV = a1 / (1 + V ** n) - U, a2 / (1 + U ** n) - V
ax.quiver(U, V, dU, dV, color="#cccccc", angles="xy")
for Y, high in zip(trajs, ends_high_u):
    ax.plot(Y[0], Y[1], "-", lw=0.8,
            color="#1f77b4" if high else "#ff7f0e")
x = np.linspace(0, 12, 400)
ax.plot(a1 / (1 + x ** n), x, "k-", lw=2, label="u-nullcline (du/dt = 0)")
ax.plot(x, a2 / (1 + x ** n), "k--", lw=2, label="v-nullcline (dv/dt = 0)")
for s in states:
    ax.plot(s["u"], s["v"], "o", ms=10, mec="k", mew=1.5,
            color="k" if s["stable"] else "white")
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_xlabel("Repressor 1, u")
ax.set_ylabel("Repressor 2, v")
ax.set_title("Two stable states (filled) and a saddle (open)")
ax.legend(frameon=True, loc="upper right", fontsize=9)
sbp.save_figure(fig, __file__, "fig-toggle-switch")

sbp.save_results(__file__, {
    "steady_states": states,
    "fraction_to_high_u": np.mean(ends_high_u),
})
sbp.show()
