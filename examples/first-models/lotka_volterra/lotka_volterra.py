"""The Lotka-Volterra predator-prey model (book Eqs. 1.4 and 1.5).

    dD/dt = b*D - p*L*D      deer (prey): born at rate b, eaten by lions
    dL/dt = r*D*L - d*L      lions (predators): grow by eating, die at rate d

The populations oscillate. Each starting point gives its own closed orbit
around the steady state (D*, L*) = (d/r, b/p), because the quantity
    V = r*D - d*ln(D) + p*L - b*ln(L)
stays constant along every solution.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

b, p, r, d = 1.0, 0.1, 0.02, 0.5      # per year (p, r per animal per year)
D_star, L_star = d / r, b / p         # steady state: 25 deer, 10 lions
tol = dict(rtol=1e-10, atol=1e-10)


def lotka_volterra(t, y):
    D, L = y
    return [b * D - p * L * D, r * D * L - d * L]


def V(D, L):
    return r * D - d * np.log(D) + p * L - b * np.log(L)


t = np.linspace(0, 30, 301)
starts = [(40, 9), (60, 9), (80, 9)]
sols = [solve_ivp(lotka_volterra, [0, 30], y0, t_eval=t, **tol)
        for y0 in starts]

# Period of the first orbit: times at which the deer population passes
# upwards through D*, found by the solver as 'events'
def crossing(t, y):
    return y[0] - D_star
crossing.direction = 1
ev = solve_ivp(lotka_volterra, [0, 30], starts[0], events=crossing, **tol)
period = np.diff(ev.t_events[0]).mean()

D, L = sols[0].y
drift = np.abs(V(D, L) - V(D[0], L[0])).max()
print(f"Steady state: D* = {D_star:g} deer, L* = {L_star:g} lions")
small = 2 * np.pi / np.sqrt(b * d)
print(f"Period of the orbit from (40, 9): {period:.4f} years "
      f"(small oscillations: 2*pi/sqrt(b*d) = {small:.4f})")
print(f"Largest change in V along the solution: {drift:.1e}")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(t, D, "-", color="#1f77b4", label="deer, D")
ax1.plot(t, L, "-", color="#d62728", label="lions, L")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Population")
ax1.set_title("Starting from 40 deer and 9 lions")
ax1.set_ylim(0, 50)
ax1.legend(frameon=False, loc="upper center", ncol=2)
colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
for sol, y0, col in zip(sols, starts, colors):
    ax2.plot(sol.y[0], sol.y[1], "-", color=col,
             label=f"start ({y0[0]}, {y0[1]})")
ax2.plot(D_star, L_star, "k+", ms=10, mew=2, label="steady state")
ax2.set_xlabel("Deer, D")
ax2.set_ylabel("Lions, L")
ax2.set_title("Phase plane")
ax2.set_ylim(0, 32)
ax2.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
sbp.save_figure(fig, __file__, "fig-lotka-volterra")

sbp.save_results(__file__, {
    "steady_state": [D_star, L_star],
    "period": period,
    "V_drift": drift,
    "D_every_5_years": D[::50], "L_every_5_years": L[::50],
})
sbp.show()
