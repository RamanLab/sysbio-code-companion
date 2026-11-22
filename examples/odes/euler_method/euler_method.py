"""Solving dx/dt = a - b*x with the forward and backward Euler methods.

This simple ODE has an exact solution, so we can see how the step size
dt affects accuracy and, for forward Euler, stability.
"""
import numpy as np
import matplotlib.pyplot as plt
import sbp

a, b, c = 20, 2, 5      # dx/dt = a - b*x,  x(0) = c
t_end = 4


def exact(t):
    return a / b - (a / b - c) * np.exp(-b * t)


def forward_euler(dt):
    """x(t + dt) = x(t) + f(x(t)) * dt"""
    n = round(t_end / dt)
    t = dt * np.arange(n + 1)
    x = np.zeros(n + 1)
    x[0] = c
    for i in range(n):
        x[i + 1] = x[i] + (a - b * x[i]) * dt
    return t, x


def backward_euler(dt):
    """x(t + dt) = x(t) + f(x(t + dt)) * dt, solved for x(t + dt).

    For a linear f this can be done by hand:
    x(t + dt) = (x(t) + a*dt) / (1 + b*dt).
    """
    n = round(t_end / dt)
    t = dt * np.arange(n + 1)
    x = np.zeros(n + 1)
    x[0] = c
    for i in range(n):
        x[i + 1] = (x[i] + a * dt) / (1 + b * dt)
    return t, x


# --- Forward Euler with different step sizes
steps = [0.05, 0.25, 0.9, 1.1]
max_error = []
for dt in steps:
    t, x = forward_euler(dt)
    max_error.append(np.abs(x - exact(t)).max())
    print(f"forward Euler, dt = {dt:4}: max error = {max_error[-1]:.4g}")
print(f"Forward Euler is stable only for dt < 2/b = {2 / b}")

# --- Forward vs backward Euler at a step size beyond the stability limit
t_f, x_f = forward_euler(1.1)
t_b, x_b = backward_euler(1.1)

# --- Plot
sbp.style()
colors = ["#0072B2", "#009E73", "#E69F00"]
t_fine = np.linspace(0, 4.4, 441)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(t_fine, exact(t_fine), "k-", lw=2, label="exact")
for dt, col in zip(steps[:3], colors):  # dt = 1.1 is in the right panel
    t, x = forward_euler(dt)
    ax1.plot(t, x, "o-", color=col, ms=4, label=f"Δt = {dt}")
ax1.set_ylim(0, 20)
ax1.set_xlabel("t")
ax1.set_ylabel("x(t)")
ax1.set_title("Forward Euler")
ax1.legend(frameon=False, loc="lower right")

ax2.plot(t_fine, exact(t_fine), "k-", lw=2, label="exact")
ax2.plot(t_f, x_f, "o-", color="#D55E00", ms=4, label="forward, Δt = 1.1")
ax2.plot(t_b, x_b, "s-", color="#0072B2", ms=4, label="backward, Δt = 1.1")
ax2.set_ylim(-2, 25)
ax2.set_xlabel("t")
ax2.set_title("Forward vs backward Euler")
ax2.legend(frameon=False, loc="upper left")
sbp.save_figure(fig, __file__, "fig-euler")

sbp.save_results(__file__, {
    "steps": steps,
    "max_error_forward": max_error,
    "forward_dt_1_1": x_f,
    "backward_dt_1_1": x_b,
})
sbp.show()
