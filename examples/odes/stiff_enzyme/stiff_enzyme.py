"""Stiffness: when fast binding makes a simple model expensive to solve.

The same enzyme model as before, but now binding and unbinding (k1, k_1)
are much faster than catalysis (k2). The fast and slow time scales make
the ODEs 'stiff'. We compare a non-stiff solver (RK45, like ode45) with a
stiff one (BDF, like ode15s) by counting how often each one evaluates the
right-hand side of the ODEs.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

k2, E0, S0, t_end = 1.0, 0.1, 10.0, 50.0
tol = dict(rtol=1e-6, atol=1e-9)


def enzyme(t, y, k1, k_1):
    S, ES, P = y
    E = E0 - ES
    return [-k1 * E * S + k_1 * ES,
            k1 * E * S - (k_1 + k2) * ES,
            k2 * ES]


# --- How the cost grows as binding gets faster (k1 = k_1 = k)
speeds = [1, 10, 100, 1000]
cost = {"nonstiff": [], "stiff": []}
final_P = []
for k in speeds:
    nonstiff = solve_ivp(enzyme, [0, t_end], [S0, 0, 0], method="RK45",
                         args=(k, k), **tol)
    stiff = solve_ivp(enzyme, [0, t_end], [S0, 0, 0], method="BDF",
                      args=(k, k), **tol)
    cost["nonstiff"].append(nonstiff.nfev)
    cost["stiff"].append(stiff.nfev)
    final_P.append(stiff.y[2, -1])
    print(f"k = {k:5}: RK45 used {nonstiff.nfev:7} evaluations, "
          f"BDF {stiff.nfev:5};  P({t_end:g}) = {stiff.y[2, -1]:.4f} mM")

# --- The fast transient, for k = 100, on a logarithmic time axis
t = np.logspace(-4, np.log10(t_end), 41)
sol = solve_ivp(enzyme, [0, t_end], [S0, 0, 0], method="BDF", t_eval=t,
                args=(100, 100), **tol)

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogx(t, sol.y[1] / E0, "-", color="#009E73", lw=2,
             label="ES / $E_0$ (fast)")
ax1.semilogx(t, sol.y[2] / S0, "-", color="#0072B2", lw=2,
             label="P / $S_0$ (slow)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Fraction")
ax1.set_title("Two time scales ($k_1 = k_{-1} = 100$)")
ax1.legend(frameon=False)

ax2.loglog(speeds, cost["nonstiff"], "o-", color="#E69F00",
           label="non-stiff solver")
ax2.loglog(speeds, cost["stiff"], "s-", color="#0072B2",
           label="stiff solver")
ax2.set_xlabel("Binding rate constants $k_1 = k_{-1}$")
ax2.set_ylabel("Function evaluations")
ax2.set_title("Cost of solving to t = 50 s")
ax2.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-stiff")

sbp.save_results(__file__, {
    "speeds": speeds,
    "final_P": final_P,
    "t": t,
    "ES_k100": sol.y[1],
    "P_k100": sol.y[2],
    "evaluations_nonstiff": cost["nonstiff"],
    "evaluations_stiff": cost["stiff"],
})
sbp.show()
