"""Full mass-action enzyme model vs the Michaelis-Menten rate law.

    E + S <-> ES -> E + P      (rate constants k1, k_1, k2)

The full model tracks S, ES and P; the enzyme is E = E0 - ES. The
Michaelis-Menten (Briggs-Haldane) approximation replaces all of this
with a single rate law, v = vmax*S/(KM + S). How good is it?
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

k1, k_1, k2 = 1.0, 1.0, 1.0       # 1/(mM s), 1/s, 1/s
KM = (k_1 + k2) / k1               # Michaelis constant: 2 mM
S0 = 10.0                          # initial substrate (mM)
tol = dict(rtol=1e-8, atol=1e-10)  # tight tolerances for both solvers


def full_model(t, y, E0):
    """Mass-action ODEs for S, ES and P (book Eqs. 5.4a–d)."""
    S, ES, P = y
    E = E0 - ES                    # enzyme conservation (Eq. 5.6)
    return [-k1 * E * S + k_1 * ES,
            k1 * E * S - (k_1 + k2) * ES,
            k2 * ES]


def mm_model(t, y, E0):
    """A single ODE for S, with the Michaelis-Menten rate law."""
    S = y[0]
    vmax = k2 * E0
    return [-vmax * S / (KM + S)]


def simulate(E0, t_end):
    t = np.linspace(0, t_end, 51)
    full = solve_ivp(full_model, [0, t_end], [S0, 0, 0], t_eval=t,
                     args=(E0,), **tol)
    mm = solve_ivp(mm_model, [0, t_end], [S0], t_eval=t, args=(E0,), **tol)
    P_full = full.y[2]
    P_mm = S0 - mm.y[0]            # in the MM model, P = S0 - S
    return t, P_full, P_mm


# Two cases: little enzyme (E0 << S0) and a lot of enzyme (E0 ~ S0)
cases = {"low_enzyme": (0.1, 150), "high_enzyme": (5.0, 6)}  # E0, t_end
results = {}
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, (E0, t_end)) in zip(axes, cases.items()):
    t, P_full, P_mm = simulate(E0, t_end)
    err = np.abs(P_full - P_mm).max() / S0
    print(f"E0 = {E0} mM: max |P_full - P_MM| = {100 * err:.1f}% of S0")
    results[name] = {"E0": E0, "t": t, "P_full": P_full, "P_mm": P_mm,
                     "max_rel_difference": err}
    ax.plot(t, P_full, "-", color="#1f77b4", lw=2, label="full mass action")
    ax.plot(t, P_mm, "--", color="#ff7f0e", lw=2, label="Michaelis–Menten")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Product P (mM)")
    ax.set_title(f"$E_0$ = {E0} mM, $S_0$ = {S0:g} mM")
    ax.legend(frameon=False, loc="lower right")
sbp.save_figure(fig, __file__, "fig-michaelis-menten")

sbp.save_results(__file__, results)
sbp.show()
