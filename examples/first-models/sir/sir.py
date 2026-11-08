"""The SIR model of an epidemic (book Eqs. 1.6 to 1.8).

    dS/dt = -beta*S*I
    dI/dt =  beta*S*I - gamma*I
    dR/dt =  gamma*I

S, I and R are fractions of the population. With almost everyone
susceptible at the start, an outbreak grows only if the basic
reproduction number R0 = beta/gamma exceeds 1. The fraction finally
infected, R_inf, then satisfies  R_inf = 1 - S0*exp(-R0*R_inf).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import sbp

gamma = 0.1                        # recovery rate: 10 days to recover
I0 = 1e-4                          # one person in 10,000 infected at first
S0 = 1 - I0
tol = dict(rtol=1e-8, atol=1e-10)


def sir(t, y, beta):
    S, I, R = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]


def final_size(R0):
    """Solve R_inf = 1 - S0*exp(-R0*R_inf) for the epidemic's final size.
    The equation has exactly one root between 0 and 1."""
    return brentq(lambda x: x - (1 - S0 * np.exp(-R0 * x)), 0, 1)


# --- One epidemic, R0 = 3
beta = 0.3
t = np.linspace(0, 200, 201)       # days
sol = solve_ivp(sir, [0, 200], [S0, I0, 0], t_eval=t, args=(beta,), **tol)
S, I, R = sol.y
peak = np.argmax(I)
print(f"R0 = {beta / gamma:g}: peak of {I[peak]:.4f} infected on day "
      f"{t[peak]:g}; after 200 days R = {R[-1]:.4f} "
      f"(final-size equation: {final_size(beta / gamma):.4f})")

# --- Final size for a range of R0
R0s = np.linspace(0.5, 4, 36)
simulated = [solve_ivp(sir, [0, 5000], [S0, I0, 0], args=(R0 * gamma,),
                       **tol).y[2, -1] for R0 in R0s]
theory = [final_size(R0) for R0 in R0s]

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for y, lab, col in [(S, "S, susceptible", "#1f77b4"),
                    (I, "I, infected", "#d62728"),
                    (R, "R, recovered", "#2ca02c")]:
    ax1.plot(t, y, "-", color=col, label=lab)
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Fraction of the population")
ax1.set_title(r"An epidemic with $R_0$ = 3")
ax1.legend(frameon=False, loc="center right")
ax2.plot(R0s, theory, "-", color="#999999", label="final-size equation")
ax2.plot(R0s, simulated, "o", color="#2ca02c", ms=4, label="simulation")
ax2.axvline(1, color="#999999", ls=":")
ax2.set_xlabel("Basic reproduction number $R_0$")
ax2.set_ylabel("Fraction ever infected")
ax2.set_title("Final size of the epidemic")
ax2.legend(frameon=False, loc="lower right")
sbp.save_figure(fig, __file__, "fig-sir")

sbp.save_results(__file__, {
    "peak_infected": I[peak], "peak_day": t[peak], "final_R": R[-1],
    "final_size_R0_3": final_size(3.0),
    "R0": R0s, "final_size_simulated": simulated,
})
sbp.show()
