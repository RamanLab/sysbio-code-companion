"""A generalised Lotka-Volterra (gLV) model of a three-species community,
and estimating its parameters from time courses.

    dx_i/dt = x_i (r_i + sum_j a_ij x_j)

r_i is the growth rate of species i on its own, a_ii < 0 limits its
growth, and a_ij (i != j) is the effect of species j on species i:
negative for competition, positive for cross-feeding. We find the
steady state where all three species coexist, x* = -A^-1 r, and its
stability, from the eigenvalues of the Jacobian diag(x*) A.

Then we turn the problem around, as one does with real abundance data
(book §11.1.2): we simulate three experiments that start from different
mixtures, "measure" the abundances every half day with 5% noise, and
estimate r and A by linear regression, since
    d ln x_i / dt = r_i + sum_j a_ij x_j
is linear in the parameters. The noise is Gaussian, made from
sbp.uniform by the Box-Muller method, so that it is identical in MATLAB
and Python.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sbp

species = ["A", "B", "C"]
r = np.array([1.0, 0.8, 0.6])
A = np.array([[-1.0, -0.4, 0.3],
              [-0.5, -1.0, 0.0],
              [0.2, -0.3, -0.8]])


def simulate(x0, t):
    sol = solve_ivp(lambda _, x: x * (r + A @ x), [0, t[-1]], x0, t_eval=t,
                    method="LSODA", rtol=1e-10, atol=1e-12)
    return sol.y.T                                   # time x species


x_star = -np.linalg.solve(A, r)
eigenvalues = np.linalg.eigvals(np.diag(x_star) @ A)
eigenvalues = eigenvalues[np.lexsort((eigenvalues.imag, eigenvalues.real))]
print("Coexistence steady state: " + ", ".join(
    f"{s} = {x:.3f}" for s, x in zip(species, x_star)))
print("Eigenvalues of the Jacobian: " + ", ".join(
    f"{e.real:.3f}{e.imag:+.3f}i" for e in eigenvalues)
      + "  (all real parts negative: stable)")

# Three experiments, sampled every half day for 10 days, with 5% noise
t = np.arange(0, 10.01, 0.5)
starts = [[0.05, 0.05, 0.05], [0.5, 0.02, 0.1], [0.02, 0.6, 0.3]]
u = sbp.uniform(5, 2 * len(starts) * len(t) * 3)
gauss = (np.sqrt(-2 * np.log(u[0::2])) * np.cos(2 * np.pi * u[1::2]))
gauss = gauss.reshape(len(starts), len(t), 3)
data = [simulate(x0, t) * (1 + 0.05 * gauss[k]) for k, x0 in enumerate(starts)]

# Regression: (ln x(t + dt) - ln x(t)) / dt  =  r + A x(t)
rows, targets = [], []
for x in data:
    rows.append(np.c_[np.ones(len(t) - 1), x[:-1]])
    targets.append(np.diff(np.log(x), axis=0) / np.diff(t)[:, None])
X, Y = np.vstack(rows), np.vstack(targets)
theta = np.linalg.lstsq(X, Y, rcond=None)[0]     # one column per species
r_hat, A_hat = theta[0], theta[1:].T
print("Estimated growth rates:  " + ", ".join(f"{x:.2f}" for x in r_hat)
      + "   (true " + ", ".join(f"{x:.2f}" for x in r) + ")")
print("Estimated interactions:")
for i, s in enumerate(species):
    print(f"  {s}: " + "  ".join(f"{a:+.2f}" for a in A_hat[i])
          + "   (true " + "  ".join(f"{a:+.2f}" for a in A[i]) + ")")

# --- Plot: data, and the fitted model's predictions
sbp.style()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
tt = np.linspace(0, 10, 201)
colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for k, ax in enumerate(axes):
    fitted = solve_ivp(lambda _, x: x * (r_hat + A_hat @ x), [0, 10],
                       data[k][0], t_eval=tt, rtol=1e-8).y.T
    for i in range(3):
        ax.plot(t, data[k][:, i], "o", color=colours[i], ms=3)
        ax.plot(tt, fitted[:, i], color=colours[i], label=species[i])
    ax.set_xlabel("Time (days)")
    ax.set_title(f"Experiment {k + 1}")
axes[0].set_ylabel("Abundance")
axes[0].legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-glv")

sbp.save_results(__file__, {
    "steady_state": x_star, "eigenvalues_real": eigenvalues.real,
    "eigenvalues_imag": eigenvalues.imag, "r_hat": r_hat, "A_hat": A_hat,
})
sbp.show()
