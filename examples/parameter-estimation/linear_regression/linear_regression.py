"""Linearising the Michaelis-Menten equation, and why we should not.

Three transformations turn v = vmax S / (KM + S) into a straight line
(book §6.2.1):
  Lineweaver-Burk  1/v vs 1/S:   1/v = (KM/vmax)(1/S) + 1/vmax
  Eadie-Hofstee    v vs v/S:     v = -KM (v/S) + vmax
  Hanes-Woolf      S/v vs S:     S/v = S/vmax + KM/vmax
We estimate vmax and KM from each line, and by non-linear least squares,
first for the data of Exercise 6.5, and then for 200 simulated datasets
with known parameters (vmax = 2.5 mM/s, KM = 0.8 mM, noise of standard
deviation 0.1 mM/s), to see how the transformations distort the errors.
"""
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import sbp

methods = ["Lineweaver-Burk", "Eadie-Hofstee", "Hanes-Woolf", "non-linear"]


def estimate(S, v):
    """vmax and KM by each of the four methods (one row per method)."""
    slope, intercept = np.polyfit(1 / S, 1 / v, 1)
    lineweaver_burk = [1 / intercept, slope / intercept]
    slope, intercept = np.polyfit(v / S, v, 1)
    eadie_hofstee = [intercept, -slope]
    slope, intercept = np.polyfit(S, S / v, 1)
    hanes_woolf = [1 / slope, intercept / slope]
    # Non-linear least squares, started from the Hanes-Woolf estimate
    sol = least_squares(lambda p: p[0] * S / (p[1] + S) - v, hanes_woolf,
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return np.array([lineweaver_burk, eadie_hofstee, hanes_woolf, sol.x])


# Exercise 6.5 (the column headers in the book are swapped: S is the
# first row, v the second)
S = np.array([0.15, 0.35, 0.65, 1.2, 2.0, 3.0, 4.0, 5.0, 6.0])
v = np.array([0.35, 0.70, 1.26, 1.62, 1.5, 2.21, 2.167, 2.33, 2.12])
exercise = estimate(S, v)
print("Exercise 6.5:")
for name, (vm, km) in zip(methods, exercise):
    sse = np.sum((vm * S / (km + S) - v) ** 2)
    print(f"  {name:16s} vmax = {vm:.3f}, KM = {km:.3f}, SSE = {sse:.4f}")

# 200 simulated datasets with known parameters
replicates = np.loadtxt(sbp.DATA / "mm_replicates.csv", delimiter=",",
                        skiprows=1)
true = np.array([2.5, 0.8])
estimates = np.array([estimate(S, r) for r in replicates])  # 200 x 4 x 2
median = np.median(estimates, axis=0)
error = np.median(np.abs(estimates - true) / true, axis=0)   # relative
print("200 simulated datasets (vmax = 2.5, KM = 0.8):")
for k, name in enumerate(methods):
    print(f"  {name:16s} median vmax = {median[k, 0]:.3f}, KM = "
          f"{median[k, 1]:.3f}; median error {100 * error[k, 0]:.1f}%, "
          f"{100 * error[k, 1]:.1f}%")

# --- Plot: the Lineweaver-Burk line for Exercise 6.5, and the spread of
# the KM estimates across the simulated datasets
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x = np.linspace(0, 1 / S.min(), 50)
vm, km = exercise[0]
ax1.plot(1 / S, 1 / v, "o", color="black")
ax1.plot(x, km / vm * x + 1 / vm, color="#d62728")
ax1.set_xlabel("1/S (1/mM)")
ax1.set_ylabel("1/v (s/mM)")
ax1.set_title("Lineweaver-Burk plot, Exercise 6.5")
ax2.boxplot([estimates[:, k, 1] for k in range(4)], showfliers=False)
ax2.set_xticks(range(1, 5), ["LB", "EH", "HW", "non-linear"])
ax2.axhline(true[1], ls=":", color="black")
ax2.set_ylabel("Estimated KM (mM)")
ax2.set_title("KM from 200 simulated datasets")
sbp.save_figure(fig, __file__, "fig-linear-regression")

sbp.save_results(__file__, {
    "methods": methods, "exercise": exercise,
    "median": median, "median_error": error, "estimates": estimates,
})
sbp.show()
