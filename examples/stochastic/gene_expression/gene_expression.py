"""Noise in gene expression: transcription and translation.

    0 -> M       at rate km           (transcription)
    M -> 0       at rate gm * M       (mRNA degradation)
    M -> M + P   at rate kp * M       (translation)
    P -> 0       at rate gp * P       (protein degradation)

Two genes make the same mean number of proteins, 50. Gene A is
transcribed often and each mRNA is translated a few times; gene B is
transcribed rarely and each mRNA is translated many times, so its
proteins are made in bursts. Theory (for example, Thattai and van
Oudenaarden, 2001) gives, at steady state,
    mean P = km*kp/(gm*gp),   Fano factor = var/mean = 1 + kp/(gm + gp).
"""
import numpy as np
import matplotlib.pyplot as plt
import sbp
from sbp.ssa import gillespie

gm, gp = 1.0, 0.1                       # per minute
genes = {"A": {"km": 2.0, "kp": 2.5},   # frequent transcription
         "B": {"km": 0.2, "kp": 25.0}}  # rare transcription, big bursts
t_grid = np.arange(0, 40001.0)          # one long run, sampled each minute
burn_in = 200                           # discard the approach to steady state
rng = np.random.default_rng(1)

# Changes in (M, P) for each of the four reactions
changes = [[1, 0], [-1, 0], [0, 1], [0, -1]]

results = {}
traces = {}
for name, g in genes.items():
    def propensities(x, km=g["km"], kp=g["kp"]):
        M, P = x
        return np.array([km, gm * M, kp * M, gp * P])

    X = gillespie([0, 0], changes, propensities, t_grid, rng)
    P = X[burn_in:, 1]
    theory_mean = g["km"] * g["kp"] / (gm * gp)
    theory_fano = 1 + g["kp"] / (gm + gp)
    results[name] = {"mean": P.mean(), "fano": P.var() / P.mean()}
    traces[name] = X[:, 1]
    print(f"Gene {name}: mean {P.mean():.1f} (theory {theory_mean:g}), "
          f"Fano factor {P.var() / P.mean():.2f} (theory {theory_fano:.2f})")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                               gridspec_kw={"width_ratios": [2, 1]})
window = slice(1000, 1301)
colors = {"A": "#1f77b4", "B": "#ff7f0e"}
for name in genes:
    ax1.plot(t_grid[window] - t_grid[window][0], traces[name][window],
             color=colors[name], lw=1.2, label=f"gene {name}")
    ax2.hist(traces[name][burn_in:], bins=np.arange(0, 251, 5),
             density=True, histtype="step", lw=1.5, color=colors[name],
             label=f"gene {name}")
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("Number of protein molecules")
ax1.set_title("Same mean, different noise")
ax1.legend(frameon=False, loc="upper left")
ax2.set_xlabel("Number of protein molecules")
ax2.set_ylabel("Probability density")
ax2.set_title("Distribution")
ax2.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-gene-expression")

sbp.save_results(__file__, results)
sbp.show()
