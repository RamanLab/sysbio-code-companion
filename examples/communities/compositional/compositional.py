"""Association networks from relative abundances: beware of compositionality.

Microbial association networks (book §11.1.1.1) are built from the
co-occurrence of taxa across many samples. Sequencing, however, gives
relative abundances: the reads of each taxon as a fraction of the total.
Fractions must add up to one, so if one taxon rises, the others must
fall, even if they are entirely unaffected. Here, we simulate 10 taxa
whose absolute abundances vary *independently* across 200 samples
(log-normally, with one dominant taxon), and compare the correlations
between taxa computed from
  * the absolute abundances (no true associations at all),
  * the relative abundances, and
  * the centred log-ratio (CLR) transform of the relative abundances,
    log(x_i) - mean_j log(x_j), used by SparCC-like methods.
An "edge" is a pair of taxa with |r| > 0.2 (p < 0.005 for 200 samples).
The Gaussian numbers are made from sbp.uniform (Box-Muller), so they are
identical in MATLAB and Python.
"""
import numpy as np
import matplotlib.pyplot as plt
import sbp

taxa, samples = 10, 200
mean_log = np.array([4.0] + [1.0] * (taxa - 1))   # taxon 1 dominates
sd_log = np.full(taxa, 1.0)
u = sbp.uniform(77, 2 * samples * taxa)
z = (np.sqrt(-2 * np.log(u[0::2])) * np.cos(2 * np.pi * u[1::2]))
z = z.reshape(samples, taxa)
absolute = np.exp(mean_log + sd_log * z)            # samples x taxa
relative = absolute / absolute.sum(axis=1, keepdims=True)
log_rel = np.log(relative)
clr = log_rel - log_rel.mean(axis=1, keepdims=True)

upper = np.triu(np.ones((taxa, taxa), dtype=bool), 1)
results = {}
for name, data in (("absolute", absolute), ("relative", relative),
                   ("clr", clr)):
    r = np.corrcoef(data.T)[upper]
    results[name] = {"mean_r": r.mean(), "edges": int(np.sum(np.abs(r) > 0.2)),
                     "dominant_mean_r": np.corrcoef(data.T)[0, 1:].mean()}
    print(f"{name:9s} mean r = {r.mean():+.3f}; edges (|r| > 0.2): "
          f"{results[name]['edges']:2d} of {upper.sum()}; mean r with the "
          f"dominant taxon = {results[name]['dominant_mean_r']:+.3f}")

# --- Plot: correlation matrices
sbp.style()
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = {"absolute": "Absolute abundances", "relative": "Relative abundances",
          "clr": "CLR-transformed"}
for ax, (name, data) in zip(axes, (("absolute", absolute),
                                   ("relative", relative), ("clr", clr))):
    R = np.corrcoef(data.T)
    np.fill_diagonal(R, np.nan)
    im = ax.imshow(R, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(taxa), range(1, taxa + 1))
    ax.set_yticks(range(taxa), range(1, taxa + 1))
    ax.set_title(f"{titles[name]}\n{results[name]['edges']} edges")
fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
sbp.save_figure(fig, __file__, "fig-compositional")

sbp.save_results(__file__, results)
sbp.show()
