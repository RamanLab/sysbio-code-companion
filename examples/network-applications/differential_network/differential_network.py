"""Differential co-expression: how gene-gene correlations change between
two conditions.

The data are RNA-seq profiles from PRECISE-1K (Lamoureux et al., 2023) of
wild-type E. coli grown on glucose (200 samples) or glycerol (108
samples), for 44 genes of central carbon and glycerol metabolism. For
each condition we compute the Pearson correlation of every pair of genes
(a co-expression network), and test whether the correlation differs
between the conditions, using Fisher's z-transformation (book §4.3):
    z = (atanh r1 - atanh r2) / sqrt(1/(n1 - 3) + 1/(n2 - 3)).
A pair is differentially co-expressed if |z| exceeds the Bonferroni
threshold for all 946 pairs (overall significance 0.05). The genes with
the most differential partners are the most "rewired".
"""
import numpy as np
import pandas as pd
from scipy.special import erfcinv
import matplotlib.pyplot as plt
import sbp

expr = pd.read_csv(sbp.DATA / "precise1k_carbon_log_tpm.csv", index_col=0)
samples = pd.read_csv(sbp.DATA / "precise1k_carbon_samples.csv")
genes = list(expr.index)
g = len(genes)

r, n = {}, {}
for carbon in ("glucose", "glycerol"):
    columns = samples.loc[samples["carbon_source"] == carbon, "sample"]
    r[carbon] = np.corrcoef(expr[columns].to_numpy())
    n[carbon] = len(columns)

# Fisher's z (the diagonal, r = 1, is not used, so we set it to 0)
fisher = {c: np.arctanh(m - np.diag(np.diag(m))) for c, m in r.items()}
z = ((fisher["glucose"] - fisher["glycerol"])
     / np.sqrt(1 / (n["glucose"] - 3) + 1 / (n["glycerol"] - 3)))
pairs = g * (g - 1) // 2
threshold = np.sqrt(2) * erfcinv(0.05 / pairs)   # two-sided, Bonferroni
upper = np.triu(np.ones((g, g), dtype=bool), 1)
differential = (np.abs(z) > threshold) & upper
rewired = (differential | differential.T).sum(axis=1)

order = np.argsort(-np.abs(z[upper]), kind="stable")
i, j = np.nonzero(upper)
top = [(genes[i[k]], genes[j[k]], r["glucose"][i[k], j[k]],
        r["glycerol"][i[k], j[k]]) for k in order[:8]]
most = np.lexsort((genes, -rewired))[:8]           # ties: alphabetical

print(f"{pairs} gene pairs; |z| > {threshold:.2f} for {differential.sum()} "
      "of them")
print("Largest changes in correlation (glucose -> glycerol):")
for a, b, r1, r2 in top:
    print(f"  {a:5s} {b:5s}  {r1:+.2f} -> {r2:+.2f}")
print("Most rewired genes: " + ", ".join(
    f"{genes[k]} ({rewired[k]})" for k in most))

# --- Plot: the two correlation matrices
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, carbon in zip(axes, ("glucose", "glycerol")):
    im = ax.imshow(r[carbon], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(g), genes, rotation=90, fontsize=6)
    ax.set_yticks(range(g), genes, fontsize=6)
    ax.set_title(f"{carbon} ({n[carbon]} samples)")
fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
sbp.save_figure(fig, __file__, "fig-differential-network")

sbp.save_results(__file__, {
    "genes": genes, "n": n, "threshold": threshold,
    "r_glucose": r["glucose"], "r_glycerol": r["glycerol"],
    "differential": int(differential.sum()), "rewired": rewired,
    "most_rewired": [genes[k] for k in most],
})
sbp.show()
