"""Are central proteins more often essential?

The E. coli K-12 protein interaction network from STRING (v12.0, links
with a combined score of at least 700), and the genes that NetGenes
predicts to be essential. We compute three centralities for each protein,
see how the fraction of essential genes changes with degree, and measure
how well each centrality alone separates essential from other genes.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sbp

edges = np.loadtxt(sbp.DATA / "ecoli_string_ppi.csv", delimiter=",",
                   skiprows=1, usecols=(0, 1), dtype=str)
names = dict(np.loadtxt(sbp.DATA / "ecoli_string_proteins.csv",
                        delimiter=",", skiprows=1, usecols=(0, 1), dtype=str))
essential = set(np.loadtxt(sbp.DATA / "ecoli_netgenes_essential.csv",
                           delimiter=",", skiprows=1, usecols=0, dtype=str))

G = nx.Graph()
G.add_edges_from(edges)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
nodes = sorted(G.nodes)                 # a fixed order, as in MATLAB
is_ess = np.array([v in essential for v in nodes])
print(f"Largest connected component: {len(nodes)} proteins, "
      f"{G.number_of_edges()} interactions, {is_ess.sum()} essential")

cc = nx.closeness_centrality(G)
ec = nx.eigenvector_centrality_numpy(G)
centrality = {
    "degree": np.array([G.degree(v) for v in nodes], dtype=float),
    "closeness": np.array([cc[v] for v in nodes]),
    "eigenvector": np.array([ec[v] for v in nodes]),
}


def auc(score, positive):
    """Area under the ROC curve: the probability that a randomly chosen
    positive scores higher than a randomly chosen negative (ties count
    one half), from the ranks of the scores."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):                     # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


aucs = {name: auc(c, is_ess) for name, c in centrality.items()}
for name, a in aucs.items():
    print(f"AUC of {name}: {a:.4f}")

# Fraction essential in ten equal groups, from least to most connected
order = np.argsort(centrality["degree"], kind="stable")
groups = np.array_split(order, 10)
frac_ess = np.array([is_ess[g].mean() for g in groups])
mean_deg = np.array([centrality["degree"][g].mean() for g in groups])
print("Fraction essential by degree group:", np.round(frac_ess, 3))

top = np.argsort(-centrality["degree"], kind="stable")[:10]
print("Ten most connected proteins:")
for i in top:
    print(f"  {names[nodes[i]]:6s} degree {centrality['degree'][i]:.0f}"
          f"{'  essential' if is_ess[i] else ''}")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(mean_deg, frac_ess, "o-", color="#1f77b4")
ax1.axhline(is_ess.mean(), color="#999999", ls=":", lw=1)
ax1.text(1.1, is_ess.mean() + 0.01, "all proteins", color="#666666",
         fontsize=9)
ax1.set_xscale("log")
ax1.set_xticks([1, 2, 5, 10, 20, 50, 100], labels=["1", "2", "5", "10",
                                                  "20", "50", "100"])
ax1.minorticks_off()
ax1.set_xlabel("Mean degree of the group")
ax1.set_ylabel("Fraction essential")
ax1.set_title("Ten groups of proteins, by degree")
for name, colour in [("degree", "#1f77b4"), ("closeness", "#ff7f0e"),
                     ("eigenvector", "#2ca02c")]:
    c = centrality[name]
    thresholds = np.sort(np.unique(c))[::-1]
    tpr = [(c[is_ess] >= t).mean() for t in thresholds]
    fpr = [(c[~is_ess] >= t).mean() for t in thresholds]
    ax2.plot([0] + fpr, [0] + tpr, color=colour,
             label=f"{name} (AUC {aucs[name]:.2f})")
ax2.plot([0, 1], [0, 1], ":", color="#999999")
ax2.set_xlabel("Fraction of other genes above threshold")
ax2.set_ylabel("Fraction of essential genes above threshold")
ax2.set_title("ROC curves")
ax2.legend(frameon=False, loc="lower right")
sbp.save_figure(fig, __file__, "fig-centrality-lethality")

sbp.save_results(__file__, {
    "n_nodes": len(nodes), "n_edges": G.number_of_edges(),
    "n_essential": int(is_ess.sum()), "auc": aucs,
    "fraction_essential_by_degree": frac_ess,
    "top_degree": [names[nodes[i]] for i in top],
})
sbp.show()
