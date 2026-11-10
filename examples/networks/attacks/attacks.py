"""Robustness of a network to random failures and targeted attacks.

We remove an increasing fraction of the nodes of the E. coli STRING
network (combined score at least 700), and follow the size of the
largest connected component (book §3.5.1). Nodes are removed either in
random order (failures) or in decreasing order of their degree in the
intact network (an attack on the hubs; ties broken by name). The random
order comes from sbp.uniform, so that it is the same in MATLAB and
Python. For comparison, we do the same for an Erdős–Rényi random graph
with the same numbers of nodes and edges, also drawn with sbp.uniform.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sbp

table = pd.read_csv(sbp.DATA / "ecoli_string_ppi.csv")[["protein1", "protein2"]]
names = np.unique(table.to_numpy())
index = {name: i for i, name in enumerate(names)}
edges = np.array([[index[a], index[b]] for a, b in table.to_numpy()])
n, m = len(names), len(edges)


def er_graph(n, m, seed):
    """m distinct random edges among n nodes (Erdős–Rényi G(n, m))."""
    u = sbp.uniform(seed, 4 * m)
    chosen, k = set(), 0
    while len(chosen) < m:
        a, b = int(u[k] * n), int(u[k + 1] * n)
        k += 2
        if a != b:
            chosen.add((min(a, b), max(a, b)))
    return np.array(sorted(chosen))


def giant_fraction(edges, n, removed):
    """Size of the largest component, as a fraction of all n nodes."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    G.remove_nodes_from(removed)
    return max((len(c) for c in nx.connected_components(G)), default=0) / n


steps = np.arange(20)                       # remove 0%, 5%, ..., 95%
fractions = steps / 20
counts = steps * n // 20                    # whole numbers of nodes
results = {"fractions": fractions}
for label, e in (("string", edges), ("random_graph", er_graph(n, m, 7))):
    degree = np.bincount(e.ravel(), minlength=n)
    by_degree = np.lexsort((np.arange(n), -degree))   # ties: by name
    random_order = np.argsort(sbp.uniform(2024, n))
    curves = {}
    for strategy, order in (("attack", by_degree), ("failure", random_order)):
        curves[strategy] = [giant_fraction(e, n, order[:c]) for c in counts]
    results[label] = curves
    print(f"{label}: largest component after removing 20%, 50%, 70% of nodes:"
          f" failures {curves['failure'][4]:.2f}, {curves['failure'][10]:.2f},"
          f" {curves['failure'][14]:.2f}; attack {curves['attack'][4]:.2f}, "
          f"{curves['attack'][10]:.2f}, {curves['attack'][14]:.2f}")

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(6, 4.5))
styles = {("string", "failure"): ("#1f77b4", "-"),
          ("string", "attack"): ("#d62728", "-"),
          ("random_graph", "failure"): ("#1f77b4", ":"),
          ("random_graph", "attack"): ("#d62728", ":")}
for (label, strategy), (colour, ls) in styles.items():
    ax.plot(fractions, results[label][strategy], ls, color=colour, marker="o",
            ms=3, label=f"{'STRING' if label == 'string' else 'random graph'}, "
                        f"{strategy}")
ax.set_xlabel("Fraction of nodes removed")
ax.set_ylabel("Largest component (fraction of nodes)")
ax.legend(frameon=False, fontsize=9)
sbp.save_figure(fig, __file__, "fig-attacks")

sbp.save_results(__file__, results)
sbp.show()
