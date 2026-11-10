"""Communities by modularity maximisation: Newman's spectral method.

Modularity (book §3.3.1) compares the edges within communities with the
number expected at random, for the same degrees:
    Q = (1/2m) sum_ij [A_ij - k_i k_j / 2m] delta(c_i, c_j).
The matrix B = A - k k'/2m is the modularity matrix. Newman's method
splits the network in two by the signs of the leading eigenvector of B,
and then splits each part again, using the "generalised" modularity
matrix of that part, until no split increases Q (Newman, 2006, PNAS).
Unlike many other methods, it involves no random choices.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sbp

edges = pd.read_csv(sbp.DATA / "karate_edges.csv").to_numpy() - 1
faction = pd.read_csv(sbp.DATA / "karate_factions.csv")["faction"].to_numpy()
n = len(faction)
A = np.zeros((n, n))
for i, j in edges:
    A[i, j] = A[j, i] = 1
k = A.sum(axis=1)
two_m = k.sum()
B = A - np.outer(k, k) / two_m


def modularity(labels):
    same = labels[:, None] == labels[None, :]
    return np.sum(B * same) / two_m


def split(group):
    """Split a group of nodes in two, or return None if no split
    increases the modularity."""
    Bg = B[np.ix_(group, group)]
    Bg = Bg - np.diag(Bg.sum(axis=1))         # generalised modularity matrix
    values, vectors = np.linalg.eigh(Bg)
    if values[-1] < 1e-10:
        return None
    s = np.where(vectors[:, -1] >= 0, 1, -1)
    if s @ Bg @ s / (2 * two_m) < 1e-10:      # the gain in Q
        return None
    return group[s > 0], group[s < 0]


communities, queue = [], [np.arange(n)]
first_split = None
while queue:
    group = queue.pop(0)
    parts = split(group)
    if parts is None:
        communities.append(group)
    else:
        if first_split is None:
            first_split = parts
        queue.extend(parts)

# Label communities 1, 2, ... in order of their lowest-numbered member
communities.sort(key=lambda g: g.min())
labels = np.zeros(n, dtype=int)
for c, g in enumerate(communities, start=1):
    labels[g] = c
halves = np.zeros(n, dtype=int)
halves[first_split[0]], halves[first_split[1]] = 1, 2
if halves[0] != 1:
    halves = 3 - halves                         # member 1 in the first half

# Check against NetworkX's own modularity function
G = nx.from_numpy_array(A)
assert abs(nx.community.modularity(G, [set(g) for g in communities])
           - modularity(labels)) < 1e-12

first = np.where(halves == 1, "Mr. Hi", "Officer")
print(f"First split: Q = {modularity(halves):.4f}; it places "
      f"{np.sum(first == faction)} of {n} members on the side they joined "
      f"(misplaced: {', '.join(str(x) for x in np.flatnonzero(first != faction) + 1)})")
print(f"Final: {len(communities)} communities, Q = {modularity(labels):.4f}")
for c, g in enumerate(communities, start=1):
    print(f"  community {c}: members {', '.join(str(x) for x in g + 1)}")

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(6, 5))
pos = nx.spring_layout(G, seed=1)
palette = ["#1f77b4", "#aec7e8", "#d62728", "#ff9896", "#2ca02c", "#98df8a"]
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200,
                       node_color=[palette[c - 1] for c in labels],
                       node_shape="o")
nx.draw_networkx_labels(G, pos, {i: i + 1 for i in range(n)}, ax=ax,
                        font_size=7)
ax.set_title(f"{len(communities)} communities, Q = {modularity(labels):.3f}")
ax.axis("off")
sbp.save_figure(fig, __file__, "fig-modularity")

sbp.save_results(__file__, {
    "halves": halves, "q_halves": modularity(halves),
    "communities": labels, "q": modularity(labels),
})
sbp.show()
