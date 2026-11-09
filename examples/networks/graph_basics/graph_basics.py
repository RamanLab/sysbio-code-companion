"""Basic network parameters for a small undirected network.

Ten proteins, A to J, in two tightly knit groups joined through E.
We compute degrees, clustering coefficients, shortest-path lengths and
two centrality measures, using the definitions in the book (Ch. 3).
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sbp

nodes = list("ABCDEFGHIJ")
edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"),
         ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G"), ("F", "H"),
         ("G", "H"), ("G", "I"), ("H", "I"), ("I", "J")]
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
n = G.number_of_nodes()

# Adjacency matrix and degrees (book Eq. 3.1)
A = nx.to_numpy_array(G, nodelist=nodes)
degree = A.sum(axis=1)

# Clustering coefficient (Eq. 3.6): the fraction of pairs of neighbours
# that are themselves connected; (A^3)_ii counts each triangle twice
triangles = np.diag(A @ A @ A) / 2
pairs = degree * (degree - 1) / 2
clustering = np.divide(triangles, pairs, out=np.zeros(n), where=pairs > 0)

# Shortest-path lengths between all pairs of nodes
D = nx.floyd_warshall_numpy(G, nodelist=nodes)
diameter = D.max()                                         # Eq. 3.11
char_path_length = D[np.triu_indices(n, k=1)].mean()      # Eq. 3.12
closeness = (n - 1) / D.sum(axis=1)                        # Eq. 3.14

# Betweenness centrality (Eq. 3.18), counting each pair of nodes once
bc = nx.betweenness_centrality(G, normalized=False)
betweenness = np.array([bc[v] for v in nodes])

print(f"{n} nodes, {G.number_of_edges()} edges, mean degree "
      f"{degree.mean():.2f}")
print(f"mean clustering coefficient {clustering.mean():.4f}, diameter "
      f"{diameter:g}, characteristic path length {char_path_length:.4f}")
print("node  degree  clustering  closeness  betweenness")
for i, v in enumerate(nodes):
    print(f"  {v}   {degree[i]:4g}    {clustering[i]:8.3f}   "
          f"{closeness[i]:8.3f}   {betweenness[i]:9.2f}")

# --- Plot: node size by degree, colour by betweenness, fixed positions
x = [0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
y = [1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0]
sbp.style()
fig, ax = plt.subplots(figsize=(7, 3.2))
idx = {v: i for i, v in enumerate(nodes)}
for u, w in edges:
    ax.plot([x[idx[u]], x[idx[w]]], [y[idx[u]], y[idx[w]]], "-",
            color="#999999", zorder=1)
sc = ax.scatter(x, y, s=150 * degree, c=betweenness, cmap="viridis_r",
                edgecolors="k", zorder=2)
for i, v in enumerate(nodes):
    ax.annotate(v, (x[i], y[i]), xytext=(0, 14 + 2 * degree[i]),
                textcoords="offset points", ha="center")
fig.colorbar(sc, ax=ax, label="Betweenness centrality")
ax.set_title("Node size: degree; colour: betweenness")
ax.set_axis_off()
ax.set_ylim(-0.4, 1.6)
sbp.save_figure(fig, __file__, "fig-graph-basics")

sbp.save_results(__file__, {
    "degree": degree, "clustering": clustering,
    "mean_clustering": clustering.mean(), "diameter": diameter,
    "char_path_length": char_path_length, "closeness": closeness,
    "betweenness": betweenness,
})
sbp.show()
