"""Degree assortativity, and comparing a network with randomised versions.

Assortativity (book §3.1.3) is the correlation between the degrees of the
two nodes at either end of an edge: positive when hubs link to hubs,
negative when hubs link to sparsely connected nodes. To see whether a
value says something about a network, rather than about its degrees
alone, we compare it with randomised networks that keep every node's
degree (book §3.4.1). Each randomisation step picks two edges, a-b and
c-d, and rewires them to a-d and c-b, unless that would create a
self-loop or an edge that already exists. The random choices come from
sbp.uniform, so that MATLAB and Python make exactly the same swaps.

Two networks: Zachary's karate club, and the E. coli STRING network
(combined score at least 700).
"""
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import sbp


def load(edge_file, columns):
    """Edges as pairs of node indices; nodes numbered in sorted order."""
    table = pd.read_csv(sbp.DATA / edge_file)[columns].astype(str)
    names = np.unique(table.to_numpy())
    index = {name: i for i, name in enumerate(names)}
    edges = np.array([[index[a], index[b]] for a, b in table.to_numpy()])
    return edges, len(names)


def degree_correlation(edges, n):
    """Newman's degree correlation coefficient."""
    k = np.bincount(edges.ravel(), minlength=n).astype(float)
    j, l = k[edges[:, 0]], k[edges[:, 1]]
    mean = np.mean((j + l) / 2)
    return ((np.mean(j * l) - mean ** 2)
            / (np.mean((j ** 2 + l ** 2) / 2) - mean ** 2))


def clustering(edges, n):
    """Mean clustering coefficient (nodes of degree < 2 count as 0)."""
    A = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                          shape=(n, n)).tocsr()
    A = A + A.T
    k = np.asarray(A.sum(axis=1)).ravel()
    triangles = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() / 2
    c = np.zeros(n)
    ok = k > 1
    c[ok] = 2 * triangles[ok] / (k[ok] * (k[ok] - 1))
    return c.mean()


def randomise(edges, n, seed, swaps_per_edge=10):
    """Degree-preserving randomisation by double edge swaps."""
    edges = edges.copy()
    m = len(edges)
    A = np.zeros((n, n), dtype=bool)
    A[edges[:, 0], edges[:, 1]] = A[edges[:, 1], edges[:, 0]] = True
    u = sbp.uniform(seed, 3 * swaps_per_edge * m)
    for t in range(swaps_per_edge * m):
        e1, e2 = int(u[3 * t] * m), int(u[3 * t + 1] * m)
        a, b = edges[e1]
        c, d = edges[e2]
        if u[3 * t + 2] < 0.5:                  # either orientation of c-d
            c, d = d, c
        if len({a, b, c, d}) < 4 or A[a, d] or A[c, b]:
            continue                            # self-loop or existing edge
        A[a, b] = A[b, a] = A[c, d] = A[d, c] = False
        A[a, d] = A[d, a] = A[c, b] = A[b, c] = True
        edges[e1], edges[e2] = (a, d), (c, b)
    return edges


networks = {
    "karate": (load("karate_edges.csv", ["member1", "member2"]), 20),
    "string": (load("ecoli_string_ppi.csv", ["protein1", "protein2"]), 3),
}
results = {}
for name, ((edges, n), copies) in networks.items():
    r_rand, c_rand = [], []
    for seed in range(1, copies + 1):
        shuffled = randomise(edges, n, seed=1000 + seed)
        assert np.array_equal(np.bincount(shuffled.ravel(), minlength=n),
                              np.bincount(edges.ravel(), minlength=n))
        r_rand.append(degree_correlation(shuffled, n))
        c_rand.append(clustering(shuffled, n))
    results[name] = {"nodes": n, "edges": len(edges),
                     "assortativity": degree_correlation(edges, n),
                     "clustering": clustering(edges, n),
                     "assortativity_random": r_rand,
                     "clustering_random": c_rand}
    r = results[name]
    print(f"{name}: {n} nodes, {len(edges)} edges")
    print(f"  assortativity {r['assortativity']:+.3f}; randomised "
          f"{np.mean(r_rand):+.3f} (range {min(r_rand):+.3f} to "
          f"{max(r_rand):+.3f}, {copies} networks)")
    print(f"  mean clustering {r['clustering']:.3f}; randomised "
          f"{np.mean(c_rand):.3f}")

# --- Plot
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (key, label) in zip(axes, [("assortativity", "Assortativity r"),
                                   ("clustering", "Mean clustering")]):
    for x, name in enumerate(results):
        r = results[name]
        ax.plot([x], [r[key]], "o", color="#d62728", ms=9,
                label="network" if x == 0 else None)
        ax.plot(np.full(len(r[f"{key}_random"]), x + 0.15),
                r[f"{key}_random"], "_", color="#1f77b4", ms=12,
                label="randomised" if x == 0 else None)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks([0, 1], ["karate club", "E. coli STRING"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(label)
axes[0].legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-assortativity")

sbp.save_results(__file__, results)
sbp.show()
