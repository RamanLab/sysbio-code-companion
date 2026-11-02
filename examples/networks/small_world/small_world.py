"""Small-world networks (Watts and Strogatz, 1998).

Start from a ring lattice of n nodes, each linked to its k nearest
neighbours on either side, and rewire each edge with probability p. A
little rewiring makes paths short, as in a random network, while keeping
the high clustering of the lattice. We also compare with Erdős–Rényi
random networks with the same numbers of nodes and edges.
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
import sbp

n, k = 500, 5                     # 500 nodes, 2k = 10 neighbours each
probs = np.logspace(-4, 0, 13)    # rewiring probabilities
reps = 10                         # networks averaged for each p
rng = np.random.default_rng(1)


def ring_lattice(n, k):
    A = np.zeros((n, n), dtype=bool)
    for j in range(1, k + 1):
        idx = np.arange(n)
        A[idx, (idx + j) % n] = A[(idx + j) % n, idx] = True
    return A


def watts_strogatz(n, k, p):
    """Rewire each lattice edge (i, i+j) with probability p, moving its far
    end to a random node, avoiding self-loops and duplicate edges."""
    A = ring_lattice(n, k)
    for j in range(1, k + 1):
        for i in range(n):
            old = (i + j) % n
            if A[i, old] and rng.random() < p:
                new = rng.integers(n)          # draw until the new end is
                while new == i or A[i, new]:   # neither i nor a neighbour
                    new = rng.integers(n)
                A[i, old] = A[old, i] = False
                A[i, new] = A[new, i] = True
    return A


def erdos_renyi(n, p):
    upper = np.triu(rng.random((n, n)) < p, 1)
    return upper | upper.T


def clustering_and_path_length(A):
    """Mean clustering coefficient, and mean shortest-path length over all
    pairs of nodes that are connected by some path."""
    A = A.astype(float)
    deg = A.sum(axis=1)
    tri = ((A @ A) * A).sum(axis=1) / 2
    pairs = deg * (deg - 1) / 2
    C = np.divide(tri, pairs, out=np.zeros(len(A)), where=pairs > 0).mean()
    D = shortest_path(A, unweighted=True, directed=False)
    d = D[np.triu_indices(len(A), k=1)]
    return C, d[np.isfinite(d)].mean()


C0, L0 = clustering_and_path_length(ring_lattice(n, k))
C, L = np.zeros((reps, probs.size)), np.zeros((reps, probs.size))
for r in range(reps):
    for i, p in enumerate(probs):
        C[r, i], L[r, i] = clustering_and_path_length(watts_strogatz(n, k, p))
C_ratio, L_ratio = C.mean(axis=0) / C0, L.mean(axis=0) / L0

p_er = n * k / (n * (n - 1) / 2)  # same expected number of edges, n*k
er = np.array([clustering_and_path_length(erdos_renyi(n, p_er))
               for _ in range(reps)]).mean(axis=0)

print(f"Ring lattice: C = {C0:.4f}, L = {L0:.4f}")
for p, c, l in zip(probs, C_ratio, L_ratio):
    print(f"  p = {p:8.4f}: C(p)/C(0) = {c:.3f}, L(p)/L(0) = {l:.3f}")
print(f"Erdős–Rényi, same n and m: C = {er[0]:.4f}, L = {er[1]:.4f}")

# --- Plot
sbp.style()
fig, ax = plt.subplots()
ax.semilogx(probs, C_ratio, "o-", color="#1f77b4", label="C(p) / C(0)")
ax.semilogx(probs, L_ratio, "s-", color="#ff7f0e", label="L(p) / L(0)")
ax.set_xlabel("Rewiring probability p")
ax.set_ylabel("Relative to the ring lattice")
ax.set_title(f"Watts–Strogatz networks, n = {n}, {2 * k} neighbours")
ax.legend(frameon=False, loc="lower left")
sbp.save_figure(fig, __file__, "fig-small-world")

at = {f"p_{p:g}".replace(".", "_"): i
      for i, p in enumerate(probs) if p in (0.1, 1.0)}
sbp.save_results(__file__, {
    "lattice": {"C": C0, "L": L0},
    "ws": {name: {"C_ratio": C_ratio[i], "L_ratio": L_ratio[i]}
           for name, i in at.items()},
    "er": {"C": er[0], "L": er[1]},
    "probs": probs, "C_ratio": C_ratio, "L_ratio": L_ratio,
})
sbp.show()
