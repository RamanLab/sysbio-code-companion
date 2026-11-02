"""Scale-free networks by preferential attachment (Barabási and Albert).

Nodes are added one at a time; each new node links to m existing nodes,
chosen with probability proportional to their degree. The resulting
degree distribution has a heavy tail, P(k) ~ k^-3, with a few highly
connected hubs. We compare it with an Erdős–Rényi network of the same
size and mean degree.
"""
import numpy as np
import matplotlib.pyplot as plt
import sbp

n, m = 2000, 2
reps = 10
rng = np.random.default_rng(1)


def barabasi_albert(n, m):
    """Edge list of a preferential-attachment network. 'ends' lists every
    edge end, so a node appears in it once per link: picking a uniformly
    random element of 'ends' picks a node with probability proportional
    to its degree."""
    edges = [(i, j) for i in range(m + 1) for j in range(i)]  # a clique
    ends = [v for e in edges for v in e]
    for new in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            targets.add(ends[rng.integers(len(ends))])
        for t in targets:
            edges.append((new, t))
            ends += [new, t]
    return edges


def degrees(edges, n):
    return np.bincount(np.ravel(edges), minlength=n)


k_max = 200
P_ba, P_er = np.zeros(k_max + 1), np.zeros(k_max + 1)
max_deg_ba, max_deg_er = [], []
for _ in range(reps):
    d = degrees(barabasi_albert(n, m), n)
    P_ba += np.bincount(np.minimum(d, k_max), minlength=k_max + 1) / n / reps
    max_deg_ba.append(d.max())
    # Erdős–Rényi with the same expected mean degree
    mean_k = d.mean()
    upper = np.triu(rng.random((n, n)) < mean_k / (n - 1), 1)
    d = (upper | upper.T).sum(axis=1)
    P_er += np.bincount(np.minimum(d, k_max), minlength=k_max + 1) / n / reps
    max_deg_er.append(d.max())

k = np.arange(k_max + 1)
print(f"Mean degree {mean_k:.4f}")
print(f"Largest degree, averaged over {reps} networks: preferential "
      f"attachment {np.mean(max_deg_ba):.1f}, Erdős–Rényi "
      f"{np.mean(max_deg_er):.1f}")
print(f"Fraction of nodes with degree >= 10: {P_ba[10:].sum():.4f} "
      f"(theory for large n: {m * (m + 1) / (10 * 11):.4f}); "
      f"Erdős–Rényi {P_er[10:].sum():.4f}")

# --- Plot. Right: the fraction of nodes with degree at least k, which is
# much less noisy in the tail than P(k) itself
ccdf_ba, ccdf_er = P_ba[::-1].cumsum()[::-1], P_er[::-1].cumsum()[::-1]
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(k, P_ba, "o", color="#1f77b4", ms=4, label="preferential attachment")
ax1.plot(k, P_er, "s", color="#ff7f0e", ms=4, label="Erdős–Rényi")
ax1.set_xlim(-0.5, 30)
ax1.set_xlabel("Degree k")
ax1.set_ylabel("Fraction of nodes with degree k")
ax1.set_title("Degree distribution")
ax1.legend(frameon=False)
ax2.loglog(k[1:][ccdf_ba[1:] > 0], ccdf_ba[1:][ccdf_ba[1:] > 0], "o",
           color="#1f77b4", ms=4, label="preferential attachment")
ax2.loglog(k[1:][ccdf_er[1:] > 0], ccdf_er[1:][ccdf_er[1:] > 0], "s",
           color="#ff7f0e", ms=4, label="Erdős–Rényi")
ax2.loglog(k[m:], m * (m + 1) / (k[m:] * (k[m:] + 1.0)), "-",
           color="#999999", label="theory, $m(m+1)/k(k+1)$")
ax2.set_ylim(1e-5, 1.5)
ax2.set_xlabel("Degree k")
ax2.set_ylabel("Fraction of nodes with degree $\\geq$ k")
ax2.set_title("Tail, on logarithmic axes")
ax2.legend(frameon=False, loc="lower left")
sbp.save_figure(fig, __file__, "fig-scale-free")

sbp.save_results(__file__, {
    "mean_degree": mean_k,
    "ba": {"P2": P_ba[2], "P3": P_ba[3], "P_ge_10": P_ba[10:].sum(),
           "mean_max_degree": np.mean(max_deg_ba)},
    "er": {"mean_max_degree": np.mean(max_deg_er)},
    "er_P_ge_10": P_er[10:].sum(),
    "P_ba": P_ba[:51], "P_er": P_er[:51],
})
sbp.show()
