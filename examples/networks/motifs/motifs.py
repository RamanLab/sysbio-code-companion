"""Network motifs in the E. coli transcriptional regulatory network.

A motif is a small pattern of interactions that occurs far more often in
a real network than in randomised networks with the same degrees (book
§3.4). We count two three-node patterns in the network of transcription
factors and their targets (data/ecoli_trn.csv, from RegulonDB via
PRECISE-1K):
  feed-forward loop (FFL)   X -> Y, Y -> Z and X -> Z
  feedback loop (FBL)       X -> Y, Y -> Z and Z -> X
With A the adjacency matrix (no self-loops), the number of FFLs is
sum(A .* (A A)), and the number of FBLs is trace(A^3) / 3. We compare
the counts with 20 randomised networks that keep every node's in- and
out-degree (book §3.4.1), and classify FFLs as coherent or incoherent
from the signs of their edges. Random choices come from sbp.uniform, so
that MATLAB and Python make the same swaps.
"""
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import sbp

trn = pd.read_csv(sbp.DATA / "ecoli_trn.csv")
names = np.unique(trn[["regulator", "target"]].to_numpy())
index = {name: i for i, name in enumerate(names)}
n = len(names)
src = trn["regulator"].map(index).to_numpy()
dst = trn["target"].map(index).to_numpy()
sign = trn["effect"].map({"+": 1, "-": -1}).fillna(0).to_numpy()
auto = src == dst
print(f"{n} nodes, {len(trn)} edges; {auto.sum()} regulators regulate "
      "themselves")
src, dst, sign = src[~auto], dst[~auto], sign[~auto]


def counts(src, dst):
    """Numbers of FFLs and FBLs."""
    A = sparse.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    AA = A @ A
    return int(AA.multiply(A).sum()), int(round((AA @ A).diagonal().sum() / 3))


def randomise(src, dst, seed, swaps_per_edge=10):
    """Swap targets between pairs of edges, a->b and c->d becoming a->d
    and c->b, unless that creates a self-loop or an existing edge."""
    src, dst = src.copy(), dst.copy()
    m = len(src)
    A = np.zeros((n, n), dtype=bool)
    A[src, dst] = True
    u = sbp.uniform(seed, 2 * swaps_per_edge * m)
    for t in range(swaps_per_edge * m):
        e1, e2 = int(u[2 * t] * m), int(u[2 * t + 1] * m)
        a, b, c, d = src[e1], dst[e1], src[e2], dst[e2]
        if a == d or c == b or b == d or A[a, d] or A[c, b]:
            continue
        A[a, b] = A[c, d] = False
        A[a, d] = A[c, b] = True
        dst[e1], dst[e2] = d, b
    return src, dst


ffl, fbl = counts(src, dst)
random_counts = np.array([counts(*randomise(src, dst, 100 + s))
                          for s in range(1, 21)])
mean, sd = random_counts.mean(axis=0), random_counts.std(axis=0, ddof=1)
z = (np.array([ffl, fbl]) - mean) / sd

# Coherent FFLs: the sign of X -> Z equals the product of the signs of
# X -> Y and Y -> Z. For each FFL whose three signs are all known, the
# product of the three signs is +1 if it is coherent and -1 if not.
S = sparse.csr_matrix((sign, (src, dst)), shape=(n, n))       # +1, -1, 0
K = abs(S)                                                    # sign known
signed = (K @ K).multiply(K).sum()          # FFLs with all signs known
product = (S @ S).multiply(S).sum()         # sum of their sign products
coherent = int(round((signed + product) / 2))
incoherent = int(round((signed - product) / 2))

print(f"Feed-forward loops: {ffl}; randomised {mean[0]:.1f} ± {sd[0]:.1f}; "
      f"Z = {z[0]:.1f}")
print(f"Feedback loops:     {fbl}; randomised {mean[1]:.1f} ± {sd[1]:.1f}; "
      f"Z = {z[1]:.1f}")
print(f"FFLs with all signs known: {coherent} coherent, {incoherent} "
      "incoherent")

# --- Plot
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, k, label in zip(axes, (0, 1), ("Feed-forward loops", "Feedback loops")):
    ax.hist(random_counts[:, k], bins=8, color="#1f77b4",
            label="randomised")
    ax.axvline([ffl, fbl][k], color="#d62728", lw=2, label="E. coli")
    ax.set_title(f"{label} (Z = {z[k]:.1f})")
    ax.set_xlabel("Count")
axes[0].set_ylabel("Randomised networks")
axes[0].legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-motifs")

sbp.save_results(__file__, {
    "nodes": n, "edges": len(trn), "autoregulated": int(auto.sum()),
    "ffl": ffl, "fbl": fbl, "random": random_counts, "z": z,
    "coherent": coherent, "incoherent": incoherent,
})
sbp.show()
