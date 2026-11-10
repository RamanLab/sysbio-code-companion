"""Link prediction: can the structure of a network predict missing edges?

We hide 10% of the edges of the E. coli STRING network (combined score at
least 700), and score pairs of nodes using only the remaining network,
by four similarity indices (book §4.7):
  common neighbours    CN(x, y) = |N(x) ∩ N(y)|
  Jaccard              CN(x, y) / |N(x) ∪ N(y)|
  Adamic-Adar          sum over common neighbours z of 1 / log k_z
  preferential attachment   k_x k_y
An index is good if it scores the hidden edges above pairs that are not
linked in the full network. We measure this by the AUC: the probability
that a randomly chosen hidden edge scores higher than a randomly chosen
non-edge (ties count one half). The hidden edges, and an equal number of
non-edges, are chosen with sbp.uniform, so that MATLAB and Python use the
same ones.
"""
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import sbp

table = pd.read_csv(sbp.DATA / "ecoli_string_ppi.csv")[["protein1", "protein2"]]
names = np.unique(table.to_numpy())
index = {name: i for i, name in enumerate(names)}
edges = np.array([[index[a], index[b]] for a, b in table.to_numpy()])
n, m = len(names), len(edges)


def adjacency(e):
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (A + A.T).tocsr()


# Hide 10% of the edges, chosen at random
order = np.argsort(sbp.uniform(11, m))
hidden = edges[order[:m // 10]]
train = edges[order[m // 10:]]

# The same number of non-edges: random pairs not linked in the full network
full = adjacency(edges)
u = sbp.uniform(12, 6 * len(hidden))
non_edges, seen, k = [], set(), 0
while len(non_edges) < len(hidden):
    a, b = int(u[k] * n), int(u[k + 1] * n)
    k += 2
    a, b = min(a, b), max(a, b)
    if a != b and full[a, b] == 0 and (a, b) not in seen:
        seen.add((a, b))
        non_edges.append((a, b))
non_edges = np.array(non_edges)

# Similarity indices, from the training network only
A = adjacency(train)
degree = np.asarray(A.sum(axis=1)).ravel()
common = A @ A                                            # CN for every pair
weights = np.zeros(n)
weights[degree > 1] = 1 / np.log(degree[degree > 1])
adamic_adar = A @ sparse.diags(weights) @ A


def scores(pairs):
    x, y = pairs[:, 0], pairs[:, 1]
    cn = np.asarray(common[x, y]).ravel()
    union = degree[x] + degree[y] - cn
    jaccard = np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
    aa = np.asarray(adamic_adar[x, y]).ravel()
    # Round away floating-point noise, so that exact ties stay ties
    return {"common_neighbours": cn, "jaccard": np.round(jaccard, 12),
            "adamic_adar": np.round(aa, 12),
            "preferential_attachment": degree[x] * degree[y]}


def auc(positive, negative):
    """Probability that a positive scores above a negative (ties: 1/2),
    from the average ranks of all the scores (Mann-Whitney)."""
    values = np.r_[positive, negative]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):                      # average ranks over ties
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    p = len(positive)
    return (ranks[:p].sum() - p * (p + 1) / 2) / (p * len(negative))


positive, negative = scores(hidden), scores(non_edges)
results = {"hidden": len(hidden),
           "auc": {k: auc(positive[k], negative[k]) for k in positive},
           "no_common_neighbour": float(np.mean(positive["common_neighbours"] == 0))}
print(f"{len(hidden)} hidden edges, and as many non-edges")
for k, v in results["auc"].items():
    print(f"  AUC, {k.replace('_', ' ')}: {v:.3f}")
print(f"{100 * results['no_common_neighbour']:.1f}% of the hidden edges join "
      "nodes with no common neighbour in the training network")

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["common\nneighbours", "Jaccard", "Adamic–Adar",
          "preferential\nattachment"]
ax.bar(labels, list(results["auc"].values()), color="#1f77b4")
ax.axhline(0.5, ls=":", color="black")
ax.set_ylim(0, 1)
ax.set_ylabel("AUC")
ax.set_title("Predicting hidden STRING edges")
sbp.save_figure(fig, __file__, "fig-link-prediction")

sbp.save_results(__file__, results)
sbp.show()
