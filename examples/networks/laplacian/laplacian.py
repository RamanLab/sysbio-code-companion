"""The Laplacian matrix, and what its eigenvectors say about a network.

Zachary's karate club: 34 members, and 78 friendships observed outside
the club. A dispute between the instructor (member 1, Mr. Hi) and the
administrator (member 34, the Officer) split the club in two. The
Laplacian is L = D - A (book §2.4.3), where D holds the degrees on its
diagonal. Its eigenvalues are all >= 0; the number of zero eigenvalues
is the number of connected components; and the eigenvector of the
second-smallest eigenvalue (the Fiedler vector) splits the network into
two well-separated parts, by the signs of its entries.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sbp

edges = pd.read_csv(sbp.DATA / "karate_edges.csv")
faction = pd.read_csv(sbp.DATA / "karate_factions.csv")["faction"].to_numpy()
n = len(faction)
A = np.zeros((n, n))
for i, j in edges.to_numpy() - 1:                   # members are 1 to 34
    A[i, j] = A[j, i] = 1
L = np.diag(A.sum(axis=1)) - A

eigenvalues, eigenvectors = np.linalg.eigh(L)       # ascending order
fiedler = eigenvectors[:, 1]
fiedler *= np.sign(fiedler[0])                      # member 1 positive
side = np.where(fiedler > 0, "Mr. Hi", "Officer")
agree = np.sum(side == faction)

print(f"Smallest eigenvalues: {np.round(eigenvalues[:4], 4)}")
print(f"Zero eigenvalues: {np.sum(eigenvalues < 1e-9)} (connected components)")
print(f"Algebraic connectivity: {eigenvalues[1]:.4f}")
print(f"The Fiedler vector puts {agree} of {n} members on the side they "
      f"actually joined; it misplaces member(s) "
      f"{', '.join(str(k) for k in np.flatnonzero(side != faction) + 1)}")

# Remove the friendships between the two halves found by the Fiedler
# vector: the network falls apart, and a second zero eigenvalue appears
cut = [(i, j) for i, j in edges.to_numpy() - 1 if side[i] != side[j]]
A_cut = A.copy()
for i, j in cut:
    A_cut[i, j] = A_cut[j, i] = 0
ev_cut = np.linalg.eigvalsh(np.diag(A_cut.sum(axis=1)) - A_cut)
print(f"Cutting the {len(cut)} edges between the halves leaves "
      f"{np.sum(ev_cut < 1e-9)} zero eigenvalues")

# --- Plot: the network, coloured by the sign of the Fiedler vector
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
G = nx.from_numpy_array(A)
pos = nx.spring_layout(G, seed=1)
colours = ["#1f77b4" if s == "Mr. Hi" else "#d62728" for s in side]
nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=colours, node_size=180,
                       edgecolors=["black" if s != f else "none"
                                   for s, f in zip(side, faction)])
nx.draw_networkx_labels(G, pos, {k: k + 1 for k in range(n)}, ax=ax1,
                        font_size=7, font_color="white")
ax1.set_title("Split by the Fiedler vector")
ax1.axis("off")
order = np.argsort(fiedler)
ax2.bar(range(n), fiedler[order], color=[colours[k] for k in order])
ax2.set_xticks(range(n), order + 1, fontsize=6, rotation=90)
ax2.set_xlabel("Member")
ax2.set_ylabel("Fiedler vector entry")
sbp.save_figure(fig, __file__, "fig-laplacian")

sbp.save_results(__file__, {
    "eigenvalues": eigenvalues, "fiedler": fiedler, "agree": agree,
    "misplaced": np.flatnonzero(side != faction) + 1,
    "cut_edges": len(cut), "zero_eigenvalues_after_cut": np.sum(ev_cut < 1e-9),
})
sbp.show()
