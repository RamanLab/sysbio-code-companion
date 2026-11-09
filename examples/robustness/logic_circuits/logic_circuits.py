"""Robustness and evolvability in a space of logic circuits.

A circuit is a chain of four NAND gates. Each input of gate g (g = 1..4)
is wired to one of the two circuit inputs, A and B, or to the output of
an earlier gate; the output of gate 4 is the output of the circuit. The
wiring is the genotype, and the logic function the circuit computes, one
of 16 possible truth tables, is its phenotype. A mutation rewires one
input of one gate. We enumerate all 14,400 circuits, and ask how robust
each phenotype is to mutations, and how many new phenotypes lie one
mutation away (its evolvability).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import sbp

# Truth tables as 4-bit numbers, outputs for (A, B) = 00, 01, 10, 11
A, B = 0b0011, 0b0101
NAMES = ["FALSE", "A AND B", "A AND NOT B", "A", "NOT A AND B", "B",
         "A XOR B", "A OR B", "A NOR B", "A XNOR B", "NOT B", "A OR NOT B",
         "NOT A", "NOT A OR B", "A NAND B", "TRUE"]

# A genotype is 8 digits, the sources of the two inputs of each gate;
# gate g can choose among 2 + (g - 1) sources
radix = np.array([2, 2, 3, 3, 4, 4, 5, 5])
weight = np.r_[np.cumprod(radix[::-1])[::-1][1:], 1]   # first digit slowest
n = int(np.prod(radix))
index = np.arange(n)
digits = (index[:, None] // weight) % radix              # n x 8

# Phenotypes: evaluate all circuits at once, with bitwise NAND
signals = [np.full(n, A), np.full(n, B)]
for g in range(4):
    stacked = np.array(signals)                          # sources so far
    x = stacked[digits[:, 2 * g], index]
    y = stacked[digits[:, 2 * g + 1], index]
    signals.append(15 - (x & y))
phenotype = signals[-1]

# One-mutation neighbours: change one digit to each of its other values
same = np.zeros(n)                       # neighbours with the same phenotype
seen = np.zeros(n, dtype=int)            # phenotypes of neighbours, as bits
edges = []
for d in range(8):
    for v in range(radix[d]):
        mutant = digits[:, d] != v
        nb = index[mutant] + (v - digits[mutant, d]) * weight[d]
        same_ph = phenotype[nb] == phenotype[mutant]
        same[mutant] += same_ph
        seen[mutant] |= 1 << phenotype[nb]
        edges.append(np.c_[index[mutant][same_ph], nb[same_ph]])
n_neighbours = int(np.sum(radix - 1))                    # 20
robustness = same / n_neighbours
own = 1 << phenotype
popcount = np.vectorize(lambda b: bin(b).count("1"))
evolvability = popcount(seen & ~own)
edges = np.vstack(edges)
graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                   shape=(n, n))
_, component = connected_components(graph, directed=False)

# Summarise for each phenotype that occurs
found = np.unique(phenotype)
summary = {"phenotype": found.tolist(), "name": [NAMES[f] for f in found],
           "n_genotypes": [], "robustness": [], "evolvability": [],
           "n_components": []}
for f in found:
    g = phenotype == f
    summary["n_genotypes"].append(int(g.sum()))
    summary["robustness"].append(robustness[g].mean())
    reach = np.bitwise_or.reduce(seen[g]) & ~(1 << f)
    summary["evolvability"].append(int(popcount(reach)))
    summary["n_components"].append(len(np.unique(component[g])))
    print(f"{NAMES[f]:12s} {g.sum():5d} circuits, robustness "
          f"{robustness[g].mean():.3f}, {summary['evolvability'][-1]:2d} "
          f"new phenotypes nearby, {summary['n_components'][-1]} "
          f"genotype network(s)")
print(f"{n} circuits compute {len(found)} of the 16 functions")

r_genotype = np.corrcoef(robustness, evolvability)[0, 1]
r_phenotype = np.corrcoef(summary["robustness"],
                          summary["evolvability"])[0, 1]
print(f"Correlation of robustness and evolvability: {r_genotype:.3f} for "
      f"genotypes, {r_phenotype:.3f} for phenotypes")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.semilogx(summary["n_genotypes"], summary["robustness"], "o",
             color="#1f77b4")
for name, x, y in zip(summary["name"], summary["n_genotypes"],
                      summary["robustness"]):
    if name in ["A XOR B", "A NOR B", "FALSE", "A OR B", "A NAND B"]:
        ax1.annotate(name, (x, y), textcoords="offset points",
                     xytext=(5, -3), fontsize=8)
ax1.set_xlim(8, 8000)
ax1.set_xlabel("Number of circuits with this function")
ax1.set_ylabel("Robustness of the function")
ax1.set_title("Common functions are robust")
levels = np.unique(robustness)
ax2.plot(levels, [evolvability[robustness == r].mean() for r in levels],
         "o-", color="#ff7f0e", label="single circuits")
ax2.plot(summary["robustness"], summary["evolvability"], "s",
         color="#1f77b4", label="functions (all their circuits)")
ax2.set_xlabel("Robustness")
ax2.set_ylabel("New functions one mutation away")
ax2.set_title("Robustness and evolvability")
ax2.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-logic-circuits")

sbp.save_results(__file__, {
    "n_circuits": n, "summary": summary,
    "correlation_genotype": r_genotype,
    "correlation_phenotype": r_phenotype,
})
sbp.show()
