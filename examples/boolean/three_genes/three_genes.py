"""A three-gene Boolean network (book Fig. 7.1, Eqs. 7.3).

    g1* = g2 AND NOT g3
    g2* = NOT g1
    g3* = NOT g3 OR g2

With synchronous updating, all genes are updated at once from the
previous state. We list the successor of each of the 2^3 = 8 states,
and find the attractors (the states the network ends up cycling
through) and their basins (the states that lead to them).
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sbp

genes = ["g1", "g2", "g3"]
n = len(genes)


def update(x):
    g1, g2, g3 = x
    return (g2 and not g3, not g1, (not g3) or g2)


def to_bits(s):
    """State number -> tuple of 0/1, first gene as the leftmost digit."""
    return tuple((s >> (n - 1 - i)) & 1 for i in range(n))


def to_number(x):
    return sum(int(b) << (n - 1 - i) for i, b in enumerate(x))


def label(s):
    return "".join(map(str, to_bits(s)))


# Successor of every state
succ = [to_number(update(to_bits(s))) for s in range(2 ** n)]
for s in range(2 ** n):
    print(f"{label(s)} => {label(succ[s])}")

# Attractors: follow each state until a state repeats; the repeating part
# is the attractor. Every state is assigned to the attractor it reaches.
attractor_of = {}
attractors = []
for s in range(2 ** n):
    path = []
    while s not in path:
        path.append(s)
        s = succ[s]
    cycle = path[path.index(s):]
    key = min(cycle)
    if key not in [min(a) for a in attractors]:
        i = cycle.index(key)                    # start each cycle at its
        attractors.append(cycle[i:] + cycle[:i])  # smallest state
    attractor_of[path[0]] = key
basins = {min(a): sum(1 for v in attractor_of.values() if v == min(a))
          for a in attractors}
for a in attractors:
    print(f"Attractor of period {len(a)}: "
          f"{' -> '.join(label(s) for s in a)}; basin of {basins[min(a)]} "
          f"states")

# --- Plot the state transition map
G = nx.DiGraph([(label(s), label(succ[s])) for s in range(2 ** n)])
pos = {"100": (0, 1), "110": (0, 0), "001": (1, 1), "101": (1, 0),
       "010": (2, 1.5), "111": (2, 0.5), "000": (3, 1), "011": (3, 0)}
in_attractor = {label(s) for a in attractors for s in a}
sbp.style()
fig, ax = plt.subplots(figsize=(7, 3.5))
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1100, edgecolors="k",
                       node_color=["#ff7f0e" if v in in_attractor else "white"
                                   for v in G.nodes])
nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)
nx.draw_networkx_edges(G, pos, ax=ax, node_size=1100, arrowsize=18,
                       connectionstyle="arc3,rad=0.15")
ax.annotate("steady state\n(011 => 011)", pos["011"], xytext=(0, -38),
            textcoords="offset points", ha="center", va="top", fontsize=9)
ax.set_title("State transitions (orange: states in an attractor)")
ax.set_axis_off()
ax.margins(0.12)
sbp.save_figure(fig, __file__, "fig-three-genes")

sbp.save_results(__file__, {
    "successor": [label(s) for s in succ],
    "attractors": [[label(s) for s in a] for a in attractors],
    "basin_sizes": [basins[min(a)] for a in attractors],
})
sbp.show()
