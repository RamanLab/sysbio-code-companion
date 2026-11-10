"""A metabolic network as a graph: the substrate graph of E. coli core.

In a substrate graph (book §2.5.5), metabolites are nodes, and two are
joined if one is converted into the other by some reaction: a substrate
and a product of the same reaction. We build it from the stoichiometric
matrix of the E. coli core model, leaving out the exchange reactions and
the biomass reaction. Currency metabolites, such as H+, water, ATP and
NAD(H), take part in so many reactions that they connect almost
everything to everything else, and shortest paths through them make no
biochemical sense. We compare the graph with and without them.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
S = create_stoichiometric_matrix(model)
mets = [m.id for m in model.metabolites]
internal = [j for j, r in enumerate(model.reactions)
            if not r.id.startswith(("EX_", "Biomass"))]

G = nx.Graph()
G.add_nodes_from(mets)
for j in internal:
    for a in np.flatnonzero(S[:, j] < 0):           # substrates
        for b in np.flatnonzero(S[:, j] > 0):       # products
            G.add_edge(mets[a], mets[b])

currency = ["h_c", "h_e", "h2o_c", "h2o_e", "atp_c", "adp_c", "amp_c",
            "pi_c", "pi_e", "nad_c", "nadh_c", "nadp_c", "nadph_c",
            "co2_c", "co2_e", "coa_c", "nh4_c", "nh4_e", "o2_c",
            "q8_c", "q8h2_c"]
H = G.copy()
H.remove_nodes_from(currency)


def summary(graph):
    """Size, and path lengths within the largest connected component."""
    largest = graph.subgraph(max(nx.connected_components(graph), key=len))
    d = dict(nx.all_pairs_shortest_path_length(largest))
    lengths = [d[u][v] for u in largest for v in largest if u < v]
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
            "components": nx.number_connected_components(graph),
            "largest": largest.number_of_nodes(),
            "mean_path": np.mean(lengths), "diameter": max(lengths),
            "path_counts": np.bincount(lengths)[1:]}


results = {"with_currency": summary(G), "without_currency": summary(H)}
degree = sorted(G.degree, key=lambda x: (-x[1], x[0]))
results["top_degree"] = [m for m, _ in degree[:10]]
results["top_degree_values"] = [k for _, k in degree[:10]]
results["glc_to_succ"] = {
    "with_currency": nx.shortest_path_length(G, "g6p_c", "succ_c"),
    "without_currency": nx.shortest_path_length(H, "g6p_c", "succ_c")}

print("Most connected metabolites: " + ", ".join(
    f"{m} ({k})" for m, k in degree[:10]))
for key, r in results.items():
    if key.endswith("currency"):
        print(f"{key.replace('_', ' ')}: {r['nodes']} metabolites, "
              f"{r['edges']} edges, {r['components']} component(s); in the "
              f"largest ({r['largest']}), mean path {r['mean_path']:.2f}, "
              f"diameter {r['diameter']}")
print("A shortest path from glucose 6-phosphate to succinate:")
print("  with currency metabolites:    " + " - ".join(
    nx.shortest_path(G, "g6p_c", "succ_c")))
print("  without currency metabolites: " + " - ".join(
    nx.shortest_path(H, "g6p_c", "succ_c")))

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
top = degree[:15]
ax1.bar(range(15), [k for _, k in top],
        color=["#d62728" if m in currency else "#1f77b4" for m, _ in top])
ax1.set_xticks(range(15), [m for m, _ in top], rotation=60, fontsize=8)
ax1.set_ylabel("Degree")
ax1.set_title("Most connected metabolites (red: currency)")
for key, colour, label in (("with_currency", "#d62728", "with currency"),
                           ("without_currency", "#1f77b4", "without")):
    counts = results[key]["path_counts"]
    ax2.plot(range(1, len(counts) + 1), counts / counts.sum(), "o-",
             color=colour, label=label)
ax2.set_xlabel("Shortest path length")
ax2.set_ylabel("Fraction of pairs")
ax2.legend(frameon=False)
ax2.set_title("Path lengths in the largest component")
sbp.save_figure(fig, __file__, "fig-metabolic-graphs")

sbp.save_results(__file__, results)
sbp.show()
