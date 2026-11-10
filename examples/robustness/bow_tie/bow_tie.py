"""The bow-tie structure of a genome-scale metabolic network.

Following Ma and Zeng (2003), we build a directed substrate graph from
the E. coli model iML1515: an edge from every substrate of a reaction to
every product, and back again if the reaction is reversible. Exchange
and other one-metabolite reactions, the biomass reactions and ATP
maintenance are left out, as are currency metabolites (in every
compartment), which would otherwise connect almost everything.
The bow-tie (book §13.1.3.1) then has four parts:
  * the core, or "knot": the largest strongly connected component, in
    which every metabolite can be converted into every other one;
  * IN: metabolites from which the core can be reached, but not back;
  * OUT: metabolites that can be reached from the core, but not back;
  * the rest, which neither reach the core nor are reached from it.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "iML1515.xml.gz"))
S = create_stoichiometric_matrix(model)
mets = [m.id for m in model.metabolites]
rxns = [r.id for r in model.reactions]
lb = np.array([r.lower_bound for r in model.reactions])

currency = {"h", "h2o", "atp", "adp", "amp", "pi", "ppi", "nad", "nadh",
            "nadp", "nadph", "co2", "coa", "nh4", "o2", "q8", "q8h2",
            "mql8", "mqn8", "2dmmq8", "2dmmql8", "fad", "fadh2", "gtp",
            "gdp", "ctp", "cdp", "utp", "udp", "h2o2", "na1", "k", "fe2",
            "fe3", "glu__L", "akg", "accoa", "so4", "gln__L"}
keep = np.array([m.rsplit("_", 1)[0] not in currency for m in mets])

G = nx.DiGraph()
for j, r in enumerate(rxns):
    column = S[:, j]
    if np.count_nonzero(column) < 2 or r.startswith("BIOMASS") or r == "ATPM":
        continue
    substrates = [mets[i] for i in np.flatnonzero((column < 0) & keep)]
    products = [mets[i] for i in np.flatnonzero((column > 0) & keep)]
    for a in substrates:
        for b in products:
            G.add_edge(a, b)
            if lb[j] < 0:
                G.add_edge(b, a)

core = max(nx.strongly_connected_components(G), key=len)
seed = min(core)
out = nx.descendants(G, seed) - core
inn = nx.ancestors(G, seed) - core
rest = set(G) - core - out - inn
parts = {"in": inn, "core": core, "out": out, "rest": rest}


def compartments(nodes):
    """How many metabolites of a part are cytosolic, periplasmic or
    extracellular (the last letter of the identifier)."""
    tags = [n.rsplit("_", 1)[1] for n in nodes]
    return [tags.count(c) for c in ("c", "p", "e")]


results = {"metabolites": G.number_of_nodes(), "edges": G.number_of_edges()}
print(f"{G.number_of_nodes()} metabolites, {G.number_of_edges()} edges")
for name, nodes in parts.items():
    results[name] = {"size": len(nodes), "by_compartment": compartments(nodes)}
    c, p, e = compartments(nodes)
    print(f"  {name:5s} {len(nodes):5d} metabolites ({100 * len(nodes) / len(G):.0f}%): "
          f"{c} cytosolic, {p} periplasmic, {e} extracellular")
results["landmarks"] = {}
for m in ("glc__D_e", "pyr_c", "succ_c", "ala__L_c", "cbl1_e", "btn_c"):
    where = next((k for k, v in parts.items() if m in v), "absent")
    results["landmarks"][m] = where
    print(f"  {m:10s} is in {where}")

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(6.5, 4))
names = ["IN", "core", "OUT", "rest"]
bottom = np.zeros(4)
for k, (label, colour) in enumerate((("cytosol", "#1f77b4"),
                                     ("periplasm", "#ff7f0e"),
                                     ("extracellular", "#2ca02c"))):
    heights = [results[p]["by_compartment"][k] for p in parts]
    ax.bar(names, heights, bottom=bottom, color=colour, label=label)
    bottom += heights
ax.set_ylabel("Metabolites")
ax.set_title("The bow-tie of iML1515")
ax.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-bow-tie")

sbp.save_results(__file__, results)
sbp.show()
