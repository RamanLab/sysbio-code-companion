"""Single gene deletions in a genome-scale model: E. coli iML1515.

The same procedure as for the core model (single_deletions), on the
genome-scale reconstruction iML1515 (Monk et al., 2017): 1516 genes and
2712 reactions, in its default medium (aerobic, glucose uptake up to
10 mmol/gDW/h). A gene deletion removes the reactions whose GPR rule it
makes false, and is lethal if growth falls below 1% of the wild type.
We also record the genes whose deletion removes no reaction at all,
because an isozyme or another subunit can stand in for them.
"""
import numpy as np
import matplotlib.pyplot as plt
import cobra
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "iML1515.xml.gz"))
wt = model.slim_optimize()
genes = sorted(g.id for g in model.genes)

ratio, lost = {}, {}
for gid in genes:
    gene = model.genes.get_by_id(gid)
    with model:                                 # undone after the block
        gene.knock_out()
        lost[gid] = sorted(r.id for r in gene.reactions if not r.functional)
        ratio[gid] = model.slim_optimize(error_value=0.0) / wt
essential = [g for g in genes if ratio[g] < 0.01]
reduced = [g for g in genes if 0.01 <= ratio[g] < 0.99]
no_rxn = [g for g in genes if not lost[g]]

print(f"Wild-type growth {wt:.4f} /h; {len(genes)} genes deleted one by one")
print(f"{len(essential)} essential genes ({100 * len(essential) / len(genes):.1f}%)")
print(f"{len(reduced)} genes whose loss reduces growth, but not below 1%")
print(f"{len(no_rxn)} genes whose loss removes no reaction (isozymes)")

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(5, 4.5))
counts = [len(essential), len(reduced),
          len(genes) - len(essential) - len(reduced)]
ax.bar(["lethal (< 1%)", "reduced", "no effect"], counts,
       color=["#d62728", "#ff7f0e", "#1f77b4"])
for x, k in enumerate(counts):
    ax.text(x, k, str(k), ha="center", va="bottom")
ax.set_ylabel("Genes")
ax.set_title("Single gene deletions, iML1515")
sbp.save_figure(fig, __file__, "fig-genome-scale-deletions")

sbp.save_results(__file__, {
    "wild_type_growth": wt,
    "growth_ratio": ratio,
    "essential_genes": essential,
    "n_reduced": len(reduced),
    "no_reaction_lost": no_rxn,
})
sbp.show()
