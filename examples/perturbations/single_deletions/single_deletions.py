"""Single reaction and gene deletions in the E. coli core model.

A reaction is deleted by fixing its flux to zero (lower and upper bounds
both 0; book Eq. 10.2); the maximum growth rate is then found by FBA. A
deletion is called lethal if growth falls below 1% of the wild type.
Genes are deleted through the gene-protein-reaction (GPR) rules, which
COBRApy evaluates for us: a reaction is lost only if the deletion makes
its rule false, so isozymes can stand in for one another.
"""
import numpy as np
import matplotlib.pyplot as plt
import cobra
from cobra.flux_analysis import single_gene_deletion
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
wt = model.slim_optimize()
cutoff = 0.01 * wt
# Delete internal reactions only: no exchange reactions, no biomass
rxns = sorted(r.id for r in model.reactions
              if not r.id.startswith(("EX_", "Biomass")))

# --- Reaction deletions
ratio = {}
for rid in rxns:
    with model:                                 # undone after the block
        model.reactions.get_by_id(rid).bounds = (0, 0)
        ratio[rid] = model.slim_optimize(error_value=0.0) / wt
essential_rxns = [r for r in rxns if ratio[r] < 0.01]
reduced = [r for r in rxns if 0.01 <= ratio[r] < 0.99]

# --- Gene deletions (the GPR rules decide which reactions are lost)
gd = single_gene_deletion(model)
gene_ratio = {next(iter(ids)): (g if np.isfinite(g) else 0.0) / wt
              for ids, g in zip(gd["ids"], gd["growth"])}
essential_genes = sorted(g for g, r in gene_ratio.items() if r < 0.01)
names = [model.genes.get_by_id(g).name for g in essential_genes]

print(f"Wild-type growth {wt:.4f} /h; {len(rxns)} reactions tested")
print(f"{len(essential_rxns)} essential reactions: "
      f"{', '.join(essential_rxns)}")
print(f"{len(reduced)} reactions whose loss reduces growth, but not below 1%")
print(f"{len(essential_genes)} essential genes, of {len(model.genes)}: "
      f"{', '.join(f'{g} ({n})' for g, n in zip(essential_genes, names))}")

# --- Plot: growth after each reaction deletion, sorted
order = sorted(rxns, key=lambda r: ratio[r])
values = np.array([ratio[r] for r in order])
colors = np.where(values < 0.01, "#d62728",
                  np.where(values < 0.99, "#ff7f0e", "#1f77b4"))
sbp.style()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(order)), values, color=colors)
lethal = np.flatnonzero(values < 0.01)      # zero-height bars: mark them
ax.plot(lethal, np.full(lethal.size, 0.03), "v", color="#d62728", ms=5)
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
ax.set_ylabel("Growth relative to wild type")
ax.set_title("Single reaction deletions, E. coli core model, aerobic glucose")
for col, lab in [("#d62728", "lethal (< 1%)"), ("#ff7f0e", "reduced"),
                 ("#1f77b4", "no effect")]:
    ax.bar(0, 0, color=col, label=lab)
ax.legend(frameon=False, loc="upper left")
sbp.save_figure(fig, __file__, "fig-single-deletions")

sbp.save_results(__file__, {
    "wild_type_growth": wt,
    "growth_ratio": ratio,
    "essential_reactions": essential_rxns,
    "essential_genes": essential_genes,
})
sbp.show()
