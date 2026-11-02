"""Synthetic lethal reaction pairs in the E. coli core model.

A pair of reactions is synthetic lethal if deleting both stops growth
(below 1% of wild type) while deleting either one alone does not. We
find all such pairs twice:
  1. exhaustively, testing every pair of non-essential reactions;
  2. as in Fast-SL (Pratapa et al., 2015), testing only pairs that can
     possibly be lethal. If a growing flux distribution uses neither
     reaction of a pair, deleting both leaves it intact, so the pair is
     not lethal. Hence at least one reaction of a lethal pair carries flux
     in the wild-type solution, and the other carries flux once the first
     is deleted. Using sparse solutions (minimal total flux, pFBA) keeps
     the number of pairs to test small.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import cobra
from cobra.flux_analysis import pfba
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
wt = model.slim_optimize()
cutoff = 0.01 * wt
rxns = sorted(r.id for r in model.reactions
              if not r.id.startswith(("EX_", "Biomass")))


def growth_without(deleted):
    with model:
        for rid in deleted:
            model.reactions.get_by_id(rid).bounds = (0, 0)
        return model.slim_optimize(error_value=0.0)


def active_without(deleted, candidates):
    """Candidate reactions carrying flux in a minimal-total-flux solution."""
    with model:
        for rid in deleted:
            model.reactions.get_by_id(rid).bounds = (0, 0)
        v = pfba(model).fluxes
    return [r for r in candidates if abs(v[r]) > 1e-9]


essential = [r for r in rxns if growth_without([r]) < cutoff]
candidates = [r for r in rxns if r not in essential]

# --- 1. Exhaustive: every pair of non-essential reactions
pairs = list(itertools.combinations(candidates, 2))
lethal_exhaustive = [p for p in pairs if growth_without(p) < cutoff]

# --- 2. Fast-SL: only pairs (i, j) with i active in the wild type and j
# active once i is deleted
tested = set()
lethal_fastsl = set()
for i in active_without([], candidates):
    for j in active_without([i], candidates):
        pair = tuple(sorted((i, j)))
        if j != i and pair not in tested:
            tested.add(pair)
            if growth_without(pair) < cutoff:
                lethal_fastsl.add(pair)
lethal_fastsl = sorted(lethal_fastsl)

print(f"{len(candidates)} non-essential reactions")
print(f"Exhaustive: {len(pairs)} pairs tested, "
      f"{len(lethal_exhaustive)} synthetic lethal")
print(f"Fast-SL:    {len(tested)} pairs tested, "
      f"{len(lethal_fastsl)} synthetic lethal")
print(f"Same pairs found: {lethal_fastsl == sorted(lethal_exhaustive)}")
partners = {r: sum(r in p for p in lethal_fastsl) for r in candidates}
top = sorted(partners, key=lambda r: (-partners[r], r))[:5]
print("Reactions in the most pairs: "
      + ", ".join(f"{r} ({partners[r]})" for r in top))

# --- Plot: which reactions form lethal pairs
involved = sorted({r for p in lethal_fastsl for r in p})
idx = {r: k for k, r in enumerate(involved)}
M = np.zeros((len(involved), len(involved)))
for a, b in lethal_fastsl:
    M[idx[a], idx[b]] = M[idx[b], idx[a]] = 1
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                               gridspec_kw={"width_ratios": [2.2, 1]})
ax1.imshow(M, cmap="Blues", vmin=0, vmax=1.3)
ax1.set_xticks(range(len(involved)), involved, rotation=90, fontsize=7)
ax1.set_yticks(range(len(involved)), involved, fontsize=7)
ax1.set_title(f"{len(lethal_fastsl)} synthetic lethal pairs")
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.bar(["exhaustive", "Fast-SL"], [len(pairs), len(tested)],
        color=["#999999", "#1f77b4"])
ax2.set_ylabel("Double deletions simulated")
ax2.set_title("Cost of the search")
sbp.save_figure(fig, __file__, "fig-synthetic-lethals")

sbp.save_results(__file__, {
    "n_candidates": len(candidates),
    "n_pairs_exhaustive": len(pairs),
    "lethal_pairs": [list(p) for p in lethal_fastsl],
    "same_as_exhaustive": lethal_fastsl == sorted(lethal_exhaustive),
    "n_pairs_fastsl": len(tested),
})
sbp.show()
