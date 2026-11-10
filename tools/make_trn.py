"""Build the E. coli transcriptional regulatory network used in §3.4.

Source: the regulator-gene table of PRECISE-1K (Lamoureux et al., 2023,
Nucleic Acids Res. 51:10176, doi:10.1093/nar/gkad750), file
data/annotation/TRN.csv of https://github.com/SBRG/precise1k (MIT
licence). We keep the transcription factor interactions taken from
RegulonDB (source "RegulonDB_TF"), and leave out sigma factors and sRNAs.

Each regulator becomes one node. A regulator encoded by several genes
(a complex such as FlhDC, or a pair such as RcsAB) merges those genes
into one node, and genes shared between such regulators (RcsB, in RcsAB,
RcsB-BglJ and GadE-RcsB) merge them all. Every other gene is a node of
its own. An edge X -> Y means that regulator X binds upstream of a gene
of node Y. Repeated edges are kept once; their effect becomes "+-" if
the signs disagree. Edges from a node to itself (autoregulation) are
kept, and flagged.

Usage: python tools/make_trn.py PATH/TO/precise1k
"""
import sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1]) / "data" / "annotation"
root = Path(__file__).resolve().parent.parent
trn = pd.read_csv(src / "TRN.csv")
trn = trn[trn["source"] == "RegulonDB_TF"]
genes = pd.read_csv(src / "gene_info.csv", index_col=0)

# Gene names (and synonyms) to b-numbers
lookup = {}
for b, row in genes.iterrows():
    for name in [row["gene_name"]] + str(row["synonyms"]).split(";"):
        lookup.setdefault(str(name).lower(), b)

# Regulators whose names are not gene names: complexes, pairs, old names
aliases = {"NtrC": ["glnG"], "IHF": ["ihfA", "ihfB"], "H-NS": ["hns"],
           "HU": ["hupA", "hupB"], "FlhDC": ["flhD", "flhC"],
           "RcsAB": ["rcsA", "rcsB"], "RcsB-BglJ": ["rcsB", "bglJ"],
           "GadE-RcsB": ["gadE", "rcsB"], "CRP-Sxy": ["crp", "sxy"],
           "YefM-YoeB": ["yefM", "yoeB"], "MazE-MazF": ["mazE", "mazF"],
           "RelB-RelE": ["relB", "relE"], "DinJ-YafQ": ["dinJ", "yafQ"],
           "HipAB": ["hipA", "hipB"], "HigBA": ["higB", "higA"],
           "PlaR": ["yiaJ"]}


def regulator_genes(name):
    names = aliases.get(name, [name])
    return [lookup[n.lower()] for n in names if n.lower() in lookup]


# Merge the genes of each regulator into one node (union-find)
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


regulators = sorted(trn["regulator"].unique())
for r in regulators:
    g = regulator_genes(r)
    for a in g[1:]:
        parent[find(a)] = find(g[0])


def node_of_gene(b):
    """Name of the node holding gene b: its gene names, joined by '/'."""
    root_b = find(b)
    members = sorted(x for x in list(parent) + [b] if find(x) == root_b)
    return "/".join(sorted({genes.loc[x, "gene_name"] for x in members}))


def node_of_regulator(r):
    g = regulator_genes(r)
    return node_of_gene(g[0]) if g else r      # unmapped: keep its own name


rows = {}
for r, b, effect in trn[["regulator", "gene_id", "effect"]].itertuples(index=False):
    if b not in genes.index:
        continue
    key = (node_of_regulator(r), node_of_gene(b))
    old = rows.get(key)
    rows[key] = effect if old in (None, effect) else "+-"
edges = pd.DataFrame([(a, b, e) for (a, b), e in sorted(rows.items())],
                     columns=["regulator", "target", "effect"])
edges.to_csv(root / "data" / "ecoli_trn.csv", index=False)
tfs = edges["regulator"].nunique()
nodes = len(set(edges["regulator"]) | set(edges["target"]))
print(f"{len(edges)} edges, {nodes} nodes, {tfs} regulators, "
      f"{(edges['regulator'] == edges['target']).sum()} autoregulated")
