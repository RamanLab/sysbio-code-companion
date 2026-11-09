"""Extract the expression data used by the E-Flux example (§9.4.2).

Source: the PRECISE compendium of E. coli K-12 MG1655 RNA-seq (Sastry et
al., 2019, Nat. Commun. 10:5536), https://github.com/SBRG/precise-db
(MIT licence). We keep six wild-type samples grown aerobically in M9
minimal medium on different carbon sources, with their measured growth
rates, and the genes of the iML1515 model.

Usage: python tools/make_precise_subset.py PATH/TO/precise-db
"""
import sys
from pathlib import Path
import gzip
import pandas as pd

src = Path(sys.argv[1]) / "data"
root = Path(__file__).resolve().parent.parent
samples = ["ssw__wt_glc__1", "ssw__wt_xyl__1", "ssw__wt_glyc__1",
           "ssw__wt_ac__1", "cra_crp__wt_fru__1", "cra_crp__wt_ac__1"]

expr = pd.read_csv(src / "log_tpm.csv", index_col=0)[samples]
meta = pd.read_csv(src / "metadata.csv").set_index("Sample ID").loc[samples]
with gzip.open(root / "data" / "iML1515.xml.gz", "rt") as f:
    model_genes = sorted(set(pd.Series(f.read().split('fbc:id="G_')[1:])
                             .str.split('"').str[0]))
expr = expr.loc[expr.index.intersection(model_genes)].sort_index()
expr.index.name = "gene"
expr.round(4).to_csv(root / "data" / "precise_log_tpm.csv")

info = pd.DataFrame({
    "sample": samples,
    "carbon_source": meta["Carbon Source (g/L)"].values,
    "growth_rate": meta["Growth Rate (1/hr)"].values,
    "study": meta["Study"].values,
    "doi": meta["DOI"].values,
})
info.to_csv(root / "data" / "precise_samples.csv", index=False)
print(f"{expr.shape[0]} genes x {expr.shape[1]} samples")
print(info.to_string(index=False))
