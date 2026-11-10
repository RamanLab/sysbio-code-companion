"""Extract the expression data used by the differential network example (§4.3).

Source: PRECISE-1K (Lamoureux et al., 2023, Nucleic Acids Res. 51:10176,
doi:10.1093/nar/gkad750), https://github.com/SBRG/precise1k (MIT
licence), file data/precise1k/log_tpm_qc.csv and its metadata. We keep
the wild-type (unevolved) samples grown in batch culture in M9 medium at
37 °C on glucose or glycerol, and 44 genes of central carbon and glycerol
metabolism.

Usage: python tools/make_precise1k_subset.py PATH/TO/precise1k
"""
import sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1]) / "data"
root = Path(__file__).resolve().parent.parent
meta = pd.read_csv(src / "precise1k" / "metadata_qc.csv", index_col=0)
expr = pd.read_csv(src / "precise1k" / "log_tpm_qc.csv", index_col=0)
info = pd.read_csv(src / "annotation" / "gene_info.csv", index_col=0)

genes = ("ptsG ptsH ptsI crr pgi pfkA fbaA tpiA gapA pgk gpmA eno pykF pykA "
         "zwf gnd glpF glpK glpD glpA glpB glpC glpT glpQ glpR gltA acnB icd "
         "sucA sucB sucC sucD sdhA sdhB fumA mdh ppsA pckA fbp maeB aceA aceB "
         "aceK crp cyaA").split()
bnum = {name: b for b, name in info["gene_name"].items()}
genes = [g for g in genes if g in bnum]

keep = meta[(meta["Base Media"] == "M9") & (meta["Evolved Sample"] == "No")
            & (meta["Temperature (C)"] == 37) & (meta["Culture Type"] == "Batch")]
carbon = keep["Carbon Source (g/L)"].fillna("").str.split("(").str[0]
keep = keep[carbon.isin(["glucose", "glycerol"])]
samples = pd.DataFrame({"sample": keep.index,
                        "carbon_source": carbon[keep.index].values,
                        "study": keep["study"].values,
                        "strain": keep["Strain"].values})
samples = samples.sort_values(["carbon_source", "sample"])
table = expr.loc[[bnum[g] for g in genes], samples["sample"]].round(4)
table.index = genes
table.index.name = "gene"
table.to_csv(root / "data" / "precise1k_carbon_log_tpm.csv")
samples.to_csv(root / "data" / "precise1k_carbon_samples.csv", index=False)
print(f"{len(genes)} genes x {len(samples)} samples:")
print(samples.groupby("carbon_source").agg(n=("sample", "size"),
                                           studies=("study", "nunique")))
