# Data

| File | What it is | Source |
|---|---|---|
| `e_coli_core.xml` | *E. coli* core metabolic model (95 reactions, 72 metabolites, 137 genes), SBML Level 3 with the FBC package | BiGG Models `e_coli_core` (Orth, Fleming & Palsson, 2010, *EcoSal Plus*, doi:10.1128/ecosalplus.10.2.1), as distributed with COBRApy 0.32.1 and written out with `cobra.io.write_sbml_model` |
| `e_coli_core.mat` | The same model in COBRA Toolbox format, for the MATLAB examples (reading SBML in MATLAB needs a compiled libSBML, which is not available for every release) | Generated from `e_coli_core.xml` by `tools/make_mat_model.py`; regenerate it whenever the `.xml` changes |
| `competitive_inhibition.csv` | Initial reaction rates $v$ (mmol L⁻¹ s⁻¹) at 20 substrate concentrations $S$ (mM), for an enzyme with a competitive inhibitor at $I$ = 10 mM (`v_I10`) and $I$ = 20 mM (`v_I20`) | BT5240 laboratory exercise on parameter estimation (Lab 8); also the basis of Exercise 6.1 in the book |

When adding a file, record here where it came from and under what terms.
