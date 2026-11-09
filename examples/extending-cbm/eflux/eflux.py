"""E-Flux: constraining fluxes with gene expression data.

E-Flux (book §9.4.2) uses the expression of the genes behind a reaction
as an approximate upper bound on its flux. The expression of a reaction
follows its GPR rule: genes joined by 'and' (subunits of one complex)
give the minimum of their expression, and genes joined by 'or'
(isozymes) give the sum. The bounds of each reaction are then set to
+/- this value, divided by the largest value over all reactions.

The data are RNA-seq profiles of wild-type E. coli MG1655, grown
aerobically in minimal medium on six carbon sources, from the PRECISE
compendium (Sastry et al., 2019), with the growth rates measured in the
same experiments. For each sample we build E-Flux bounds for the
genome-scale model iML1515, open the uptake of that sample's carbon
source, maximise growth, and compare the predictions with the measured
growth rates. Expression is in log2(TPM + 1); we try both the raw TPM
values, as in the original E-Flux, and the log values.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cobra
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "iML1515.xml.gz"))
log_tpm = pd.read_csv(sbp.DATA / "precise_log_tpm.csv", index_col=0)
samples = pd.read_csv(sbp.DATA / "precise_samples.csv")
uptake = {"glucose": "EX_glc__D_e", "xylose": "EX_xyl__D_e",
          "glycerol": "EX_glyc_e", "acetate": "EX_ac_e",
          "fructose": "EX_fru_e"}


def gpr_value(rule, expression):
    """Evaluate a GPR rule such as '(b1 and b2) or b3': 'and' takes the
    minimum, 'or' the sum. A gene without data does not constrain the
    reaction (its value is infinite)."""
    tokens = re.findall(r"\(|\)|[^\s()]+", rule)
    position = 0

    def term():                      # a gene, or a bracketed rule
        nonlocal position
        token = tokens[position]
        position += 1
        if token == "(":
            value = either()
            position += 1            # skip ')'
            return value
        return expression.get(token, np.inf)

    def both():                      # terms joined by 'and'
        nonlocal position
        value = term()
        while position < len(tokens) and tokens[position] == "and":
            position += 1
            value = min(value, term())
        return value

    def either():                    # 'and' groups joined by 'or'
        nonlocal position
        value = both()
        while position < len(tokens) and tokens[position] == "or":
            position += 1
            value = value + both()
        return value

    return either()


def eflux_growth(expression, carbon):
    """Maximum growth with E-Flux bounds, on one carbon source."""
    with model:
        values = {r.id: gpr_value(r.gene_reaction_rule, expression)
                  for r in model.reactions if r.gene_reaction_rule}
        largest = max(v for v in values.values() if np.isfinite(v))
        for rid, v in values.items():
            if np.isfinite(v):
                r = model.reactions.get_by_id(rid)
                bound = v / largest
                r.bounds = (-bound if r.lower_bound < 0 else 0,
                            bound if r.upper_bound > 0 else 0)
        for ex in uptake.values():                 # only this carbon source
            model.reactions.get_by_id(ex).lower_bound = 0
        model.reactions.get_by_id(uptake[carbon]).lower_bound = -1000
        model.reactions.ATPM.lower_bound = 0       # see the text
        return model.slim_optimize()


carbon = samples["carbon_source"].str.split("(").str[0]
measured = samples["growth_rate"].to_numpy()
predicted = {}
for transform, data in [("TPM", 2 ** log_tpm - 1), ("log TPM", log_tpm)]:
    predicted[transform] = np.array([
        eflux_growth(data[s].to_dict(), c)
        for s, c in zip(samples["sample"], carbon)])
r = {t: np.corrcoef(measured, p)[0, 1] for t, p in predicted.items()}

print("carbon source    measured   E-Flux (TPM)   E-Flux (log TPM)")
for i, c in enumerate(carbon):
    print(f"{c:12s}  {measured[i]:8.2f}   {predicted['TPM'][i]:12.3g}"
          f"   {predicted['log TPM'][i]:14.3g}")
for t in predicted:
    print(f"Correlation with measured growth, {t}: r = {r[t]:.2f}")

# --- Plot
sbp.style()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, t in zip(axes, predicted):
    ax.plot(measured, predicted[t], "o", color="#1f77b4")
    for x, y, c in zip(measured, predicted[t], carbon):
        ax.annotate(c, (x, y), textcoords="offset points", xytext=(5, 3),
                    fontsize=9)
    ax.set_xlabel("Measured growth rate (1/h)")
    ax.set_ylabel("E-Flux growth (arbitrary units)")
    ax.set_title(f"Expression as {t}: r = {r[t]:.2f}")
sbp.save_figure(fig, __file__, "fig-eflux")

sbp.save_results(__file__, {
    "carbon_source": list(carbon), "measured": measured,
    "eflux_tpm": predicted["TPM"], "eflux_log_tpm": predicted["log TPM"],
    "r_tpm": r["TPM"], "r_log_tpm": r["log TPM"],
})
sbp.show()
