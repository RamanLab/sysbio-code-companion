"""FBA on the E. coli core metabolic model with COBRApy.

1. Predict growth with and without oxygen.
2. Vary the glucose uptake rate and watch growth respond.
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import cobra
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
print(f"{model.id}: {len(model.reactions)} reactions, "
      f"{len(model.metabolites)} metabolites, {len(model.genes)} genes")
objective = cobra.util.solver.linear_reaction_coefficients(model)
print("Objective:", ", ".join(r.id for r in objective))

# --- 1. Aerobic vs anaerobic growth
# Default medium: glucose uptake up to 10 mmol/gDW/h, oxygen unlimited.
# Uptake is a negative flux through an exchange reaction, hence lower bounds.
aerobic = model.optimize()
print(f"Aerobic growth rate:   {aerobic.objective_value:.4f} /h")

with model:  # changes inside 'with' are undone afterwards
    model.reactions.EX_o2_e.lower_bound = 0  # no oxygen uptake
    anaerobic = model.optimize()
print(f"Anaerobic growth rate: {anaerobic.objective_value:.4f} /h")

# --- 2. Growth as a function of glucose uptake
uptake = np.arange(0, 20.5, 0.5)  # mmol/gDW/h


def growth_curve(o2_uptake_limit):
    """Maximum growth for each glucose uptake limit; NaN where infeasible."""
    growth = []
    with model:
        model.reactions.EX_o2_e.lower_bound = -o2_uptake_limit
        for u in uptake:
            model.reactions.EX_glc__D_e.lower_bound = -u
            # slim_optimize returns NaN when the LP is infeasible
            growth.append(model.slim_optimize(error_value=math.nan))
    return np.array(growth)


growth_aer = growth_curve(1000)    # oxygen effectively unlimited
growth_o2_10 = growth_curve(10)    # oxygen uptake at most 10 mmol/gDW/h
growth_anaer = growth_curve(0)     # no oxygen

# --- Plot
sbp.style()
fig, ax = plt.subplots()
ax.plot(uptake, growth_aer, "o-", color="#0072B2", ms=4,
        label="O$_2$ unlimited")
ax.plot(uptake, growth_o2_10, "^-", color="#009E73", ms=4,
        label="O$_2$ uptake ≤ 10")
ax.plot(uptake, growth_anaer, "s-", color="#E69F00", ms=4,
        label="no O$_2$")
ax.set_xlabel("Maximum glucose uptake (mmol/gDW/h)")
ax.set_ylabel("Growth rate (1/h)")
ax.set_title("E. coli core: growth vs glucose uptake")
ax.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-ecoli-glucose")

sbp.save_results(__file__, {
    "n_reactions": len(model.reactions),
    "growth_aerobic": aerobic.objective_value,
    "growth_anaerobic": anaerobic.objective_value,
    "glucose_uptake": uptake,
    "growth_vs_glucose_aerobic": growth_aer,
    "growth_vs_glucose_o2_10": growth_o2_10,
    "growth_vs_glucose_anaerobic": growth_anaer,
})
sbp.show()
