"""Dynamic FBA: E. coli growing on glucose in a batch culture.

FBA predicts steady-state rates. Dynamic FBA follows a culture through
time by solving an FBA problem at each short time step: the current
concentrations set the maximum uptake rates, FBA gives the growth rate
and the exchange fluxes, and these update biomass and concentrations
(Mahadevan, Edwards and Doyle, 2002). Oxygen uptake is limited, as in a
dense culture, so cells growing fast on glucose overflow acetate, which
they consume once the glucose runs out.
"""
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import cobra
from cobra.util import create_stoichiometric_matrix
import sbp

model = cobra.io.read_sbml_model(str(sbp.DATA / "e_coli_core.xml"))
S = create_stoichiometric_matrix(model)
lb0 = np.array([r.lower_bound for r in model.reactions])
ub0 = np.array([r.upper_bound for r in model.reactions])
ids = [r.id for r in model.reactions]
bio, glc, ac, o2 = (ids.index(r) for r in
                    ["Biomass_Ecoli_core", "EX_glc__D_e", "EX_ac_e",
                     "EX_o2_e"])
n = len(ids)

# Uptake kinetics (mmol/gDW/h, mM) and the oxygen limit
vmax_glc, km_glc = 10.0, 0.015
vmax_ac, km_ac = 5.0, 0.015
o2_max = 15.0

dt, t_end = 0.01, 12.0                 # h
X, G, A = 0.01, 20.0, 0.0              # biomass (gDW/L), glucose, acetate
times = np.round(np.arange(0, t_end + dt / 2, dt), 10)
traj = np.zeros((len(times), 4))       # X, G, A, growth rate
for k, t in enumerate(times):
    # Uptake is limited by kinetics, and by what is left in the medium
    lb, ub = lb0.copy(), ub0.copy()
    lb[glc] = -min(vmax_glc * G / (km_glc + G), G / (X * dt))
    lb[ac] = -min(vmax_ac * A / (km_ac + A), A / (X * dt))
    lb[o2] = -o2_max
    c = np.zeros(n)
    c[bio] = -1
    res = linprog(c, A_eq=S, b_eq=np.zeros(S.shape[0]),
                  bounds=np.c_[lb, ub], method="highs")
    if res.status == 0:
        mu, v = max(0.0, -res.fun), res.x
    else:                              # cannot even meet maintenance
        mu, v = 0.0, np.zeros(n)
    traj[k] = X, G, A, mu
    # Euler step: dX/dt = mu X, dC/dt = v_exchange X
    G = max(0.0, G + v[glc] * X * dt)
    A = max(0.0, A + v[ac] * X * dt)
    X = X + mu * X * dt

X_t, G_t, A_t, mu_t = traj.T
t_glc = times[np.argmax(G_t < 1e-9)]          # glucose used up
k_peak = np.argmax(A_t)
growing = mu_t > 0
t_stop = times[np.flatnonzero(growing)[-1] + 1]
print(f"Glucose runs out at t = {t_glc:.2f} h; acetate peaks at "
      f"{A_t[k_peak]:.3f} mM at t = {times[k_peak]:.2f} h")
print(f"Growth on glucose at {mu_t[0]:.4f} /h, then on acetate at "
      f"{mu_t[times > t_glc + 0.1][0]:.4f} /h; growth stops at t = "
      f"{t_stop:.2f} h with {X_t[-1]:.4f} gDW/L")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogy(times, X_t, color="#2ca02c")
ax1.set_xlabel("Time (h)")
ax1.set_ylabel("Biomass (gDW/L)")
ax1.set_title("Biomass")
ax2.plot(times, G_t, color="#1f77b4", label="glucose")
ax2.plot(times, A_t, color="#ff7f0e", label="acetate")
ax2.set_xlabel("Time (h)")
ax2.set_ylabel("Concentration (mM)")
ax2.set_title("Substrates and by-products")
ax2.legend(frameon=False)
for ax in (ax1, ax2):
    ax.axvline(t_glc, color="#999999", ls=":", lw=1)
sbp.save_figure(fig, __file__, "fig-dynamic-fba")

hourly = np.isclose(times % 1, 0) | np.isclose(times % 1, 1)
sbp.save_results(__file__, {
    "t_glucose_exhausted": t_glc, "acetate_peak": A_t[k_peak],
    "t_acetate_peak": times[k_peak], "t_growth_stops": t_stop,
    "growth_on_glucose": mu_t[0], "final_biomass": X_t[-1],
    "hours": times[hourly], "biomass_hourly": X_t[hourly],
    "glucose_hourly": G_t[hourly], "acetate_hourly": A_t[hourly],
})
sbp.show()
