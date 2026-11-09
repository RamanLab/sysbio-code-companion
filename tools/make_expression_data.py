"""Write data/gene_expression_timecourse.csv: a synthetic time course of
mRNA and protein after a gene is switched on at t = 0, used by the
parameter-estimation chapter.

    dm/dt = k_m - g_m m,    dp/dt = k_p m - g_p p,    m(0) = p(0) = 0

with k_m = 10 /h, g_m = 1 /h, k_p = 5 /h, g_p = 0.2 /h, sampled every
half hour for 12 h, with 10% normally distributed relative noise. The
values are rounded to three significant figures. Rerunning this script
reproduces the file exactly.
"""
from pathlib import Path

import numpy as np

k_m, g_m, k_p, g_p = 10.0, 1.0, 5.0, 0.2
t = np.arange(0.5, 12.01, 0.5)
m = k_m / g_m * (1 - np.exp(-g_m * t))
p = (k_m * k_p / (g_m * g_p)
     * (1 + (g_p * np.exp(-g_m * t) - g_m * np.exp(-g_p * t)) / (g_m - g_p)))
rng = np.random.default_rng(2026)
m_obs = m * (1 + 0.1 * rng.standard_normal(len(t)))
p_obs = p * (1 + 0.1 * rng.standard_normal(len(t)))

out = (Path(__file__).resolve().parents[1] / "data"
       / "gene_expression_timecourse.csv")
with open(out, "w") as f:
    f.write("t_h,mRNA,protein\n")
    for row in zip(t, m_obs, p_obs):
        f.write(",".join(f"{x:.3g}" for x in row) + "\n")
print(f"Wrote {out}")
