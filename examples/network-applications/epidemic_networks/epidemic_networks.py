"""Epidemics on networks, and vaccinating the hubs.

An SIR epidemic on a contact network: each infected person infects each
susceptible contact with probability T (the transmissibility) and then
recovers. We compare Erdős–Rényi and preferential-attachment (scale-free)
networks with the same mean degree, 4, and then compare two ways of
vaccinating 10% of the scale-free network: at random, or the people with
the most contacts.
"""
import numpy as np
import matplotlib.pyplot as plt
import sbp

n, m = 2000, 2
n_networks, n_seeds = 10, 50
T_values = np.round(np.arange(0, 0.6001, 0.05), 10)
rng = np.random.default_rng(1)


def barabasi_albert(n, m):
    """Preferential attachment, as in the networks chapter."""
    edges = [(i, j) for i in range(m + 1) for j in range(i)]
    ends = [v for e in edges for v in e]
    for new in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            targets.add(ends[rng.integers(len(ends))])
        for t in targets:
            edges.append((new, t))
            ends += [new, t]
    return np.array(edges)


def erdos_renyi(n, mean_k):
    upper = np.triu(rng.random((n, n)) < mean_k / (n - 1), 1)
    return np.argwhere(upper)


def neighbour_lists(edges, n):
    nbrs = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return [np.array(x, dtype=int) for x in nbrs]


def outbreak(nbrs, T, seed, immune=None):
    """Final number of people infected, starting from one person. Each
    newly infected person gets one chance to infect each contact."""
    state = np.zeros(len(nbrs), dtype=bool)        # ever infected (or immune)
    if immune is not None:
        state[immune] = True
    state[seed] = True
    current, total = [seed], 1
    while current:
        new = []
        for i in current:
            for j in nbrs[i][rng.random(len(nbrs[i])) < T]:
                if not state[j]:
                    state[j] = True
                    new.append(j)
        total += len(new)
        current = new
    return total


def threshold(nbrs):
    """Epidemic threshold T_c = <k> / (<k^2> - <k>)."""
    k = np.array([len(x) for x in nbrs], dtype=float)
    return k.mean() / ((k ** 2).mean() - k.mean())


size = {"er": np.zeros(len(T_values)), "ba": np.zeros(len(T_values))}
Tc = {"er": [], "ba": []}
for _ in range(n_networks):
    networks = {"ba": neighbour_lists(barabasi_albert(n, m), n),
                "er": neighbour_lists(erdos_renyi(n, 2 * m), n)}
    for name, nbrs in networks.items():
        Tc[name].append(threshold(nbrs))
        for t, T in enumerate(T_values):
            for seed in rng.integers(n, size=n_seeds):
                size[name][t] += outbreak(nbrs, T, seed) / n
for name in size:
    size[name] /= n_networks * n_seeds
    Tc[name] = float(np.mean(Tc[name]))
print(f"Epidemic threshold <k>/(<k^2> - <k>): Erdős–Rényi {Tc['er']:.4f}, "
      f"scale-free {Tc['ba']:.4f}")
for t in [3, 6, 10]:
    print(f"T = {T_values[t]:.2f}: mean fraction infected, Erdős–Rényi "
          f"{size['er'][t]:.4f}, scale-free {size['ba'][t]:.4f}")

# --- Vaccination of 10% of a scale-free network, with T = 0.5
T_vacc, n_vacc = 0.5, n // 10
vacc = {"none": [], "random": [], "hubs": []}
for _ in range(n_networks):
    nbrs = neighbour_lists(barabasi_albert(n, m), n)
    degree = np.array([len(x) for x in nbrs])
    choices = {"none": np.array([], dtype=int),
               "random": rng.choice(n, n_vacc, replace=False),
               "hubs": np.argsort(-degree, kind="stable")[:n_vacc]}
    for name, immune in choices.items():
        others = np.setdiff1d(np.arange(n), immune)
        for seed in rng.choice(others, n_seeds):
            vacc[name].append(outbreak(nbrs, T_vacc, seed, immune) / n)
vacc = {name: float(np.mean(v)) for name, v in vacc.items()}
print(f"Vaccinating 10% at T = {T_vacc}: mean fraction infected "
      f"{vacc['none']:.4f} (no vaccination), {vacc['random']:.4f} "
      f"(random), {vacc['hubs']:.4f} (hubs)")

# --- Plot
sbp.style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                               gridspec_kw={"width_ratios": [3, 2]})
ax1.plot(T_values, size["er"], "s-", color="#ff7f0e", label="Erdős–Rényi")
ax1.plot(T_values, size["ba"], "o-", color="#1f77b4", label="scale-free")
for name, colour in [("er", "#ff7f0e"), ("ba", "#1f77b4")]:
    ax1.axvline(Tc[name], color=colour, ls=":", lw=1)
ax1.set_xlabel("Transmissibility T")
ax1.set_ylabel("Mean fraction infected")
ax1.set_title("Epidemic size (dotted: thresholds)")
ax1.legend(frameon=False)
labels = ["none", "random\n10%", "top 10%\nby degree"]
ax2.bar(labels, [vacc["none"], vacc["random"], vacc["hubs"]],
        color=["#999999", "#9ecae1", "#1f77b4"])
ax2.set_ylabel("Mean fraction infected")
ax2.set_title(f"Vaccination, scale-free, T = {T_vacc}")
ax2.set_xlabel("Who is vaccinated")
sbp.save_figure(fig, __file__, "fig-epidemic-networks")

high = T_values >= 0.4      # well above both thresholds
sbp.save_results(__file__, {
    "threshold": Tc, "T": T_values, "size_er": size["er"],
    "size_ba": size["ba"], "size_er_high_T": size["er"][high],
    "size_ba_high_T": size["ba"][high], "vaccination": vacc,
})
sbp.show()
