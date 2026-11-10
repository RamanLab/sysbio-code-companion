"""A probabilistic Boolean network (PBN), analysed exactly.

We start from the three-gene network of §7.2 (book Fig. 7.1):
    g1* = g2 AND NOT g3,    g2* = NOT g1,    g3* = NOT g3 OR g2,
whose attractors are the steady state 011 and the cycle 001 -> 010 -> 111.
In the PBN (book §7.3.1), g3 follows its rule with probability 0.8, and
an alternative rule, g3* = NOT g1, with probability 0.2; at each step,
one of the two is picked at random. Also, as proposed by Shmulevich and
colleagues, each gene may be flipped at random with probability
p = 0.01, as if by noise. The PBN is then a Markov chain on the 2^3 = 8
states, and we can compute its transition matrix, and the long-run
probability of each state (the stationary distribution), exactly. For
comparison, we do the same for the original network with noise alone.
"""
import numpy as np
import matplotlib.pyplot as plt
import sbp

n = 3
p_flip = 0.01


def bits(s):
    """State number -> (g1, g2, g3), first gene as the leftmost digit."""
    return [(s >> (n - 1 - i)) & 1 for i in range(n)]


def number(x):
    return sum(int(b) << (n - 1 - i) for i, b in enumerate(x))


def transition_matrix(rules_g3):
    """Markov chain of the PBN, for a list of (rule for g3, probability)."""
    T = np.zeros((2 ** n, 2 ** n))
    for s in range(2 ** n):
        g1, g2, g3 = bits(s)
        for rule, prob in rules_g3:
            T[s, number([g2 and not g3, not g1, rule(g1, g2, g3)])] += prob
    # With probability (1 - p)^n no gene flips and the network acts;
    # otherwise, the genes that flip are flipped instead
    P = (1 - p_flip) ** n * T
    for s in range(2 ** n):
        for flips in range(1, 2 ** n):            # which genes flip (as bits)
            k = bin(flips).count("1")
            P[s, s ^ flips] += p_flip ** k * (1 - p_flip) ** (n - k)
    return P


def stationary(P):
    """pi P = pi, with the probabilities summing to 1."""
    A = np.vstack([P.T - np.eye(len(P)), np.ones(len(P))])
    b = np.r_[np.zeros(len(P)), 1]
    return np.linalg.lstsq(A, b, rcond=None)[0]


original = lambda g1, g2, g3: (not g3) or g2
alternative = lambda g1, g2, g3: not g1
P_noise = transition_matrix([(original, 1.0)])
P_pbn = transition_matrix([(original, 0.8), (alternative, 0.2)])
pi = {"noise": stationary(P_noise), "pbn": stationary(P_pbn)}

labels = ["".join(map(str, bits(s))) for s in range(2 ** n)]
cycle = [labels.index(x) for x in ("001", "010", "111")]
steady = labels.index("011")
print("State   noise only   PBN")
for s in range(2 ** n):
    print(f"  {labels[s]}     {pi['noise'][s]:.4f}    {pi['pbn'][s]:.4f}")
for key, name in (("noise", "Noise only"), ("pbn", "PBN")):
    print(f"{name}: steady state 011 {pi[key][steady]:.3f}; cycle "
          f"{pi[key][cycle].sum():.3f}")

# --- Plot
sbp.style()
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2 ** n)
ax.bar(x - 0.2, pi["noise"], 0.4, color="#999999", label="noise only")
ax.bar(x + 0.2, pi["pbn"], 0.4, color="#1f77b4", label="PBN")
ax.set_xticks(x, labels)
ax.set_xlabel("State (g1 g2 g3)")
ax.set_ylabel("Long-run probability")
ax.legend(frameon=False)
sbp.save_figure(fig, __file__, "fig-probabilistic")

sbp.save_results(__file__, {"states": labels, "transition": P_pbn,
                            "stationary_noise": pi["noise"],
                            "stationary_pbn": pi["pbn"]})
sbp.show()
