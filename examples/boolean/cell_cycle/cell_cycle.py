"""A Boolean model of the mammalian cell cycle (Fauré et al., 2006).

Ten genes and proteins; the update rules are the book's Eq. 7.4. With
2^10 = 1024 possible states, we can examine every one of them:
  1. synchronous updating, as in the previous example;
  2. asynchronous updating, where one variable changes at a time, in
     any order. The states then form a 'state transition graph', and the
     attractors are the groups of states that, once entered, are never
     left: its terminal strongly connected components.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sbp

names = ["CycD", "Rb", "E2F", "CycE", "CycA", "p27", "Cdc20", "Cdh1",
         "Ubc", "CycB"]
n = len(names)


def update(x):
    CycD, Rb, E2F, CycE, CycA, p27, Cdc20, Cdh1, Ubc, CycB = x
    return (
        CycD,
        (not CycA and not CycB and not CycD and not CycE)
        or (p27 and not CycB and not CycD),
        (not Rb and not CycA and not CycB) or (p27 and not Rb and not CycB),
        E2F and not Rb,
        (E2F and not Rb and not Cdc20 and not (Cdh1 and Ubc))
        or (CycA and not Rb and not Cdc20 and not (Cdh1 and Ubc)),
        (not CycD and not CycE and not CycA and not CycB)
        or (p27 and not (CycE and CycA) and not CycB and not CycD),
        CycB,
        (not CycA and not CycB) or Cdc20 or (p27 and not CycB),
        (not Cdh1) or (Cdh1 and Ubc and (Cdc20 or CycA or CycB)),
        not Cdc20 and not Cdh1,
    )


def to_bits(s):
    return tuple((s >> (n - 1 - i)) & 1 for i in range(n))


def to_number(x):
    return sum(int(b) << (n - 1 - i) for i, b in enumerate(x))


states = range(2 ** n)
target = [to_number(update(to_bits(s))) for s in states]

# --- 1. Synchronous updating: follow every state to its attractor
sync_attractors = {}           # smallest state of the cycle -> cycle
reaches = []
for s in states:
    path = []
    while s not in path:
        path.append(s)
        s = target[s]
    cycle = path[path.index(s):]
    sync_attractors.setdefault(min(cycle), cycle)
    reaches.append(min(cycle))
sync_summary = [{"period": len(c), "basin": reaches.count(k),
                 "CycD": to_bits(k)[0]}
                for k, c in sorted(sync_attractors.items())]
print("Synchronous updating:")
for a in sync_summary:
    print(f"  attractor of period {a['period']} (CycD = {a['CycD']}), "
          f"basin of {a['basin']} states")

# --- 2. Asynchronous updating: from each state, any one variable whose
# rule says it should change may change next
G = nx.DiGraph()
G.add_nodes_from(states)
for s in states:
    x, fx = to_bits(s), to_bits(target[s])
    for i in range(n):
        if x[i] != fx[i]:
            y = list(x)
            y[i] = fx[i]
            G.add_edge(s, to_number(y))
C = nx.condensation(G)       # one node per strongly connected component
terminal = [c for c in C.nodes if C.out_degree(c) == 0]
async_summary = sorted(
    ({"size": len(C.nodes[c]["members"]),
      "CycD": to_bits(min(C.nodes[c]["members"]))[0]} for c in terminal),
    key=lambda a: (a["CycD"], a["size"]))
print("Asynchronous updating:")
for a in async_summary:
    print(f"  attractor of {a['size']} states (CycD = {a['CycD']})")

# --- A synchronous trajectory: start from a G1-like state, with CycD on
x0 = dict.fromkeys(names, 0)
x0.update(CycD=1, Rb=1, p27=1, Cdh1=1)
traj = [tuple(x0[g] for g in names)]
for _ in range(20):
    traj.append(tuple(int(v) for v in update(traj[-1])))
traj = np.array(traj)

sbp.style()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.imshow(traj.T, cmap="Blues", aspect="auto", vmin=0, vmax=1.3)
ax.set_yticks(range(n), names)
ax.set_xticks(range(0, 21, 2))
ax.set_xlabel("Time step")
ax.set_title("Synchronous trajectory from a G1-like state "
             "(dark: ON, light: OFF)")
for spine in ax.spines.values():
    spine.set_visible(False)
sbp.save_figure(fig, __file__, "fig-cell-cycle")

sbp.save_results(__file__, {
    "synchronous": sync_summary,
    "asynchronous": async_summary,
    "trajectory": traj,
})
sbp.show()
