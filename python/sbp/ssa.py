"""Gillespie's stochastic simulation algorithm (SSA), shared by examples."""
import numpy as np


def gillespie(x0, changes, propensities, t_grid, rng):
    """Simulate one trajectory and return the state at the times t_grid.

    x0            initial numbers of molecules (one per species)
    changes       one row per reaction: how each species changes when the
                  reaction fires
    propensities  function of the state x, returning one rate per reaction
    t_grid        increasing times at which to record the state
    rng           a numpy random generator
    """
    changes = np.asarray(changes)
    x = np.array(x0, dtype=float)
    t = 0.0
    out = np.zeros((len(t_grid), len(x)))
    k = 0
    while k < len(t_grid):
        a = propensities(x)
        a0 = a.sum()
        # Time to the next reaction: exponential with mean 1/a0
        tau = rng.exponential(1 / a0) if a0 > 0 else np.inf
        # Record the current state at every grid time before the event
        while k < len(t_grid) and t_grid[k] < t + tau:
            out[k] = x
            k += 1
        t += tau
        if k < len(t_grid):
            # Which reaction: reaction j with probability a_j / a0
            j = np.searchsorted(np.cumsum(a), rng.random() * a0, side="right")
            x += changes[j]
    return out
