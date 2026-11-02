function out = gillespie(x0, changes, propensities, t_grid)
%GILLESPIE  Gillespie's stochastic simulation algorithm (SSA).
%   out = sbp.gillespie(x0, changes, propensities, t_grid) simulates one
%   trajectory and returns the state at the times t_grid (one row each).
%
%   x0            initial numbers of molecules (one per species)
%   changes       one row per reaction: how each species changes when the
%                 reaction fires
%   propensities  function of the state x, returning one rate per reaction
%   t_grid        increasing times at which to record the state

x = x0(:)';
t = 0;
out = zeros(numel(t_grid), numel(x));
k = 1;
while k <= numel(t_grid)
    a = propensities(x);
    a0 = sum(a);
    % Time to the next reaction: exponential with mean 1/a0
    if a0 > 0
        tau = -log(rand) / a0;
    else
        tau = Inf;
    end
    % Record the current state at every grid time before the event
    while k <= numel(t_grid) && t_grid(k) < t + tau
        out(k, :) = x;
        k = k + 1;
    end
    t = t + tau;
    if k <= numel(t_grid)
        % Which reaction: reaction j with probability a_j / a0
        j = find(cumsum(a) > rand * a0, 1);
        x = x + changes(j, :);
    end
end
end
