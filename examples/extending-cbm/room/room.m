%% Regulatory on-off minimisation (ROOM) for single deletions
% ROOM (book §9.2, Eq. 9.5) assumes that a mutant changes as few fluxes as
% possible, compared with the wild type w. A change is significant if the
% flux leaves the range [w_l, w_u], with w_l = w - delta|w| - epsilon and
% w_u = w + delta|w| + epsilon. A binary variable y_i is 1 if flux i changes
% significantly, and ROOM minimises sum(y): a mixed-integer linear
% programme (MILP). Many flux distributions reach that minimum, so we then
% report the fastest growth among them, which is unique. The wild type w is
% the flux distribution of smallest norm at optimal growth, as for MoMA.
% Binary variables make ROOM much slower than FBA or MoMA, so we look at
% five deletions in glycolysis and the pentose phosphate pathway,
% including pyruvate kinase (PYK), the example of the book. To keep the
% MILP well conditioned, the bounds lb and ub in Eq. 9.5 are the tightest
% the network allows, found by flux variability analysis (FVA), rather
% than the nominal +/-1000.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
S = model.S; lb = model.lb; ub = model.ub; ids = model.rxns;
[m, n] = size(S);
bio = find(strcmp(ids, 'Biomass_Ecoli_core'));
delta = 0.03; epsilon = 0.001;        % as Shlomi et al. used for fluxes

% Wild type: optimal growth, smallest norm
mu_wt = fba(S, lb, ub, bio);
lb_wt = lb; ub_wt = ub;
lb_wt(bio) = mu_wt * (1 - 1e-9); ub_wt(bio) = mu_wt;
w = min_norm(S, lb_wt, ub_wt);
w(abs(w) < 1e-9) = 0;
fprintf('Wild-type growth %.4f /h; %d reactions carry flux\n', mu_wt, nnz(w));

% Tightest bounds the network allows (FVA, at any growth rate)
lo = zeros(n, 1); hi = zeros(n, 1);
opts = optimoptions('linprog', 'Display', 'none');
for j = 1:n
    c = zeros(n, 1); c(j) = 1;
    [~, lo(j)] = linprog(c, [], [], S, zeros(m, 1), lb, ub, opts);
    [~, f] = linprog(-c, [], [], S, zeros(m, 1), lb, ub, opts);
    hi(j) = -f;
end

rxns = {'PGI', 'PFK', 'TPI', 'G6PDH2r', 'PYK'};
growth_fba = zeros(1, numel(rxns)); growth_room = growth_fba;
changes = growth_fba;
for k = 1:numel(rxns)
    j = strcmp(ids, rxns{k});
    lb_d = lo; ub_d = hi;
    lb_d(j) = 0; ub_d(j) = 0;
    [changes(k), mu_room] = solve_room(S, w, lb_d, ub_d, bio, delta, epsilon);
    growth_fba(k) = fba(S, lb_d, ub_d, bio) / mu_wt;
    growth_room(k) = mu_room / mu_wt;
    fprintf(['  %-8s %2d significant changes; growth FBA %.3f, ROOM ' ...
        '%.3f of wild type\n'], rxns{k}, changes(k), growth_fba(k), ...
        growth_room(k));
end

%% Plot
fig = figure('Position', [100 100 1000 400]);
subplot(1, 2, 1)
b = bar(categorical(rxns, rxns), [growth_fba; growth_room]');
b(1).FaceColor = [1.000 0.498 0.055];
b(2).FaceColor = [0.122 0.467 0.706];
ylabel('Growth (fraction of wild type)'); title('Growth after each deletion');
legend({'FBA', 'ROOM'}, 'Box', 'off', 'Location', 'southwest');
box off
subplot(1, 2, 2)
bar(categorical(rxns, rxns), changes, 'FaceColor', [0.122 0.467 0.706]);
ylabel('Significant flux changes'); title('Fewest flux changes (ROOM)');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-room');

results.growth_wild_type = mu_wt;
results.reactions = rxns;
results.growth_fba = growth_fba;
results.growth_room = growth_room;
results.changes = changes;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function mu = fba(S, lb, ub, bio)
% Maximum growth rate.
c = zeros(size(lb)); c(bio) = -1;              % linprog minimises
opts = optimoptions('linprog', 'Display', 'none');
[~, f, exitflag] = linprog(c, [], [], S, zeros(size(S, 1), 1), lb, ub, ...
    opts);
assert(exitflag == 1, 'linprog failed');
mu = max(0, -f);
end

function v = min_norm(S, lb, ub)
% The flux vector of smallest Euclidean norm (QP, as in MoMA). Dropping
% dependent rows of S, and using the active-set algorithm, makes quadprog
% exact and reliable here.
n = numel(lb);
S = full(S);
[~, R, order] = qr(S', 'vector');               % independent rows of S
independent = order(abs(diag(R)) > 1e-9 * abs(R(1, 1)));
S = S(sort(independent), :);
opts = optimoptions('quadprog', 'Algorithm', 'active-set', ...
    'Display', 'none');
x0 = min(max(zeros(n, 1), lb), ub);
[v, ~, exitflag] = quadprog(eye(n), zeros(n, 1), [], [], S, ...
    zeros(size(S, 1), 1), lb, ub, x0, opts);
assert(exitflag > 0, 'quadprog failed');
end

function [changes, mu] = solve_room(S, w, lb, ub, bio, delta, epsilon)
% Fewest significant changes from w, and the fastest growth that keeps
% to that number. Variables are [v; y], y binary.
[m, n] = size(S);
wl = w - delta * abs(w) - epsilon;
wu = w + delta * abs(w) + epsilon;
I = speye(n);
% S v = 0;  v - y (lb - wl) >= wl;  v - y (ub - wu) <= wu
Aeq = [sparse(S), sparse(m, n)];
A = [-I, spdiags(lb - wl, 0, n, n);      % -(v - y (lb - wl)) <= -wl
      I, -spdiags(ub - wu, 0, n, n)];    %   v - y (ub - wu)  <=  wu
b = [-wl; wu];
lower = [lb; zeros(n, 1)]; upper = [ub; ones(n, 1)];
intcon = n + (1:n);
opts = optimoptions('intlinprog', 'Display', 'none', ...
    'RelativeGapTolerance', 0, 'AbsoluteGapTolerance', 0, ...
    'IntegerTolerance', 1e-6, 'ConstraintTolerance', 1e-9);
[~, f, exitflag] = intlinprog([zeros(n, 1); ones(n, 1)], intcon, A, b, ...
    Aeq, zeros(m, 1), lower, upper, opts);
assert(exitflag == 1, 'intlinprog failed (exit flag %d)', exitflag);
changes = round(f);
% Second step: fastest growth with at most that many changes
c = zeros(2 * n, 1); c(bio) = -1;
A = [A; zeros(1, n), ones(1, n)];
b = [b; changes];
[x, f, exitflag] = intlinprog(c, intcon, A, b, Aeq, zeros(m, 1), ...
    lower, upper, opts);
assert(exitflag == 1, 'intlinprog failed (exit flag %d)', exitflag);
v = x(1:n);
tol = 1e-6;                       % check: count the changes from v itself
assert(sum(v < wl - tol | v > wu + tol) <= changes);
mu = -f;
end
