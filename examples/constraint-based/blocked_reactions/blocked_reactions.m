%% Blocked reactions and dead-end metabolites
% 1. The small network of book Figure 8.3 (Eq. 8.13): find its blocked
%    reactions from the reduced row echelon form of S (Eq. 8.14), and
%    confirm them by FVA.
% 2. The E. coli core model: find dead-end metabolites from the
%    stoichiometry alone, and the reactions that cannot carry flux, first
%    with every exchange reaction open, then on glucose.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

%% 1. The network of book Figure 8.3
names = {'v1', 'v2', 'v3', 'b1', 'b2', 'b3'};
%    v1  v2  v3  b1  b2  b3
S = [-1   0  -1   1   0   0;     % A
     -1   1  -1   0   0   0;     % B
      1  -1   0   0   0   0;     % C
      0   1   0   0  -1   0;     % D
      0   0   1   0   0  -1];    % E
R = rref(S);
disp('Reduced row echelon form of S (book Eq. 8.14):');
disp(R);
% A row with a single non-zero entry says that flux must be zero
single = sum(R ~= 0, 2) == 1;
[~, col] = max(R(single, :) ~= 0, [], 2);
by_rref = names(col);
[vmin, vmax] = fva(S, zeros(6, 1), 1000 * ones(6, 1));
by_fva = names(abs(vmin) < 1e-9 & abs(vmax) < 1e-9);
fprintf('Blocked, from the RREF: %s; from FVA: %s\n', ...
    strjoin(by_rref, ', '), strjoin(by_fva, ', '));

%% 2. The E. coli core model
s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
S = full(model.S); lb = model.lb; ub = model.ub;
rxns = model.rxns; mets = model.mets;
exchange = startsWith(rxns, 'EX_');

% Every exchange reaction open. The ATP maintenance demand (a lower bound
% > 0) is relaxed to 0 here, since it asks for flux rather than allowing it.
lb_open = min(lb, 0);
lb_open(exchange) = -1000;
dead_ends = dead_end_metabolites(S, lb_open, ub, mets);
dead_ends_glucose = dead_end_metabolites(S, lb, ub, mets);
fprintf(['Dead-end metabolites with every exchange open: %d; on ' ...
    'glucose: %s\n'], numel(dead_ends), strjoin(dead_ends_glucose, ', '));

[vmin, vmax] = fva(S, lb_open, ub);
blocked = rxns(abs(vmin) < 1e-9 & abs(vmax) < 1e-9);
fprintf('Reactions blocked in every condition: %d\n', numel(blocked));

% No flux on glucose (the model's default medium), at any growth rate
[vmin, vmax] = fva(S, lb, ub);
no_flux = rxns(abs(vmin) < 1e-9 & abs(vmax) < 1e-9);
fprintf('Reactions that cannot carry flux on glucose: %d\n  %s\n', ...
    numel(no_flux), strjoin(no_flux, ', '));

results.rref = R;
results.blocked_rref = by_rref;
results.blocked_fva = by_fva;
results.dead_ends = dead_ends;
results.dead_ends_glucose = dead_ends_glucose;
results.blocked_core = blocked;
results.no_flux_glucose = no_flux';
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function [vmin, vmax] = fva(S, lb, ub)
% Smallest and largest value of every flux, with S v = 0 and the bounds
% (no objective constraint).
n = size(S, 2);
opts = optimoptions('linprog', 'Display', 'none');
vmin = zeros(n, 1); vmax = zeros(n, 1);
for j = 1:n
    c = zeros(n, 1); c(j) = 1;
    [~, vmin(j)] = linprog(c, [], [], S, zeros(size(S, 1), 1), lb, ub, opts);
    [~, f] = linprog(-c, [], [], S, zeros(size(S, 1), 1), lb, ub, opts);
    vmax(j) = -f;
end
end

function d = dead_end_metabolites(S, lb, ub, mets)
% Metabolites that, given the directions the reactions can run, can only
% ever be produced or only ever be consumed.
can_make = (S > 0 & (ub > 0)') | (S < 0 & (lb < 0)');
can_use = (S < 0 & (ub > 0)') | (S > 0 & (lb < 0)');
d = mets(~(any(can_make, 2) & any(can_use, 2)))';
end
