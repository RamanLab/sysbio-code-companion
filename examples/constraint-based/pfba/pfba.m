%% Loops, parsimonious FBA (pFBA), and the pFBA classes of reactions
% 1. The network of book Figure 8.4, which contains a stoichiometrically
%    balanced cycle: FVA shows the cycle can carry any flux, and minimising
%    the total flux, sum_j |v_j|, removes it.
% 2. The E. coli core model on glucose: classify every internal reaction,
%    as in book Section 8.8.5, as essential, pFBA optimal, enzymatically
%    less efficient (ELE), metabolically less efficient (MLE), no flux in
%    this medium, or blocked.
% Total flux is minimised as a linear programme, with variables t_j >= |v_j|.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

%% 1. Book Figure 8.4: A -> B -> C, and C -> D -> A closes a cycle
names = {'b1', 'v1', 'v2', 'b2', 'v3', 'v4'};
%    b1  v1  v2  b2  v3  v4
S = [ 1  -1   0   0   0   1;     % A
      0   1  -1   0   0   0;     % B
      0   0   1  -1  -1   0;     % C
      0   0   0   0   1  -1];    % D
lb = zeros(6, 1); ub = [10; 1000; 1000; 1000; 1000; 1000];
c = zeros(6, 1); c(4) = -1;      % maximise b2
best = -lp(c, S, lb, ub, [], []);
lb_opt = lb; lb_opt(4) = best;   % hold b2 at its maximum
toy_fva = fva(S, lb_opt, ub, [], [], 1:6);
[S2, lb2, ub2, A, b] = with_abs(S, lb_opt, ub);
c2 = [zeros(6, 1); ones(6, 1)];
toy_min_total = lp(c2, S2, lb2, ub2, A, b);
% Every flux is fixed once the total is minimal: check with FVA
toy_pfba = fva(S2, lb2, ub2, [A; c2'], [b; toy_min_total * (1 + 1e-9)], 1:6);
fprintf('Figure 8.4: max b2 = %g\n', best);
for k = 1:6
    fprintf('  %s: FVA [%g, %g]; with minimal total flux [%.4g, %.4g]\n', ...
        names{k}, toy_fva(k, 1), toy_fva(k, 2), toy_pfba(k, 1), ...
        toy_pfba(k, 2));
end

%% 2. pFBA classes for the E. coli core model
s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
S = full(model.S); lb = model.lb; ub = model.ub; rxns = model.rxns;
n = numel(rxns);
bio = find(strcmp(rxns, 'Biomass_Ecoli_core'));
internal = find(~startsWith(rxns, {'EX_', 'Biomass'}))';
c = zeros(n, 1); c(bio) = -1;
mu = -lp(c, S, lb, ub, [], []);
iszero = @(r) abs(r(1)) < 1e-6 && abs(r(2)) < 1e-6;

% Ranges: on glucose at any growth; at optimal growth; at optimal growth
% with minimal total flux
any_growth = fva(S, lb, ub, [], [], internal);
lb_mu = lb; lb_mu(bio) = mu * (1 - 1e-9);
at_optimum = fva(S, lb_mu, ub, [], [], internal);
[S2, lb2, ub2, A, b] = with_abs(S, lb_mu, ub);
c2 = [zeros(n, 1); ones(n, 1)];
min_total = lp(c2, S2, lb2, ub2, A, b);
parsimonious = fva(S2, lb2, ub2, [A; c2'], [b; min_total * (1 + 1e-9)], ...
    internal);
% Essential: growth below 1% of wild type without the reaction
essential = false(size(internal));
for k = 1:numel(internal)
    lb_d = lb; ub_d = ub;
    lb_d(internal(k)) = 0; ub_d(internal(k)) = 0;
    g = lp(c, S, lb_d, ub_d, [], []);
    essential(k) = isempty(g) || -g < 0.01 * mu;
end

labels = {'essential', 'pFBA optima', 'ELE', 'MLE', 'no flux', 'blocked'};
fields = {'essential', 'pFBA_optima', 'ELE', 'MLE', 'no_flux', 'blocked'};
members = repmat({{}}, 1, 6);
for k = 1:numel(internal)
    if essential(k)
        cls = 1;
    elseif ~iszero(parsimonious(k, :))
        cls = 2;
    elseif ~iszero(at_optimum(k, :))
        cls = 3;
    elseif ~iszero(any_growth(k, :))
        cls = 4;
    else
        cls = 5;           % the core model has no blocked reactions
    end
    members{cls}{end + 1} = rxns{internal(k)};
end
fprintf(['\nE. coli core on glucose: growth %.4f /h; minimal total flux ' ...
    '%.2f mmol/gDW/h\n'], mu, min_total);
for k = 1:6
    fprintf('  %-12s %3d  %s\n', labels{k}, numel(members{k}), ...
        strjoin(members{k}, ', '));
end

%% Plot
fig = figure('Position', [100 100 700 350]);
counts = cellfun(@numel, members);
barh(categorical(labels, fliplr(labels)), counts, ...
    'FaceColor', [0.122 0.467 0.706]);
text(counts + 0.3, categorical(labels, fliplr(labels)), ...
    string(counts), 'VerticalAlignment', 'middle');
xlabel('Number of internal reactions');
title('pFBA classes, E. coli core on glucose');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-pfba');

results.toy_max_b2 = best;
results.toy_fva = toy_fva;
results.toy_pfba = toy_pfba;
results.toy_min_total = toy_min_total;
results.growth = mu;
results.min_total_flux = min_total;
for k = 1:6
    results.classes.(fields{k}) = members{k};
end
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function f = lp(c, S, lb, ub, A, b)
% Minimum of c'v with S v = 0, A v <= b and the bounds; [] if infeasible.
opts = optimoptions('linprog', 'Display', 'none');
[~, f, exitflag] = linprog(c, A, b, S, zeros(size(S, 1), 1), lb, ub, opts);
if exitflag ~= 1, f = []; end
end

function [S2, lb2, ub2, A, b] = with_abs(S, lb, ub)
% Add variables t_j >= |v_j|: for the variables [v; t], the inequality
% rows are v - t <= 0 and -v - t <= 0.
[m, n] = size(S);
S2 = [S zeros(m, n)];
lb2 = [lb; zeros(n, 1)]; ub2 = [ub; 1000 * ones(n, 1)];
A = [eye(n) -eye(n); -eye(n) -eye(n)];
b = zeros(2 * n, 1);
end

function r = fva(S, lb, ub, A, b, which)
% Range of each of the variables listed in which.
r = zeros(numel(which), 2);
for k = 1:numel(which)
    c = zeros(size(S, 2), 1); c(which(k)) = 1;
    r(k, :) = [lp(c, S, lb, ub, A, b), -lp(-c, S, lb, ub, A, b)];
end
end
