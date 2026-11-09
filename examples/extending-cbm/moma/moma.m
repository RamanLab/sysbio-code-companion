%% Predicting the fluxes of a mutant: FBA and MoMA
% FBA assumes that a mutant, like the wild type, grows as fast as its
% network allows. MoMA (minimisation of metabolic adjustment) assumes that
% a freshly made mutant stays as close as possible to the wild-type flux
% distribution w: it minimises the distance ||v - w||^2, subject to the
% same constraints S v = 0 and lb <= v <= ub, with the deleted reaction
% fixed at zero. That is a quadratic programme (QP). We write both
% problems out with the stoichiometric matrix, solve them with linprog and
% quadprog, and compare FBA and MoMA for every deletion of an internal
% reaction of the E. coli core model.
% Requires: Optimization Toolbox (the COBRA Toolbox is not needed).

s = load(fullfile(sbp.data_dir(), 'e_coli_core.mat'));
model = s.e_coli_core;
S = model.S; lb = model.lb; ub = model.ub; ids = model.rxns;
n = numel(ids);
bio = find(strcmp(ids, 'Biomass_Ecoli_core'));

% Wild type. Many flux distributions achieve the maximum growth rate; as
% reference we take the one with the smallest Euclidean norm, which is
% unique.
mu_wt = fba(S, lb, ub, bio);
[l, u] = at_growth(mu_wt, lb, ub, bio);
w = nearest(S, zeros(n, 1), l, u);
fprintf('Wild-type growth rate %.4f /h\n', mu_wt);

%% Every deletion of an internal reaction
internal = ~startsWith(ids, {'EX_', 'Biomass'});
rxns = sort(ids(internal));
growth_fba = zeros(1, numel(rxns)); growth_moma = zeros(1, numel(rxns));
for k = 1:numel(rxns)
    j = strcmp(ids, rxns{k});
    lb_d = lb; ub_d = ub;
    lb_d(j) = 0; ub_d(j) = 0;
    mu = fba(S, lb_d, ub_d, bio);
    if isempty(mu)          % no flux distribution meets the constraints,
        continue            % e.g. the ATP maintenance demand: growth 0
    end
    growth_fba(k) = mu / mu_wt;
    if abs(w(j)) < 1e-9       % w already meets the new constraints, so
        v = w;                % it is its own nearest point
    else
        v = nearest(S, w, lb_d, ub_d);
    end
    growth_moma(k) = v(bio) / mu_wt;
end
fprintf(['%d deletions; MoMA predicts slower growth than FBA for %d ' ...
    'of them\n'], numel(rxns), nnz(growth_moma < growth_fba - 0.01));
for r = {'PGI', 'PFK', 'TPI', 'G6PDH2r', 'PYK', 'CS'}
    k = strcmp(rxns, r{1});
    fprintf('  %-8s FBA %.3f, MoMA %.3f of wild-type growth\n', r{1}, ...
        growth_fba(k), growth_moma(k));
end

%% Fluxes after deleting phosphofructokinase (PFK)
show = {'PGI', 'PFK', 'FBA', 'G6PDH2r', 'GND', 'TKT1', 'TALA', 'GAPD', 'PYK'};
[~, idx] = ismember(show, ids);
lb_d = lb; ub_d = ub;
lb_d(strcmp(ids, 'PFK')) = 0; ub_d(strcmp(ids, 'PFK')) = 0;
[l, u] = at_growth(fba(S, lb_d, ub_d, bio), lb_d, ub_d, bio);
v_fba = nearest(S, zeros(n, 1), l, u);
v_moma = nearest(S, w, lb_d, ub_d);
pfk.wild_type = w(idx)'; pfk.fba = v_fba(idx)'; pfk.moma = v_moma(idx)';

%% Plot
fig = figure('Position', [100 100 1000 420]);
subplot(1, 5, 1:2)
plot([0 1.05], [0 1.05], ':', 'Color', [0.6 0.6 0.6]);
hold on
plot(growth_fba, growth_moma, 'o', 'Color', [0.122 0.467 0.706], ...
    'MarkerFaceColor', [0.122 0.467 0.706]);
xlabel('FBA (fraction of wild type)'); ylabel('MoMA (fraction of wild type)');
title('Growth after 74 deletions');
box off
subplot(1, 5, 3:5)
b = bar(categorical(show, show), [pfk.wild_type; pfk.fba; pfk.moma]');
b(1).FaceColor = [0.6 0.6 0.6];
b(2).FaceColor = [1.000 0.498 0.055];
b(3).FaceColor = [0.122 0.467 0.706];
ylabel('Flux (mmol/gDW/h)'); title('Fluxes after deleting PFK');
legend({'wild type', 'FBA', 'MoMA'}, 'Box', 'off');
box off
sbp.save_figure(fig, mfilename('fullpath'), 'fig-moma');

results.growth_wild_type = mu_wt;
results.reactions = rxns';
results.growth_fba = growth_fba;
results.growth_moma = growth_moma;
results.pfk_reactions = show;
results.pfk_fluxes = pfk;
sbp.save_results(mfilename('fullpath'), results);

%% Local functions
function mu = fba(S, lb, ub, bio)
% Maximum growth rate, or [] if the constraints cannot be met.
c = zeros(size(lb)); c(bio) = -1;              % linprog minimises
opts = optimoptions('linprog', 'Display', 'none');
[~, f, exitflag] = linprog(c, [], [], S, zeros(size(S, 1), 1), lb, ub, ...
    opts);
mu = [];
if exitflag == 1
    mu = max(0, -f);
end
end

function v = nearest(S, target, lb, ub)
% The flux vector closest to target: minimise ||v - target||^2 / 2, that
% is v'v/2 - target'v, subject to S v = 0 and lb <= v <= ub. Some rows of
% S are linear combinations of others; dropping them, and using the
% active-set algorithm, makes quadprog exact and reliable here.
n = numel(lb);
S = full(S);
[~, R, order] = qr(S', 'vector');               % independent rows of S
independent = order(abs(diag(R)) > 1e-9 * abs(R(1, 1)));
S = S(sort(independent), :);
opts = optimoptions('quadprog', 'Algorithm', 'active-set', ...
    'Display', 'none');
x0 = min(max(target, lb), ub);                  % a starting point
[v, ~, exitflag, output] = quadprog(eye(n), -target, [], [], S, ...
    zeros(size(S, 1), 1), lb, ub, x0, opts);
assert(exitflag > 0, 'quadprog stopped (exit flag %d): %s', exitflag, ...
    output.message);
end

function [lb, ub] = at_growth(mu, lb, ub, bio)
% Bounds that hold growth at (just below) mu.
lb(bio) = mu * (1 - 1e-9);
ub(bio) = mu;
end
